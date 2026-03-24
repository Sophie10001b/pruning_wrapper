import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.blackwell import (tma, mbarrier, fence_async_shared)

from ops.utils import get_autotune_config, get_autotune_cache, check_shared_memory_attn

RCP_LN2 = gl.constexpr(1.4426950408889634)

# SM120 gather & scatter fuse flash attention, with TMA gather4 load Q, K, V for blhd layout
# KV with 2 buffer, from https://github.com/triton-lang/kernels/pull/20
# also apply bitmask for generating causal mask, from https://zhuanlan.zhihu.com/p/2011582362362864169

@aggregate
class AttentionConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    HEAD_DIM: gl.constexpr
    num_warp: gl.constexpr

    mma_layout: gl.constexpr
    rQ_layout: gl.constexpr
    rK_layout: gl.constexpr
    rP_layout: gl.constexpr
    rV_layout: gl.constexpr

    sQ_layout: gl.constexpr
    sK_layout: gl.constexpr
    sV_layout: gl.constexpr

    dtype: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_M, BLOCK_N, HEAD_DIM, num_wrap, dtype):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.num_warp = gl.constexpr(num_wrap)
        self.dtype = gl.constexpr(dtype)

        self.mma_layout = gl.constexpr(gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[1, self.num_warp],
            instr_shape=[16, 8],
        ))
        self.rQ_layout = gl.constexpr(gl.DotOperandLayout(
            operand_index=0,
            parent=self.mma_layout,
            k_width=16,
        ))
        self.rK_layout = gl.constexpr(gl.DotOperandLayout(
            operand_index=1,
            parent=self.mma_layout,
            k_width=16,
        ))
        self.rP_layout = gl.constexpr(gl.DotOperandLayout(
            operand_index=0,
            parent=self.mma_layout,
            k_width=16,
        ))
        self.rV_layout = gl.constexpr(gl.DotOperandLayout(
            operand_index=1,
            parent=self.mma_layout,
            k_width=16,
        ))

        self.sQ_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([self.BLOCK_M, self.HEAD_DIM], self.dtype))
        self.sK_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([self.BLOCK_N, self.HEAD_DIM], self.dtype))
        self.sV_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([self.BLOCK_N, self.HEAD_DIM], self.dtype))


@gluon.jit
def _mask_scalar(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s        # 当前s组内可见的列数
    col_lim_right_cur = max(col_lim_right_s, 0)  # 至少为 0
    mask = -1 << col_lim_right_cur               # 构造位掩码
    mask_i_bit = (mask & (1 << i)) == 0          # 检查第 i 位是否为 0
    return gl.where(mask_i_bit, qk, -float("inf"))

@gluon.jit
def _apply_causal_mask(qk, col_limit_right):
    # from https://zhuanlan.zhihu.com/p/2011582362362864169
    offs_n = gl.arange(0, qk.shape[1])[None, :]  # [0, 1, 2, ..., 127]
    s = offs_n & ~0xf
    i = offs_n & 0xf
    return gl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)


@gluon.jit
def issue_load_KV(
    producer: gl.int64,
    mK: gl.tensor,
    mV: gl.tensor,
    sK: gl.shared_memory_descriptor,
    sV: gl.shared_memory_descriptor,
    mK_offset,
    mask: gl.tensor,
    HEAD_DIM: gl.constexpr,
    num_stages: gl.constexpr,
):
    index = producer % num_stages
    cp.async_copy_global_to_shared(
        sK.index(index),
        mK + mK_offset[:, None] * HEAD_DIM + gl.arange(0, HEAD_DIM)[None, :],
        mask=mask[:, None]
    )
    cp.async_copy_global_to_shared(
        sV.index(index),
        mV + mK_offset[:, None] * HEAD_DIM + gl.arange(0, HEAD_DIM)[None, :],
        mask=mask[:, None]
    )
    cp.commit_group()

    return producer + 1

@gluon.jit
def issue_mma_QK(
    consumer: gl.int64,
    rQ: gl.tensor,
    sK: gl.shared_memory_descriptor,
    sK_bar: gl.shared_memory_descriptor,
    rK_layout: gl.constexpr,
    rP: gl.tensor,
    qk_scale: gl.float32,
    num_stages: gl.constexpr,
):
    index = consumer % num_stages
    phase = consumer // num_stages & 1
    k_bar = sK_bar.index(index)

    mbarrier.wait(k_bar, phase)
    rK = sK.index(index).permute((1, 0)).load(rK_layout)
    rP = mma_v2(rQ, rK, rP)
    return consumer, rP * qk_scale

@gluon.jit
def issue_mma_PV(
    consumer: gl.int64,
    rP: gl.tensor,
    sV: gl.shared_memory_descriptor,
    sV_bar: gl.shared_memory_descriptor,
    rV_layout: gl.constexpr,
    acc: gl.tensor,
    num_stages: gl.constexpr,
):
    index = consumer % num_stages
    phase = consumer // num_stages & 1
    v_bar = sV_bar.index(index)

    mbarrier.wait(v_bar, phase)
    rV = sV.index(index).load(rV_layout)
    acc = mma_v2(rP, rV, acc)
    return consumer + 1, acc


def query_sort_prefill_sm120_impl(
    mQ, mK, mV,
    mO,
    route_mask: gl.tensor, # [B, LQ], only use sort_online
    route_indices: gl.tensor, # [B, LQ]
    pad_offset: gl.tensor, # [B]
    SEQ_LEN_Q: gl.int64,
    SEQ_LEN_K: gl.int64,
    HEAD_NUM_Q: gl.constexpr,
    HEAD_NUM_K: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    sm_scale: gl.float32,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    # block [bsz, num_heads_q, seq_block]
    tidx, tidy, tidz = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    HEAD_GROUP_NUM: gl.constexpr = HEAD_NUM_Q // HEAD_NUM_K
    mQ_base = tidx * HEAD_NUM_Q * SEQ_LEN_Q + tidy
    mK_base = tidx * HEAD_NUM_K * SEQ_LEN_K + (tidy // HEAD_GROUP_NUM)

    dtype: gl.constexpr = mQ.dtype.element_ty
    cfg = AttentionConfig(BLOCK_M, BLOCK_N, HEAD_DIM, gl.num_warps(), dtype)

    # set shared memory
    sQ = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], cfg.sQ_layout)
    sK = gl.allocate_shared_memory(dtype, [2, BLOCK_N, HEAD_DIM], cfg.sK_layout)
    sV = gl.allocate_shared_memory(dtype, [2, BLOCK_N, HEAD_DIM], cfg.sV_layout)

    # producer-consumer for QK
    producer: gl.int64 = 0
    consumer: gl.int64 = 0

    # online softmax reg state
    qk_scale = sm_scale * RCP_LN2
    accum_max = gl.full([BLOCK_M], -float('inf'), dtype=gl.float32, layout=gl.SliceLayout(1, cfg.mma_layout))
    accum_sum = gl.full([BLOCK_M], 0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.mma_layout))
    acc = gl.zeros([BLOCK_M, HEAD_DIM], dtype=gl.float32, layout=cfg.mma_layout)

    gather_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [cfg.num_warp, 1], [1, 0])
    mK_offset = mK_base + gl.arange(0, BLOCK_N, layout=gather_layout) * HEAD_NUM_K
    
    # route mask handle
    rMask = gl.load(
        route_mask + tidx * SEQ_LEN_Q + tidz * BLOCK_M + gl.arange(0, BLOCK_M, layout=gather_layout),
        mask=tidz * BLOCK_M + gl.arange(0, BLOCK_M, layout=gather_layout) < SEQ_LEN_Q,
        other=0,
    )
    is_execute = gl.reduce_or(rMask, axis=0)

    if is_execute:
        # get gather index
        rIndices = gl.load(
            route_indices + tidx * SEQ_LEN_Q + tidz * BLOCK_M + gl.arange(0, BLOCK_M, layout=gather_layout),
            mask=rMask > 0,
            other=-1,
        )
        rIndices_mma = gl.convert_layout(rIndices, gl.SliceLayout(1, cfg.mma_layout))
        mQ_offset = mQ_base + rIndices * HEAD_NUM_Q
        mQ_offset = gl.where(rMask > 0, mQ_offset, mQ_desc.shape[0])

        rPad = gl.load(pad_offset + tidx)

        # load Q, and KV's 1st pipeline
        cp.async_copy_global_to_shared(
            sQ,
            mQ + (tidx * SEQ_LEN_Q * HEAD_NUM_Q + rIndices[:, None] * HEAD_NUM_Q + tidy) * HEAD_DIM + gl.arange(0, HEAD_DIM)[None, :],
            mask=(rMask > 0)[:, None]
        )
        cp.commit_group()

        n_start = rPad
        n_end = min(SEQ_LEN_K, gl.max(rIndices) + 1)
        m_min = gl.min(gl.where(rMask > 0, rIndices, mQ_desc.shape[0])) + 1
        cp.wait_group(0)

        producer = issue_load_KV(producer, mK_desc, mV_desc, sK, sV, sK_bar, sV_bar, (mK_offset + rPad * HEAD_NUM_K).to(gl.int32), BLOCK_N, 2)
        mK_offset += rPad * HEAD_NUM_K

        
        # wait until Q arrive
        mbarrier.wait(sQ_bar, 0)
        mbarrier.invalidate(sQ_bar)
        rQ = sQ.load(cfg.rQ_layout)

        for n in range(n_start, n_end, BLOCK_N):
            if n + BLOCK_N < n_end:
                producer = issue_load_KV(producer, mK_desc, mV_desc, sK, sV, sK_bar, sV_bar, (mK_offset + producer * BLOCK_N * HEAD_NUM_K).to(gl.int32), BLOCK_N, 2)
            
            # issue QK MMA
            rP = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=cfg.mma_layout)
            consumer, rP = issue_mma_QK(consumer, rQ, sK, sK_bar, cfg.rK_layout, rP, qk_scale, 2)

            rP = gl.where((n + gl.arange(0, BLOCK_N) < SEQ_LEN_K)[None, :], rP, float("-inf"))
            if n + BLOCK_N > m_min: # causal mask
                # col_limit_right = (rIndices_mma + 1 - n)[:, None]
                # rP = _apply_causal_mask(rP, col_limit_right.to(gl.int32))
                rP = gl.where(rIndices_mma[:, None] >= n + gl.arange(0, BLOCK_N)[None, :], rP, float("-inf"))

            # online softmax update
            new_max = gl.maximum(accum_max, gl.max(rP, axis=1))
            accum_scale = gl.exp2(accum_max - new_max)
            rP = gl.exp2(rP - new_max[:, None])
            accum_sum = accum_sum * accum_scale + gl.sum(rP, axis=1)
            acc = acc * accum_scale[:, None]
            accum_max = new_max

            # issue PV MMA
            rP_16 = rP.to(dtype, fp_downcast_rounding='rtz')
            rP_16 = gl.convert_layout(rP_16, cfg.rP_layout)
            consumer, acc = issue_mma_PV(consumer, rP_16, sV, sV_bar, cfg.rV_layout, acc, 2)
        
        for i in gl.static_range(2):
            mbarrier.invalidate(sK_bar.index(i))
            mbarrier.invalidate(sV_bar.index(i))

        # correction
        acc = acc / accum_sum[:, None]

        # write back
        store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [cfg.num_warp, 1], [1, 0])
        mQ_offset = gl.convert_layout(mQ_offset, gl.SliceLayout(1, store_layout))
        acc_r2g = gl.convert_layout(acc, store_layout)
        gl.store(
            mO + (mQ_offset * HEAD_DIM)[:, None] + gl.arange(0, HEAD_DIM)[None, :],
            acc_r2g,
            mask=(mQ_offset < mQ_desc.shape[0])[:, None],
        )


#########################
# Host Implementation
#########################

class QuerySparsePrefill:

    support_kernel = [
        'sort_online',
    ]
    
    @classmethod    
    def _sort_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=2,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        B, SEQ_LEN_Q, HEAD_NUM_Q, HEAD_DIM = q.shape
        _, SEQ_LEN_K, HEAD_NUM_K, _ = k.shape

        m_sort, m_sort_indices = torch.sort(route_mask, descending=True, stable=False) # [B, LQ]

        out = torch.zeros_like(q)
        grid = lambda meta: (B, HEAD_NUM_Q, triton.cdiv(SEQ_LEN_Q, meta['BLOCK_M']))

        # tma descriptor
        gl_dtype = getattr(gl, str(q.dtype).split('.')[1])

        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N'],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            **kwargs,
        )
        kernel = get_autotune_cache(
            query_sort_prefill_sm120_impl,
            enable_autotune=True,
            config=config,
            keys=['B', 'LQ', 'LK'],
            do_not_specialize=['qk_scale'],
            is_gluon=True,
        )
        kernel[grid](
            q, k, v, out,
            m_sort, m_sort_indices.to(torch.int64),
            pad_offset.to(torch.int64),
            SEQ_LEN_Q, SEQ_LEN_K,
            HEAD_NUM_Q, HEAD_NUM_K, HEAD_DIM,
            sm_scale=HEAD_DIM ** -0.5,
        )
        return out
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for query sparse attention (prefill)\\
        **ragged:**\\
        Return [nnz, num_query_heads, head_dim]\\
        **dense & sort**\\
        Return [batch_size, query_length, num_query_heads, head_dim]
        Args:
            q, k, v: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
            route_mask: Optional[torch.Tensor] with shape [batch_size, seqlen], with 1 for active tokens
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        
        q = kwargs.get('q')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        route_mask = kwargs.pop('route_mask', None)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]
        num_stages = 2
        num_warps = 4

        if impl == 'auto': impl = 'sort_online'

        estimated_sparsity = kwargs.pop('estimated_sparsity', 1)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, LQ), dtype=torch.bool, device=device)
        
        if estimated_sparsity == 0: estimated_sparsity = 1

        BLOCK_M = triton.next_power_of_2(int(LQ * estimated_sparsity))
        if BLOCK_M >= int(LQ * estimated_sparsity): BLOCK_M = BLOCK_M >> 1
        BLOCK_M = min(128, max(16, BLOCK_M))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LQ)))

        assert D <= 128, f"sm80 style flash_attn head_dim {D} must be <= 128"

        while not check_shared_memory_attn(BLOCK_M, BLOCK_N, D, num_stages, dtype.itemsize):
            if BLOCK_N > 32: BLOCK_N >>= 1
            elif BLOCK_M > 32: BLOCK_M >>= 1
            else: num_stages -= 1
            
        while HQ * B * triton.cdiv(LQ, BLOCK_M) < num_sm:
            if BLOCK_M > 16: BLOCK_M >>= 1
        
        BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)

        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        is_offline = impl == 'sort_offline'
        impl = impl.split('_')[0]
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            pad_offset=pad_offset,
            is_offline=is_offline,
            BLOCK_M_list=[16, 32, 64, 128],
            BLOCK_N_list=[32, 64],
            num_stages_list=[2],
            **kwargs
        )