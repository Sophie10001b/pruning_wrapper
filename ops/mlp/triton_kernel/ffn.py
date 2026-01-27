import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor
from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

def generate_autotune_config(tuning_dict: Dict):
    keys = list(tuning_dict.keys())
    value_lists = [tuning_dict[key] for key in keys]
    
    combinations = []
    for value_combination in itertools.product(*value_lists):
        combination_dict = dict(zip(keys, value_combination))
        num_stages = combination_dict.pop('num_stages', 3)
        num_warps = combination_dict.pop('num_warps', 4)
        combinations.append(triton.Config(combination_dict, num_stages=num_stages, num_warps=num_warps))
    
    return combinations

class BMSparseGLUBMSparseMLP:

    support_kernel = [
        'atomic',
        'reduce',
    ]

    @classmethod
    def _atomic_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
        activation: Optional[str]='identity',
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        GROUP_SIZE: Optional[int]=4,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def glu_fuse_splitk_mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [M]
            wu: tl.tensor, # [N, K]
            wg: tl.tensor, # [N, K]
            wd: tl.tensor, # [K, N]
            bu: tl.tensor,
            bg: tl.tensor,
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            HAS_BIAS_UP: tl.constexpr,
            HAS_BIAS_GATE: tl.constexpr,
            ACTIVATION: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE: tl.constexpr,
        ):
            tmid, tnid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

            # compute indices in groups
            # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
            mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

            acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            skip_flag = tl.load(
                route_mask + mid,
            )
            if skip_flag > 0:
                bm_indices = tl.load(
                    route_indices + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                    other=-1,
                )
                bm_mask = bm_indices >= 0
                bm_indices = tl.where(bm_mask, bm_indices, 0)

                bm_offset = bm_indices * K
                bn_offset = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
                bk_offset = tl.arange(0, BLOCK_K)[None, :]
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    wu_data = tl.load(
                        wu + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    wg_data = tl.load(
                        wg + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc_up = tl.dot(x_data, wu_data.T, acc=acc_up)
                    acc_gate = tl.dot(x_data, wg_data.T, acc=acc_gate)

                    bk_offset += BLOCK_K
                
                if HAS_BIAS_GATE:
                    acc_gate += (tl.load(bg + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                if HAS_BIAS_UP:
                    acc_up += (tl.load(bu + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                if ACTIVATION == 'silu':
                    acc_gate *= tl.sigmoid(acc_gate)
                elif ACTIVATION == 'relu':
                    acc_gate = tl.maximum(acc_gate, 0.0)
                
                acc_up *= acc_gate

                # calculate down_proj across all K-axis
                bk_offset = tl.arange(0, BLOCK_K)
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    wd_data = tl.load(
                        wd + bk_offset[:, None] * N + bn_offset[None, :],
                        mask=bk_offset[:, None] < K,
                        other=0,
                    )
                    acc_down = tl.dot(acc_up.to(x.dtype.element_ty), wd_data.T, out_dtype=tl.float32) # [BM, BK]

                    tl.atomic_add(
                        out + bm_offset[:, None] + bk_offset[None, :],
                        acc_down.to(x.dtype.element_ty),
                        mask=bm_mask[:, None],
                        sem='relaxed',
                    )

                    bk_offset += BLOCK_K
        
        B, L, D = x.shape

        M = B * L
        N = wu.shape[0]
        K = D

        x_flat = x.reshape((M, D))
        m_sort = kwargs.get('sorted_mask', None)
        m_sort_indices = kwargs.get('sorted_indices', None)

        if m_sort is None:
            m_flat = route_mask.flatten(0, 1)
            m_sort, m_sort_indices = torch.sort(m_flat, descending=True, stable=False)
        
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(-1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = torch.zeros((M, K), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        glu_fuse_splitk_mlp[grid](
            x_flat, m_sort_pad, m_sort_indices,
            wu, wg, wd, bu, bg, out,
            M, N, K,
            HAS_BIAS_UP=bu is not None,
            HAS_BIAS_GATE=bg is not None,
            ACTIVATION=activation,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) K -> B L K', B=B)

    @classmethod
    def _reduce_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
        activation: Optional[str]='identity',
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        GROUP_SIZE: Optional[int]=4,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def glu_fuse_partial_mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [M]
            wu: tl.tensor, # [N, K]
            wg: tl.tensor, # [N, K]
            wd: tl.tensor, # [K, N]
            bu: tl.tensor,
            bg: tl.tensor,
            out_partial: tl.tensor, # [N // BLOCK_N, M, K]
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            HAS_BIAS_UP: tl.constexpr,
            HAS_BIAS_GATE: tl.constexpr,
            ACTIVATION: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE: tl.constexpr,
        ):
            tmid, tnid, gid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

            # compute indices in groups
            # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
            mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

            acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            skip_flag = tl.load(
                route_mask + mid,
            )
            if skip_flag > 0:
                bm_indices = tl.load(
                    route_indices + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                    other=-1,
                )
                bm_mask = bm_indices >= 0
                bm_indices = tl.where(bm_mask, bm_indices, 0)

                bm_offset = bm_indices * K
                bn_offset = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
                bk_offset = tl.arange(0, BLOCK_K)[None, :]
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    wu_data = tl.load(
                        wu + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    wg_data = tl.load(
                        wg + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc_up = tl.dot(x_data, wu_data.T, acc=acc_up)
                    acc_gate = tl.dot(x_data, wg_data.T, acc=acc_gate)

                    bk_offset += BLOCK_K
                
                if HAS_BIAS_GATE:
                    acc_gate += (tl.load(bg + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                if HAS_BIAS_UP:
                    acc_up += (tl.load(bu + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                if ACTIVATION == 'silu':
                    acc_gate *= tl.sigmoid(acc_gate)
                elif ACTIVATION == 'relu':
                    acc_gate = tl.maximum(acc_gate, 0.0)
                
                acc_up *= acc_gate

                # calculate down_proj across all K-axis
                bk_offset = tl.arange(0, BLOCK_K)
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    wd_data = tl.load(
                        wd + bk_offset[:, None] * N + bn_offset[None, :],
                        mask=bk_offset[:, None] < K,
                        other=0,
                    )
                    acc_down = tl.dot(acc_up.to(x.dtype.element_ty), wd_data.T, out_dtype=tl.float32) # [BM, BK]
                    tl.store(
                        out_partial + nid * M * K + bm_offset[:, None] + bk_offset[None, :],
                        acc_down.to(x.dtype.element_ty),
                        mask=bm_mask[:, None],
                    )

                    bk_offset += BLOCK_K
        
        B, L, D = x.shape

        M = B * L
        N = wu.shape[0]
        K = D
        NG = N // BLOCK_N

        x_flat = x.reshape((M, D))
        m_sort = kwargs.get('sorted_mask', None)
        m_sort_indices = kwargs.get('sorted_indices', None)

        if m_sort is None:
            m_flat = route_mask.flatten(0, 1)
            m_sort, m_sort_indices = torch.sort(m_flat, descending=True, stable=False)
        
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(-1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out_partial = torch.zeros((NG, M, K), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        glu_fuse_partial_mlp[grid](
            x_flat, m_sort_pad, m_sort_indices,
            wu, wg, wd, bu, bg, out_partial,
            M, N, K,
            HAS_BIAS_UP=bu is not None,
            HAS_BIAS_GATE=bg is not None,
            ACTIVATION=activation,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        out = out_partial.sum(dim=0)
        return rearrange(out, '(B L) K -> B L K', B=B)
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for fused FFN (M-axis sparse for GLU and M-axis sparse for down_proj)\\
        **atomic:**\\
        Using split-k style atomic add for down projection, **BF16 & vector type atomic add only support in sm90+**\\
        **reduce:**\\
        Explicitly reduce the partial results for down projection.\\
        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            route_mask: torch.Tensor with shape [batch_size, seqlen], 1 for active token
            wu: torch.Tensor with shape [intermediate_size, hidden_size], weight of W_up
            wg: torch.Tensor with shape [intermediate_size, hidden_size], weight of W_gate
            wd: torch.Tensor with shape [hidden_size, intermediate_size], weight of W_down
            bu: torch.Tensor with shape [hidden_size], bias of W_up
            bg: torch.Tensor with shape [hidden_size], bias of W_gate
            activation: str in ['silu', 'relu']
            estimated_sparsity: optinal, float, estimated sparsity of route_mask
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        x = kwargs.get('x')
        wu = kwargs.get('wu')
        dtype = x.dtype
        device = x.device

        B, L, D = x.shape
        M, N, K = B * L, wu.shape[0], D
        route_mask = kwargs.pop('route_mask', None)
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]
        num_stages = 3
        num_warps = 4
        group_size = 4

        estimated_sparsity = kwargs.pop('estimated_sparsity', 0)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        
        if impl == 'auto': # auto dispatch
            if cc < 9 and dtype == torch.bfloat16: impl = 'reduce'
            else: impl = 'atomic'

        if impl in ['atomic', 'reduce']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M > int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            BLOCK_N = min(128, max(16, triton.next_power_of_2(N)))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
            BLOCK_K = min(128, max(64, triton.next_power_of_2(K)))

            while (BLOCK_M * BLOCK_K >= 128 * 128) or (BLOCK_N * BLOCK_K >= 128 * 128):
                BLOCK_K = BLOCK_K >> 1

            # BK >= 64 is required for fast split-k style ffn
            while BLOCK_M * BLOCK_N * BLOCK_K > 128 * 64 * 32 and BLOCK_N > 32:
                BLOCK_N = BLOCK_N >> 1
            
            BLOCK_K = kwargs.pop('BLOCK_K', BLOCK_K)

        else:
            BLOCK_M = -1
            BLOCK_N = -1
            BLOCK_K = -1
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=group_size,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            **kwargs
        )


class BNSparseGLUBKSparseMLP:

    support_kernel = [
        'atomic',
        'reduce',
        'atomic_gather',
    ]
    
    @classmethod
    def _atomic_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
        activation: Optional[str]='identity',
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        G_iter: Optional[int]=1,
        GROUP_SIZE: Optional[int]=4,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def glu_fuse_splitk_mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [NG, cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [NG, M]
            wu: tl.tensor, # [N, K]
            wg: tl.tensor, # [N, K]
            wd: tl.tensor, # [K, N]
            bu: tl.tensor,
            bg: tl.tensor,
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            G_iter: tl.constexpr,
            HAS_BIAS_UP: tl.constexpr,
            HAS_BIAS_GATE: tl.constexpr,
            ACTIVATION: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE: tl.constexpr,
        ):
            tmid, tnid, gid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

            # compute indices in groups
            # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
            mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

            acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            skip_flag = tl.load(
                route_mask + nid * BLOCK_NUM_M + mid,
            )
            if skip_flag > 0:
                bm_indices = tl.load(
                    route_indices + nid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                    other=-1,
                )
                bm_mask = bm_indices >= 0
                bm_indices = tl.where(bm_mask, bm_indices, 0)

                bm_offset = bm_indices * K
                bn_offset = (nid * G_iter + gid) * BLOCK_N + tl.arange(0, BLOCK_N)
                bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
                bk_offset = tl.arange(0, BLOCK_K)[None, :]
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    wu_data = tl.load(
                        wu + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    wg_data = tl.load(
                        wg + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc_up = tl.dot(x_data, wu_data.T, acc=acc_up)
                    acc_gate = tl.dot(x_data, wg_data.T, acc=acc_gate)

                    bk_offset += BLOCK_K
                
                if HAS_BIAS_GATE:
                    acc_gate += (tl.load(bg + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                if HAS_BIAS_UP:
                    acc_up += (tl.load(bu + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                if ACTIVATION == 'silu':
                    acc_gate *= tl.sigmoid(acc_gate)
                elif ACTIVATION == 'relu':
                    acc_gate = tl.maximum(acc_gate, 0.0)
                
                acc_up *= acc_gate

                # calculate down_proj across all K-axis
                bk_offset = tl.arange(0, BLOCK_K)
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    wd_data = tl.load(
                        wd + bk_offset[:, None] * N + bn_offset[None, :],
                        mask=bk_offset[:, None] < K,
                        other=0,
                    )
                    acc_down = tl.dot(acc_up.to(x.dtype.element_ty), wd_data.T, out_dtype=tl.float32) # [BM, BK]

                    tl.atomic_add(
                        out + bm_offset[:, None] + bk_offset[None, :],
                        acc_down.to(x.dtype.element_ty),
                        mask=bm_mask[:, None],
                        sem='relaxed',
                    )

                    bk_offset += BLOCK_K
        
        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = wu.shape[0]
        K = D
        G = N // NG

        x_flat = x.reshape((M, D))
        m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
        m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(NG, -1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = torch.zeros((M, K), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), NG, G_iter)
        glu_fuse_splitk_mlp[grid](
            x_flat, m_sort_pad, m_sort_indices,
            wu, wg, wd, bu, bg, out,
            M, N, K,
            HAS_BIAS_UP=bu is not None,
            HAS_BIAS_GATE=bg is not None,
            ACTIVATION=activation,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) K -> B L K', B=B)
    
    @classmethod
    def _atomic_gather_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
        activation: Optional[str]='identity',
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        G_iter: Optional[int]=1,
        GROUP_SIZE: Optional[int]=4,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def glu_fuse_splitk_mlp(
            x: tl.tensor_descriptor, # [M, K]
            route_mask: tl.tensor, # [NG, cdiv(M, BLOCK_M)]
            route_indices: tl.tensor_descriptor, # [NG, M]
            wu: tl.tensor_descriptor, # [N, K]
            wg: tl.tensor_descriptor, # [N, K]
            wd: tl.tensor_descriptor, # [K, N]
            bu: tl.tensor,
            bg: tl.tensor,
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            G_iter: tl.constexpr,
            HAS_BIAS_UP: tl.constexpr,
            HAS_BIAS_GATE: tl.constexpr,
            ACTIVATION: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE: tl.constexpr,
        ):
            tmid, tnid, gid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

            # compute indices in groups
            # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
            mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

            acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            skip_flag = tl.load(route_mask + nid * BLOCK_NUM_M + mid)
            if skip_flag > 0:
                bm_indices = route_indices.load([nid, mid * BLOCK_M])
                bm_indices = tl.reshape(bm_indices, (BLOCK_M,))
                bm_mask = bm_indices >= 0
                bm_indices = tl.where(bm_mask, bm_indices, 0)
                
                bn_offset = (nid * G_iter + gid) * BLOCK_N + tl.arange(0, BLOCK_N)
                bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    x_data = x.gather(bm_indices, i * BLOCK_K)
                    wu_data = wu.load([(nid * G_iter + gid) * BLOCK_N, i * BLOCK_K])
                    wg_data = wg.load([(nid * G_iter + gid) * BLOCK_N, i * BLOCK_K])

                    acc_up = tl.dot(x_data, wu_data.T, acc=acc_up)
                    acc_gate = tl.dot(x_data, wg_data.T, acc=acc_gate)
                
                if HAS_BIAS_GATE:
                    acc_gate += (tl.load(bg + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                if HAS_BIAS_UP:
                    acc_up += (tl.load(bu + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                if ACTIVATION == 'silu':
                    acc_gate *= tl.sigmoid(acc_gate)
                elif ACTIVATION == 'relu':
                    acc_gate = tl.maximum(acc_gate, 0.0)
                
                acc_up *= acc_gate

                # calculate down_proj across all K-axis
                bk_offset = tl.arange(0, BLOCK_K)
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    wd_data = wd.load([i * BLOCK_K, (nid * G_iter + gid) * BLOCK_N])
                    acc_down = tl.dot(acc_up.to(x.dtype), wd_data.T, out_dtype=tl.float32) # [BM, BK]
                    tl.atomic_add(
                        out + bm_indices[:, None] * K + bk_offset[None, :],
                        acc_down.to(x.dtype),
                        mask=bm_mask[:, None],
                        sem='relaxed',
                    )

                    bk_offset += BLOCK_K
        
        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = wu.shape[0]
        K = D
        G = N // NG

        x_flat = x.reshape((M, D))
        m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
        m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(NG, -1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)
        m_sort_indices_pad = torch.nn.functional.pad(m_sort_indices.to(torch.int32), (0, BLOCK_M - M % BLOCK_M), value=-1)

        out = torch.zeros((M, K), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), NG, G_iter)

        # make TMA descriptors on host side
        x_tma = TensorDescriptor.from_tensor(x_flat, [1, BLOCK_K])
        indices_tma = TensorDescriptor.from_tensor(m_sort_indices_pad, [1, BLOCK_M])
        wu_tma = TensorDescriptor.from_tensor(wu, [BLOCK_N, BLOCK_K])
        wg_tma = TensorDescriptor.from_tensor(wg, [BLOCK_N, BLOCK_K])
        wd_tma = TensorDescriptor.from_tensor(wd, [BLOCK_K, BLOCK_N])

        glu_fuse_splitk_mlp[grid](
            x_tma, m_sort_pad, indices_tma,
            wu_tma, wg_tma, wd_tma, bu, bg, out,
            M, N, K,
            HAS_BIAS_UP=bu is not None,
            HAS_BIAS_GATE=bg is not None,
            ACTIVATION=activation,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) K -> B L K', B=B)

    @classmethod
    def _reduce_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
        activation: Optional[str]='identity',
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        G_iter: Optional[int]=1,
        GROUP_SIZE: Optional[int]=4,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def glu_fuse_partial_mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [NG, cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [NG, M]
            wu: tl.tensor, # [N, K]
            wg: tl.tensor, # [N, K]
            wd: tl.tensor, # [K, N]
            bu: tl.tensor,
            bg: tl.tensor,
            out_partial: tl.tensor, # [NG, M, K]
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            G_iter: tl.constexpr,
            HAS_BIAS_UP: tl.constexpr,
            HAS_BIAS_GATE: tl.constexpr,
            ACTIVATION: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE: tl.constexpr,
        ):
            tmid, tnid, gid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

            # compute indices in groups
            # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
            mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

            acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            skip_flag = tl.load(
                route_mask + nid * BLOCK_NUM_M + mid,
            )
            if skip_flag > 0:
                bm_indices = tl.load(
                    route_indices + nid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                    other=-1,
                )
                bm_mask = bm_indices >= 0
                bm_indices = tl.where(bm_mask, bm_indices, 0)

                bm_offset = bm_indices * K
                bn_offset = (nid * G_iter + gid) * BLOCK_N + tl.arange(0, BLOCK_N)
                bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)

                bk_offset = tl.arange(0, BLOCK_K)[None, :]
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    wu_data = tl.load(
                        wu + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    wg_data = tl.load(
                        wg + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc_up = tl.dot(x_data, wu_data.T, acc=acc_up)
                    acc_gate = tl.dot(x_data, wg_data.T, acc=acc_gate)

                    bk_offset += BLOCK_K
                
                if HAS_BIAS_GATE:
                    acc_gate += (tl.load(bg + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                if HAS_BIAS_UP:
                    acc_up += (tl.load(bu + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                if ACTIVATION == 'silu':
                    acc_gate *= tl.sigmoid(acc_gate)
                elif ACTIVATION == 'relu':
                    acc_gate = tl.maximum(acc_gate, 0.0)
                
                acc_up *= acc_gate

                # calculate down_proj across all K-axis
                bk_offset = tl.arange(0, BLOCK_K)
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    wd_data = tl.load(
                        wd + bk_offset[:, None] * N + bn_offset[None, :],
                        mask=bk_offset[:, None] < K,
                        other=0,
                    )
                    acc_down = tl.dot(acc_up.to(x.dtype.element_ty), wd_data.T, out_dtype=tl.float32) # [BM, BK]
                    tl.store(
                        out_partial + (nid * G_iter + gid) * M * K + bm_offset[:, None] + bk_offset[None, :],
                        acc_down.to(x.dtype.element_ty),
                        mask=bm_mask[:, None],
                    )

                    bk_offset += BLOCK_K
        
        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = wu.shape[0]
        K = D
        G = N // NG

        x_flat = x.reshape((M, D))
        m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
        m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(NG, -1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out_partial = torch.zeros((NG * G_iter, M, K), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), NG, G_iter)
        glu_fuse_partial_mlp[grid](
            x_flat, m_sort_pad, m_sort_indices,
            wu, wg, wd, bu, bg, out_partial,
            M, N, K, G_iter,
            HAS_BIAS_UP=bu is not None,
            HAS_BIAS_GATE=bg is not None,
            ACTIVATION=activation,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        out = out_partial.sum(dim=0)
        return rearrange(out, '(B L) K -> B L K', B=B)
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for MoE-like FFN (N-axis sparse for GLU and K-axis sparse for down_proj)\\
        **atomic:**\\
        Using split-k style atomic add for down projection, **BF16x2 atomic add only support in sm90+**\\
        **atomic_gather:**\\
        Using split-k style atomic add for down projection, TMA gather4 is applied for load\\
        **reduce:**\\
        Explicitly reduce the partial results for down projection.\\
        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            route_mask: torch.Tensor with shape [batch_size, seqlen, num_groups], 1 for active BN
            wu: torch.Tensor with shape [intermediate_size, hidden_size], weight of W_up
            wg: torch.Tensor with shape [intermediate_size, hidden_size], weight of W_gate
            wd: torch.Tensor with shape [hidden_size, intermediate_size], weight of W_down
            bu: torch.Tensor with shape [hidden_size], bias of W_up
            bg: torch.Tensor with shape [hidden_size], bias of W_gate
            activation: str in ['silu', 'relu']
            estimated_sparsity: optinal, float, estimated sparsity of route_mask
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        x = kwargs.get('x')
        wu = kwargs.get('wu')
        dtype = x.dtype
        device = x.device

        B, L, D = x.shape
        M, N, K = B * L, wu.shape[0], D
        route_mask = kwargs.pop('route_mask', None)
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]
        num_stages = 3
        num_warps = 4
        group_size = 4

        estimated_sparsity = kwargs.pop('estimated_sparsity', 0)
        if route_mask is None:
            estimated_sparsity = 1
            BLOCK_N = 32
            NG = N // BLOCK_N
            route_mask = torch.ones((B, L, NG), dtype=torch.bool, device=device)
        
        if impl == 'auto': # auto dispatch
            if cc < 9 and dtype == torch.bfloat16:
                impl = 'reduce'
            else: impl = 'atomic'
        
        NG = route_mask.shape[-1]
        G = N // NG
        BLOCK_N = G

        while BLOCK_N > 64: BLOCK_N = BLOCK_N >> 1
        G_iter = G // BLOCK_N

        if impl in ['atomic', 'atomic_gather', 'reduce']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M > int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_K = min(128, max(32, triton.next_power_of_2(K)))

            while (BLOCK_M * BLOCK_K >= 128 * 128) or (BLOCK_N * BLOCK_K >= 128 * 128):
                BLOCK_K = BLOCK_K >> 1

            while BLOCK_M * BLOCK_N * BLOCK_K > 128 * 64 * 32 and BLOCK_K > 32:
                BLOCK_K = BLOCK_K >> 1
            
            BLOCK_K = kwargs.pop('BLOCK_K', BLOCK_K)
        else:
            BLOCK_M = -1
            BLOCK_K = -1
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=group_size,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            **kwargs
        )