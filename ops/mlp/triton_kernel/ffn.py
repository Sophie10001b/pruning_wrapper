import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

from ops.utils import get_autotune_config, get_autotune_cache

#########################
# Kernel Implementation
#########################

def bm_sort_ffn_impl(
    x: tl.tensor, # [M, K]
    route_mask: tl.tensor, # [M] or [cdiv(M, BLOCK_M)]
    route_indices: tl.tensor, # [M]
    wu: tl.tensor, # [N, K]
    wg: tl.tensor, # [N, K]
    wd: tl.tensor, # [K, N]
    bu: tl.tensor,
    bg: tl.tensor,
    out: tl.tensor,
    M: tl.int64,
    N: tl.int64,
    K: tl.int64,
    HAS_BIAS_UP: tl.constexpr,
    HAS_BIAS_GATE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    IS_OFFLINE: tl.constexpr,
):
    tmid, tnid = tl.program_id(0), tl.program_id(1)
    BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

    # compute indices in groups
    mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if IS_OFFLINE:
        skip_flag = tl.load(route_mask + mid)
    else:
        query_mask = tl.load(
            route_mask + mid * BLOCK_M + tl.arange(0, BLOCK_M),
            mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
            other=0,
        )
        skip_flag = tl.reduce_or(query_mask, axis=-1)

    if skip_flag > 0:
        if IS_OFFLINE:
            bm_indices = tl.load(
                route_indices + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                other=-1,
            )
            bm_mask = bm_indices >= 0
            bm_indices = tl.where(bm_mask, bm_indices, 0)
        else:
            bm_mask = query_mask > 0
            bm_indices = tl.load(
                route_indices + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=bm_mask,
                other=0,
            )

        bm_offset = bm_indices * K
        bn_offset = nid * BLOCK_N + tl.arange(0, BLOCK_N)
        bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
        bk_offset = tl.arange(0, BLOCK_K)[None, :]
        
        # GLU computation
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

        # Down proj with split-k atomic add
        bk_offset = tl.arange(0, BLOCK_K)
        for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
            wd_data = tl.load(
                wd + bk_offset[:, None] * N + bn_offset[None, :],
                mask=bk_offset[:, None] < K,
                other=0,
            )
            acc_down = tl.dot(acc_up.to(x.dtype.element_ty), wd_data.T, out_dtype=tl.float32)

            tl.atomic_add(
                out + bm_offset[:, None] + bk_offset[None, :],
                acc_down.to(x.dtype.element_ty),
                mask=bm_mask[:, None],
                sem='relaxed',
            )

            bk_offset += BLOCK_K


def bn_sort_ffn_impl(
    x: tl.tensor, # [M, K]
    route_mask: tl.tensor, # [NG, M] or [NG, cdiv(M, BLOCK_M)]
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
    IS_OFFLINE: tl.constexpr,
):
    tmid, tnid, gid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

    # compute indices in groups
    mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if IS_OFFLINE:
        skip_flag = tl.load(route_mask + nid * BLOCK_NUM_M + mid)
    else:
        query_mask = tl.load(
            route_mask + nid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
            mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
            other=0,
        )
        skip_flag = tl.reduce_or(query_mask, axis=-1)

    if skip_flag > 0:
        if IS_OFFLINE:
            bm_indices = tl.load(
                route_indices + nid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                other=-1,
            )
            bm_mask = bm_indices >= 0
            bm_indices = tl.where(bm_mask, bm_indices, 0)
        else:
            bm_mask = query_mask > 0
            bm_indices = tl.load(
                route_indices + nid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=query_mask,
                other=0,
            )

        bm_offset = bm_indices * K
        bn_offset = (nid * G_iter + gid) * BLOCK_N + tl.arange(0, BLOCK_N)
        bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
        bk_offset = tl.arange(0, BLOCK_K)[None, :]
        
        # GLU computation
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

        # Down proj with split-k atomic add
        bk_offset = tl.arange(0, BLOCK_K)
        for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
            wd_data = tl.load(
                wd + bk_offset[:, None] * N + bn_offset[None, :],
                mask=bk_offset[:, None] < K,
                other=0,
            )
            acc_down = tl.dot(acc_up.to(x.dtype.element_ty), wd_data.T, out_dtype=tl.float32)

            tl.atomic_add(
                out + bm_offset[:, None] + bk_offset[None, :],
                acc_down.to(x.dtype.element_ty),
                mask=bm_mask[:, None],
                sem='relaxed',
            )

            bk_offset += BLOCK_K


#########################
# Host Implementation
#########################

class BMSparseFFN:

    support_kernel = [
        'dense',
        'sort_offline',
        'sort_online',
    ]

    @classmethod
    def _dense_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
        **kwargs
    ):
        # GLU
        out_up = torch.nn.functional.linear(x, wu, bu)
        out_gate = torch.nn.functional.linear(x, wg, bg)
        activation = kwargs.get('activation', 'identity')
        if activation == 'silu':
            out_gate = out_gate * torch.sigmoid(out_gate)
        elif activation == 'relu':
            out_gate = torch.nn.functional.relu(out_gate)
        out_glu = out_up * out_gate
        
        # Down proj
        out = torch.nn.functional.linear(out_glu, wd)
        return out.masked_fill(route_mask.logical_not()[:, :, None], 0)
    
    @classmethod
    def _sort_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        GROUP_SIZE: Optional[int]=4,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
        
        B, L, D = x.shape

        M = B * L
        N = wu.shape[0]
        K = D

        x_flat = x.reshape((M, D))
        m_sort = kwargs.get('m_sort', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)

        if m_sort_indices is None:
            m_flat = route_mask.flatten(0, 1)
            m_sort, m_sort_indices = torch.sort(m_flat, descending=True, stable=False)

            if kwargs.get('is_offline', False):
                # offline calculate skipping
                m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)
                m_sort = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(-1, BLOCK_M)
                m_sort = m_sort.any(dim=-1)
        
        out = torch.zeros((M, K), dtype=x.dtype, device=x.device)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'GROUP_SIZE', 'num_stages'],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            **kwargs,
        )
        kernel = get_autotune_cache(
            bm_sort_ffn_impl,
            enable_autotune=True,
            config=config,
            keys=['M', 'N', 'K'],
        )
        kernel[grid](
            x_flat, m_sort, m_sort_indices,
            wu, wg, wd, bu, bg, out,
            M, N, K,
            HAS_BIAS_UP=bu is not None,
            HAS_BIAS_GATE=bg is not None,
            ACTIVATION=kwargs.get('activation', 'identity'),
            IS_OFFLINE=kwargs.get('is_offline', False),
        )

        return rearrange(out, '(B L) K -> B L K', B=B, K=K), dict(
            m_sort=m_sort,
            m_sort_indices=m_sort_indices,
        )

    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for BM sparse FFN (fused GLU + down proj), where x = [M, K]\\
        **sort:**\\
        Return the same shape as input, with additional sort prologue for better skipping\\
        Down projection uses split-k style atomic add for aggregation.
        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            route_mask: torch.Tensor with shape [batch_size, seqlen], 1 for active token
            wu: torch.Tensor with shape [intermediate_size, hidden_size], weight of W_up
            wg: torch.Tensor with shape [intermediate_size, hidden_size], weight of W_gate
            wd: torch.Tensor with shape [hidden_size, intermediate_size], weight of W_down
            bu: torch.Tensor with shape [hidden_size], bias of W_up
            bg: torch.Tensor with shape [hidden_size], bias of W_gate
            sorted_mask: optional, torch.Tensor with shape [batch_size, seqlen], 1 for active token, sorted
            sorted_indices: optinal, torch.Tensor with shape [batch_size, seqlen], index of sorted route_mask
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
            if route_mask is None: impl = 'dense'
            else: impl = 'sort_offline'

        if impl in ['sort_offline', 'sort_online']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M >= int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            BLOCK_N = min(128, max(16, triton.next_power_of_2(N)))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
            BLOCK_K = min(64, max(32, triton.next_power_of_2(K)))

            while BLOCK_N * BLOCK_K > 128 * 128 and BLOCK_K > 32:
                BLOCK_K = BLOCK_K >> 1

            while BLOCK_M * BLOCK_N * BLOCK_K > 128 * 64 * 64 and BLOCK_K > 32:
                BLOCK_K = BLOCK_K >> 1
            
            BLOCK_K = kwargs.pop('BLOCK_K', BLOCK_K)
        else:
            BLOCK_M = -1
            BLOCK_N = -1
            BLOCK_K = -1
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        is_offline = impl == 'sort_offline'
        impl = impl.split('_')[0]
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=group_size,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            is_offline=is_offline,
            BLOCK_M_list=[16, 32, 64, 128],
            BLOCK_N_list=[32, 64, 128],
            BLOCK_K_list=[32, 64],
            num_stages_list=[2, 3],
            **kwargs
        )


class BNSparseFFN:

    support_kernel = [
        'dense',
        'sort_offline',
        'sort_online',
    ]

    @classmethod
    def _dense_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
        **kwargs
    ):
        # GLU
        out_up = torch.nn.functional.linear(x, wu, bu)
        out_gate = torch.nn.functional.linear(x, wg, bg)
        activation = kwargs.get('activation', 'identity')
        if activation == 'silu':
            out_gate = out_gate * torch.sigmoid(out_gate)
        elif activation == 'relu':
            out_gate = torch.nn.functional.relu(out_gate)
        out_glu = out_up * out_gate
        
        # Down proj
        out = torch.nn.functional.linear(out_glu, wd)
        out = rearrange(out, 'b l (h d) -> b l h d', h=route_mask.shape[-1]).masked_fill(route_mask.logical_not()[:, :, :, None], 0)
        return rearrange(out, 'b l h d -> b l (h d)')
    
    @classmethod
    def _sort_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        wu: torch.Tensor,
        wg: torch.Tensor,
        wd: torch.Tensor,
        bu: Optional[torch.Tensor]=None,
        bg: Optional[torch.Tensor]=None,
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

        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = wu.shape[0]
        K = D
        G = N // NG

        x_flat = x.reshape((M, D))
        m_sort = kwargs.get('m_sort', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)

        if m_sort_indices is None:
            m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
            m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)

            if kwargs.get('is_offline', False):
                # offline calculate skipping
                m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)
                m_sort = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(NG, -1, BLOCK_M)
                m_sort = m_sort.any(dim=-1)

        out = torch.zeros((M, K), dtype=x.dtype, device=x.device)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), NG, G_iter)

        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'GROUP_SIZE', 'num_stages'],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            **kwargs,
        )
        kernel = get_autotune_cache(
            bn_sort_ffn_impl,
            enable_autotune=True,
            config=config,
            keys=['M', 'N', 'K'],
        )
        kernel[grid](
            x_flat, m_sort, m_sort_indices,
            wu, wg, wd, bu, bg, out,
            M, N, K,
            HAS_BIAS_UP=bu is not None,
            HAS_BIAS_GATE=bg is not None,
            ACTIVATION=kwargs.get('activation', 'identity'),
            G_iter=G_iter,
            IS_OFFLINE=kwargs.get('is_offline', False),
        )
        return rearrange(out, '(B L) K -> B L K', B=B), dict(
            m_sort=m_sort,
            m_sort_indices=m_sort_indices,
        )
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for BN sparse FFN (fused GLU + down proj), where x = [M, K]\\
        **sort:**\\
        Return the same shape as input, with additional sort prologue for better skipping\\
        Down projection uses split-k style atomic add for aggregation.
        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            route_mask: torch.Tensor with shape [batch_size, seqlen, num_groups], 1 for active BN
            wu: torch.Tensor with shape [intermediate_size, hidden_size], weight of W_up
            wg: torch.Tensor with shape [intermediate_size, hidden_size], weight of W_gate
            wd: torch.Tensor with shape [hidden_size, intermediate_size], weight of W_down
            bu: torch.Tensor with shape [hidden_size], bias of W_up
            bg: torch.Tensor with shape [hidden_size], bias of W_gate
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
            if route_mask is None: impl = 'dense'
            else: impl = 'sort_offline'
        
        NG = route_mask.shape[-1]
        G = N // NG
        BLOCK_N = G

        while BLOCK_N > 128: BLOCK_N = BLOCK_N >> 1
        G_iter = G // BLOCK_N

        if impl in ['sort_offline', 'sort_online']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M >= int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_K = min(64, max(32, triton.next_power_of_2(K)))

            while (BLOCK_M * BLOCK_K >= 128 * 128) or (BLOCK_N * BLOCK_K >= 128 * 128):
                BLOCK_K = BLOCK_K >> 1

            while BLOCK_M * BLOCK_N * BLOCK_K > 128 * 64 * 64 and BLOCK_K > 32:
                BLOCK_K = BLOCK_K >> 1
            
            BLOCK_K = kwargs.pop('BLOCK_K', BLOCK_K)
        else:
            BLOCK_M = -1
            BLOCK_K = -1
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        is_offline = impl == 'sort_offline'
        impl = impl.split('_')[0]
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=group_size,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            is_offline=is_offline,
            BLOCK_M_list=[16, 32, 64, 128],
            BLOCK_K_list=[32, 64],
            num_stages_list=[2, 3],
            **kwargs
        )