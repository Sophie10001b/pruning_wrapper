import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

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

class BMSparseMLP:

    support_kernel = [
        'dense',
        'indexing',
        'sort_online',
        'sort_offline',
    ]

    @classmethod
    def _dense_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        **kwargs
    ):
        out = torch.nn.functional.linear(x, w, b).masked_fill(route_mask.logical_not()[:, :, None], 0)
        return out
    
    @classmethod
    def _indexing_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        **kwargs
    ):
        indices = torch.nonzero(route_mask.flatten()).flatten()
        res = torch.zeros((x.shape[0] * x.shape[1], w.shape[0]), dtype=x.dtype, device=x.device)
        tmp = x.flatten(0, 1).index_select(dim=0, index=indices)
        out = torch.nn.functional.linear(tmp, w, b)
        res[indices] = out
        return rearrange(res, '(b l) d -> b l d', b=x.shape[0])

    @classmethod
    def _sort_offline_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
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
        def mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [M]
            w: tl.tensor, # [N, K]
            b: tl.tensor,
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE: tl.constexpr,
        ):
            tmid, tnid = tl.program_id(0), tl.program_id(1)
            BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

            # compute indices in groups
            # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
            mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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
                    w_data = tl.load(
                        w + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc = tl.dot(x_data, w_data.T, acc=acc)

                    bk_offset += BLOCK_K
                
                if HAS_BIAS:
                    acc += (tl.load(b + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                tl.store(
                    out + bm_indices[:, None] * N + bn_offset[None, :],
                    acc.to(x.dtype.element_ty),
                    mask=bm_mask[:, None] & (bn_offset < N)[None, :],
                )

        B, L, D = x.shape

        M = B * L
        N = w.shape[0]
        K = D

        x_flat = x.reshape((M, D))
        m_sort_pad = kwargs.get('m_sort_pad', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)

        if m_sort_indices is None:
            m_flat = route_mask.flatten(0, 1)
            m_sort, m_sort_indices = torch.sort(m_flat, descending=True, stable=False)
        
            # offline calculate skipping
            m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(-1, BLOCK_M)
            m_sort_pad = m_sort_pad.any(dim=-1)
            m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        mlp[grid](
            x_flat, m_sort_pad, m_sort_indices,
            w, b, out,
            M, N, K,
            HAS_BIAS=b is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) N -> B L N', B=B, N=N), dict(
            m_sort_pad=m_sort_pad,
            m_sort_indices=m_sort_indices,
        )
    
    @classmethod
    def _sort_online_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
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
        def mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [M]
            route_indices: tl.tensor, # [M]
            w: tl.tensor, # [N, K]
            b: tl.tensor,
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE: tl.constexpr,
        ):
            tmid, tnid = tl.program_id(0), tl.program_id(1)
            BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

            # compute indices in groups
            # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
            mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            query_mask = tl.load(
                route_mask + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                other=0,
            )
            skip_flag = tl.reduce_or(query_mask, axis=-1)
            if skip_flag > 0:
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
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    w_data = tl.load(
                        w + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc = tl.dot(x_data, w_data.T, acc=acc)

                    bk_offset += BLOCK_K
                
                if HAS_BIAS:
                    acc += (tl.load(b + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                tl.store(
                    out + bm_indices[:, None] * N + bn_offset[None, :],
                    acc.to(x.dtype.element_ty),
                    mask=bm_mask[:, None] & (bn_offset < N)[None, :],
                )

        B, L, D = x.shape

        M = B * L
        N = w.shape[0]
        K = D

        x_flat = x.reshape((M, D))
        m_sort = kwargs.get('m_sort', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)

        if m_sort_indices is None:
            m_flat = route_mask.flatten(0, 1)
            m_sort, m_sort_indices = torch.sort(m_flat, descending=True, stable=False)

        out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        mlp[grid](
            x_flat, m_sort, m_sort_indices,
            w, b, out,
            M, N, K,
            HAS_BIAS=b is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) N -> B L N', B=B, N=N), dict(
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
        Get kernel for token sparse MLP, where x = [M, K], weight = [N, K]\\
        **sort:**\\
        Return the same shape as input, with additional sort prologue for better skipping
        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            route_mask: torch.Tensor with shape [batch_size, seqlen], 1 for active token
            w: torch.Tensor with shape [intermediate_size, hidden_size], weight
            b: torch.Tensor with shape [hidden_size], bias
            sorted_mask: optional, torch.Tensor with shape [batch_size, seqlen], 1 for active token, sorted
            sorted_indices: optinal, torch.Tensor with shape [batch_size, seqlen], index of sorted route_mask
            estimated_sparsity: optinal, float, estimated sparsity of route_mask
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """        
        x = kwargs.get('x')
        w = kwargs.get('w')
        dtype = x.dtype
        device = x.device

        B, L, D = x.shape
        M, N, K = B * L, w.shape[0], D
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
            if BLOCK_M > int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            BLOCK_N = min(128, max(16, triton.next_power_of_2(N)))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
            BLOCK_K = min(128, max(32, triton.next_power_of_2(K)))

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


class BNSparseMLP:

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
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        **kwargs
    ):
        out = torch.nn.functional.linear(x, w, b)
        out = rearrange(out, 'b l (h d) -> b l h d', h=route_mask.shape[-1]).masked_fill(route_mask.logical_not()[:, :, :, None], 0)
        return rearrange(out, 'b l h d -> b l (h d)')
    
    @classmethod
    def _sort_offline_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
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
        def mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [NG, cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [NG, M]
            w: tl.tensor, # [N, K]
            b: tl.tensor,
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            G_iter: tl.constexpr,
            HAS_BIAS: tl.constexpr,
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

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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
                    w_data = tl.load(
                        w + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc = tl.dot(x_data, w_data.T, acc=acc)

                    bk_offset += BLOCK_K
                
                if HAS_BIAS:
                    acc += (tl.load(b + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                tl.store(
                    out + bm_indices[:, None] * N + bn_offset[None, :],
                    acc.to(x.dtype.element_ty),
                    mask=bm_mask[:, None],
                )

        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = w.shape[0]
        K = D
        G = N // NG

        x_flat = x.reshape((M, D))
        m_sort_pad = kwargs.get('m_sort_pad', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)

        if m_sort_indices is None:
            m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
            m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)
            # offline calculate skipping
            m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(NG, -1, BLOCK_M)
            m_sort_pad = m_sort_pad.any(dim=-1)
            m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), NG, G_iter)
        mlp[grid](
            x_flat, m_sort_pad, m_sort_indices,
            w, b, out,
            M, N, K,
            HAS_BIAS=b is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) N -> B L N', B=B), dict(
            m_sort_pad=m_sort_pad,
            m_sort_indices=m_sort_indices,
        )
    
    @classmethod
    def _sort_online_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
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
        def mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [NG, cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [NG, M]
            w: tl.tensor, # [N, K]
            b: tl.tensor,
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            G_iter: tl.constexpr,
            HAS_BIAS: tl.constexpr,
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

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            query_mask = tl.load(
                route_mask + nid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                other=0,
            )
            skip_flag = tl.reduce_or(query_mask, axis=-1)
            if skip_flag > 0:
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
                for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    w_data = tl.load(
                        w + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc = tl.dot(x_data, w_data.T, acc=acc)

                    bk_offset += BLOCK_K
                
                if HAS_BIAS:
                    acc += (tl.load(b + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
                
                tl.store(
                    out + bm_indices[:, None] * N + bn_offset[None, :],
                    acc.to(x.dtype.element_ty),
                    mask=bm_mask[:, None],
                )

        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = w.shape[0]
        K = D
        G = N // NG

        x_flat = x.reshape((M, D))
        m_sort = kwargs.get('m_sort', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)

        if m_sort_indices is None:
            m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
            m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)

        out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), NG, G_iter)
        mlp[grid](
            x_flat, m_sort, m_sort_indices,
            w, b, out,
            M, N, K,
            HAS_BIAS=b is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) N -> B L N', B=B), dict(
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
        Get kernel for token-aware BN sparse MLP, where x = [M, K], weight = [N, K]\\
        **sort:**\\
        Return the same shape as input, with additional sort prologue for better skipping
        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            route_mask: torch.Tensor with shape [batch_size, seqlen, num_groups], 1 for active BN
            w: torch.Tensor with shape [intermediate_size, hidden_size], weight
            b: torch.Tensor with shape [hidden_size], bias
            estimated_sparsity: optinal, float, estimated sparsity of route_mask
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """        
        x = kwargs.get('x')
        w = kwargs.get('w')
        dtype = x.dtype
        device = x.device

        B, L, D = x.shape
        M, N, K = B * L, w.shape[0], D
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

        while BLOCK_N > 64: BLOCK_N = BLOCK_N >> 1
        G_iter = G // BLOCK_N

        if impl in ['sort_offline', 'sort_online']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M > int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_K = min(128, max(32, triton.next_power_of_2(K)))

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
            

class BKSparseMLP:

    support_kernel = [
        'atomic_online',
        'atomic_offline',
        'reduce',
        'dense',
    ]

    @classmethod
    def _dense_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        **kwargs
    ):
        out = rearrange(x, 'b l (h d) -> b l h d', h=route_mask.shape[-1]).masked_fill(route_mask.logical_not()[:, :, :, None], 0)
        out = torch.nn.functional.linear(out, w, b)
        return rearrange(out, 'b l h d -> b l (h d)')

    @classmethod
    def _atomic_offline_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
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
        def mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [NG, cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [NG, M]
            w: tl.tensor, # [N, K]
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            G_iter: tl.constexpr,
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

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            skip_flag = tl.load(
                route_mask + gid * BLOCK_NUM_M + mid,
            )
            if skip_flag > 0:
                bm_indices = tl.load(
                    route_indices + gid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                    other=-1,
                )
                bm_mask = bm_indices >= 0
                bm_indices = tl.where(bm_mask, bm_indices, 0)

                bm_offset = bm_indices * K
                bn_offset = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
                bk_offset = gid * G_iter * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
                for i in tl.range(0, G_iter):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    w_data = tl.load(
                        w + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc = tl.dot(x_data, w_data.T, acc=acc)

                    bk_offset += BLOCK_K
                
                tl.atomic_add(
                    out + bm_indices[:, None] * N + bn_offset[None, :],
                    acc.to(x.dtype.element_ty),
                    mask=bm_mask[:, None],
                    sem='relaxed',
                )

        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = w.shape[0]
        K = D
        G = K // NG

        x_flat = x.reshape((M, D))
        m_sort_pad = kwargs.get('m_sort_pad', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)

        if m_sort_indices is None:
            m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
            m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)
            # offline calculate skipping
            m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(NG, -1, BLOCK_M)
            m_sort_pad = m_sort_pad.any(dim=-1)
            m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), NG)
        mlp[grid](
            x_flat, m_sort_pad, m_sort_indices,
            w, out,
            M, N, K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) N -> B L N', B=B), dict(
            m_sort_pad=m_sort_pad,
            m_sort_indices=m_sort_indices,
        )
    
    @classmethod
    def _atomic_online_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
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
        def mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [NG, M]
            route_indices: tl.tensor, # [NG, M]
            w: tl.tensor, # [N, K]
            out: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            G_iter: tl.constexpr,
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

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            query_mask = tl.load(
                route_mask + gid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                other=0,
            )
            skip_flag = tl.reduce_or(query_mask, axis=-1)
            if skip_flag > 0:
                bm_mask = query_mask > 0
                bm_indices = tl.load(
                    route_indices + gid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=bm_mask,
                    other=0,
                )

                bm_offset = bm_indices * K
                bn_offset = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
                bk_offset = gid * G_iter * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
                for i in tl.range(0, G_iter):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    w_data = tl.load(
                        w + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc = tl.dot(x_data, w_data.T, acc=acc)

                    bk_offset += BLOCK_K
                
                tl.atomic_add(
                    out + bm_indices[:, None] * N + bn_offset[None, :],
                    acc.to(x.dtype.element_ty),
                    mask=bm_mask[:, None],
                    sem='relaxed',
                )

        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = w.shape[0]
        K = D
        G = K // NG

        x_flat = x.reshape((M, D))
        m_sort = kwargs.get('m_sort', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)

        if m_sort_indices is None:
            m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
            m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)

        out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), NG)
        mlp[grid](
            x_flat, m_sort, m_sort_indices,
            w, out,
            M, N, K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return rearrange(out, '(B L) N -> B L N', B=B), dict(
            m_sort=m_sort,
            m_sort_indices=m_sort_indices,
        )

    @classmethod
    def _reduce_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
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
        def mlp(
            x: tl.tensor, # [M, K]
            route_mask: tl.tensor, # [NG, cdiv(M, BLOCK_M)]
            route_indices: tl.tensor, # [NG, M]
            w: tl.tensor, # [N, K]
            out_partial: tl.tensor,
            M: tl.int64,
            N: tl.constexpr,
            K: tl.constexpr,
            G_iter: tl.constexpr,
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

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            skip_flag = tl.load(
                route_mask + gid * BLOCK_NUM_M + mid,
            )
            if skip_flag > 0:
                bm_indices = tl.load(
                    route_indices + gid * M + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                    other=-1,
                )
                bm_mask = bm_indices >= 0
                bm_indices = tl.where(bm_mask, bm_indices, 0)

                bm_offset = bm_indices * K
                bn_offset = nid * BLOCK_N + tl.arange(0, BLOCK_N)
                bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
                bk_offset = gid * G_iter * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
                for i in tl.range(0, G_iter):
                    x_data = tl.load(
                        x + bm_offset[:, None] + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )
                    w_data = tl.load(
                        w + bn_offset[:, None] * K + bk_offset,
                        mask=bk_offset < K,
                        other=0,
                    )

                    acc = tl.dot(x_data, w_data.T, acc=acc)

                    bk_offset += BLOCK_K
                
                tl.store(
                    out_partial + gid * M * N + bm_indices[:, None] * N + bn_offset[None, :],
                    acc.to(x.dtype.element_ty),
                    mask=bm_mask[:, None],
                )

        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = w.shape[0]
        K = D
        G = K // NG

        x_flat = x.reshape((M, D))
        m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
        m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(NG, -1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out_partial = torch.zeros((NG, M, N), dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), NG)
        mlp[grid](
            x_flat, m_sort_pad, m_sort_indices,
            w, out_partial,
            M, N, K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            G_iter=G_iter,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        out = out_partial.sum(dim=0)
        return rearrange(out, '(B L) N -> B L N', B=B)
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for token-aware BK sparse MLP, usually for MoE down proj\\
        **atomic:**\\
        Using split-k style atomic add for online K-axis reduction
        **reduce:**\\
        Using explicit memory to save partial sum in BK block
        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            route_mask: torch.Tensor with shape [batch_size, seqlen, num_groups], 1 for active BK
            w: torch.Tensor with shape [intermediate_size, hidden_size], weight
            b: torch.Tensor with shape [hidden_size], bias
            estimated_sparsity: optinal, float, estimated sparsity of route_mask
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """        
        x = kwargs.get('x')
        w = kwargs.get('w')
        dtype = x.dtype
        device = x.device

        B, L, D = x.shape
        M, N, K = B * L, w.shape[0], D
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
            BLOCK_K = 32
            NG = K // BLOCK_K
            route_mask = torch.ones((B, L, NG), dtype=torch.bool, device=device)
        
        if impl == 'auto': # auto dispatch
            if cc < 9 and dtype == torch.bfloat16: impl = 'reduce'
            else: impl = 'atomic_offline'
        
        NG = route_mask.shape[-1]
        G = K // NG
        BLOCK_K = G

        while BLOCK_K > 64: BLOCK_K = BLOCK_K >> 1
        G_iter = G // BLOCK_K

        if impl in ['atomic_offline', 'atomic_online', 'reduce']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M > int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_N = min(128, max(32, triton.next_power_of_2(N)))

            while BLOCK_M * BLOCK_N * BLOCK_K > 128 * 64 * 64 and BLOCK_N > 32:
                BLOCK_N = BLOCK_N >> 1
            
            BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
        elif impl == 'small_bsz':
            BLOCK_M = 16

            BLOCK_N = min(128, max(32, triton.next_power_of_2(N)))

            while BLOCK_N * BLOCK_K > 128 * 64 and BLOCK_N > 64:
                BLOCK_N = BLOCK_N >> 1
            
            BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
        else:
            BLOCK_M = -1
            BLOCK_N = -1
        
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