import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl
import tilelang
import tilelang.language as T

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

class BNSparseMLP:

    support_kernel = [
        'dense',
        'sort',
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
    def _sort_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        ACTIVATION: Optional[str]='identity',
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        G_iter: Optional[int]=1,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        **kwargs
    ):
        @tilelang.jit
        def kernel(
            x, route_mask, route_indices, w, b,
            NG, G_iter, HAS_BIAS, ACTIVATION, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_threads,
            dtype, acc_dtype, index_dtype,
        ):
            M = T.dynamic('M')
            N, K, NG = T.const('N, K, NG')

            x: T.Tensor[[M, K], dtype]
            route_mask: T.Tensor[[NG, T.ceildiv(M, BLOCK_M)], index_dtype]
            route_indices: T.Tensor[[NG, M], index_dtype]
            w: T.Tensor[[N, K], dtype]
            b: T.Tensor[[N], acc_dtype]

            out = T.empty([M, N], dtype)

            with T.Kernel(T.ceildiv(M, BLOCK_M), NG, G_iter, threads=num_threads) as (bx, by, bz):
                S_x = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
                S_w = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)

                R_out = T.alloc_fragment([BLOCK_M, BLOCK_N], acc_dtype)
                R_out_cast = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
                R_b = T.alloc_fragment([BLOCK_N], acc_dtype)
                R_indices = T.alloc_fragment([BLOCK_M], index_dtype)
                R_indices_valid = T.alloc_fragment([BLOCK_M], index_dtype)

                T.clear(R_out)
                T.use_swizzle(panel_size=10)
                if route_mask[by, bx]:
                    for k in T.Pipelined(0, T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                        T.copy(route_indices[by, bx * BLOCK_M:(bx + 1) * BLOCK_M], R_indices)
                        for i in T.Parallel(BLOCK_M, BLOCK_N):
                            R_indices_valid[i] = T.if_then_else(R_indices[i] < 0, 0, R_indices[i])
                            T.copy(x[R_indices_valid[i], k * BLOCK_K], S_x[i, :])
                        
                        T.copy(w[(by * G_iter + bz) * BLOCK_N, k * BLOCK_K], S_w)
                        T.gemm(S_x, S_w, R_out, transpose_B=True)
                    
                    if HAS_BIAS:
                        T.copy(b[(by * G_iter + bz) * BLOCK_N], R_b)
                        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                            R_out[i, j] += R_b[j]
                    
                    if ACTIVATION == 'silu':
                        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                            R_out[i, j] *= T.sigmoid(R_out[i, j])
                    
                    if ACTIVATION == 'relu':
                        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                            R_out[i, j] = T.max(R_out[i, j], 0.0)
                    
                    T.copy(R_out, R_out_cast)
                    for i in range(BLOCK_M):
                        if R_indices[i] >= 0: T.copy(R_out_cast[i, :], out[R_indices_valid[i], (by * G_iter + bz) * BLOCK_N])
            
            return out
        
        B, L, D = x.shape
        _, _, NG = route_mask.shape

        M = B * L
        N = w.shape[0]
        K = D
        G = N // NG

        x_flat = x.reshape((M, D))
        m_trans_flat = route_mask.flatten(0, 1).transpose(0, 1).contiguous()
        m_sort, m_sort_indices = torch.sort(m_trans_flat, dim=-1, descending=True, stable=False)
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(NG, -1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = kernel(
            x_flat, m_sort_pad, m_sort_indices, w, b,
            NG, G_iter, b is not None, ACTIVATION, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps * 32,
            str(x_flat.dtype).split('.')[-1],
            'float32',
            'int32',
        )
        return rearrange(out, '(B L) N -> B L N', B=B)
    
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
            else: impl = 'sort'
        
        NG = route_mask.shape[-1]
        G = N // NG
        BLOCK_N = G

        while BLOCK_N > 64: BLOCK_N = BLOCK_N >> 1
        G_iter = G // BLOCK_N

        if impl in ['sort']:
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