import os
import itertools
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

class BNSparseGLUBKSparseMLP:

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
        G_iter: Optional[int]=1,
        NG: Optional[int]=1,
        GROUP_SIZE: Optional[int]=4,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
        @tilelang.jit(out_idx=[-1])
        def glu_fuse_splitk_mlp(
            M, N, K, G_iter,
            HOPPER,
            HAS_BIAS_UP, HAS_BIAS_GATE,
            ACTIVATION,
            BLOCK_M, BLOCK_N, BLOCK_K,
            dtype,
        ):
            x_shape = (M, K)
            wu_shape = (N, K)
            wg_shape = (N, K)
            wd_shape = (K, N)
            bu_shape = (N,)
            bg_shape = (N,)
            out_shape = (M, K)
            mask_shape = (NG, T.ceildiv(M, BLOCK_M))
            indices_shape = (NG, M)
            
            acc_dtype = 'float32'
            index_dtype = 'int32'

            num_threads = num_warps * 32

            @T.prim_func
            def func(
                x: T.Tensor(x_shape, dtype),
                route_mask: T.Tensor(mask_shape, index_dtype),
                route_indices: T.Tensor(indices_shape, index_dtype),
                wu: T.Tensor(wu_shape, dtype),
                wg: T.Tensor(wg_shape, dtype),
                wd: T.Tensor(wd_shape, dtype),
                bu: T.Tensor(bu_shape, dtype),
                bg: T.Tensor(bg_shape, dtype),
                out: T.Tensor(out_shape, dtype),
            ):
                with T.Kernel(T.ceildiv(M, BLOCK_M), NG, G_iter, threads=num_threads) as (bx, by, bz):
                    S_x = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                    S_wu = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
                    S_wg = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)

                    R_indices = T.alloc_fragment((BLOCK_M), index_dtype)
                    R_indices_valid = T.alloc_fragment((BLOCK_M), index_dtype)
                    R_acc_u = T.alloc_fragment((BLOCK_M, BLOCK_N), acc_dtype)
                    R_acc_u_cast = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
                    R_acc_g = T.alloc_fragment((BLOCK_M, BLOCK_N), acc_dtype)
                    R_bu = T.alloc_fragment((BLOCK_N), acc_dtype)
                    R_bg = T.alloc_fragment((BLOCK_N), acc_dtype)

                    T.clear(R_acc_u)
                    T.clear(R_acc_g)

                    T.use_swizzle(panel_size=10)
                    if route_mask[by, bx]:
                        for k in T.Pipelined(0, T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                            T.copy(route_indices[by, (bx * BLOCK_M + dx) % M], R_indices)
                            for dx in T.Parallel(BLOCK_M):
                                R_indices_valid[dx] = T.if_then_else(R_indices[dx] > 0, R_indices[dx], 0)
                                T.copy(x[R_indices_valid[dx], :], S_x[dx, :])
                            
                            T.copy(wu[(by * G_iter + bz) * BLOCK_N, k * BLOCK_K], S_wu)
                            T.copy(wg[(by * G_iter + bz) * BLOCK_N, k * BLOCK_K], S_wg)

                            T.gemm(S_x, S_wu, R_acc_u, transpose_B=True)
                            T.gemm(S_x, S_wg, R_acc_g, transpose_B=True)
                        
                        if HAS_BIAS_UP:
                            T.copy(bu[(by * G_iter + bz) * BLOCK_N], R_bu)
                            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                                R_acc_u[i, j] += R_bu[j]
                        
                        if HAS_BIAS_GATE:
                            T.copy(bg[(by * G_iter + bz) * BLOCK_N], R_bg)
                            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                                R_acc_g[i, j] += R_bg[j]
                        
                        if ACTIVATION == 'silu':
                            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                                R_acc_g[i, j] *= T.sigmoid(R_acc_u[i, j])
                        
                        if ACTIVATION == 'relu':
                            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                                R_acc_g[i, j] = T.max(R_acc_g[i, j], 0.0)
                        
                        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                            R_acc_u[i, j] *= R_acc_g[i, j]
                        T.copy(R_acc_u, R_acc_u_cast)

                        S_wd = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
                        S_out = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                        R_acc_d = T.alloc_fragment((BLOCK_M, BLOCK_K), acc_dtype)

                        T.clear(R_acc_d)
                        for k in T.Pipelined(0, T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                            T.copy(wd[k * BLOCK_K, (by * G_iter + bz) * BLOCK_N], S_wd)
                            T.gemm(R_acc_u_cast, S_wd, R_acc_d, transpose_B=True)
                            T.copy(R_acc_d, S_out)
                            
                            for i in T.Parallel(BLOCK_M):
                                if HOPPER:
                                    for j in range(0, BLOCK_K, 4):
                                        if R_indices[i] > 0: T.atomic_addx4(out[R_indices[i], j:j+4], S_out[i, j:j+4])
                                else:
                                    for j in range(0, BLOCK_K, 2):
                                        if R_indices[i] > 0: T.atomic_addx2(out[R_indices[i], j:j+2], S_out[i, j:j+2])

            return func
        
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

        kernel = glu_fuse_splitk_mlp(
            M, N, K, G_iter,
            HOPPER=kwargs.get('hopper', False),
            HAS_BIAS_UP=bu is not None,
            HAS_BIAS_GATE=bg is not None,
            ACTIVATION=activation,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            dtype=str(x_flat.dtype).split('.')[-1]
        )
        out = kernel(x_flat, m_sort_pad, m_sort_indices, wu, wg, wd, bu, bg)
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
                if M < 4: impl = 'reduce_small_bsz'
                else: impl = 'reduce'
            else: impl = 'atomic'
        
        NG = route_mask.shape[-1]
        G = N // NG
        BLOCK_N = G

        while BLOCK_N > 64: BLOCK_N = BLOCK_N >> 1
        G_iter = G // BLOCK_N

        if impl in ['atomic', 'reduce']:
            BLOCK_M = tilelang.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M > int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_K = min(128, max(32, tilelang.next_power_of_2(K)))

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