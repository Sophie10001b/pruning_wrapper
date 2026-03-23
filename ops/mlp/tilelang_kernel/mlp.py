import torch
import tilelang
import tilelang.language as T

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

from ops.utils import check_shared_memory_gemm

#########################
# Kernel Implementation
#########################

@tilelang.jit
def bm_sort_mlp_impl(
    mA, mB, mD, mMask, mIndex,
    BM: int, BN: int, BK: int, TM: int, SplitK: int,
    activation: str='identity',
    dtype: T.DType=T.float16,
    accum_dtype: T.DType=T.float32,
    num_stages: int=3,
):
    N, K = T.const('N, K')
    M = T.dynamic('M')
    mA: T.Tensor[[M, K], dtype]
    mB: T.Tensor[[N, K], dtype]
    mD: T.Tensor[[M, N], dtype]
    mMask: T.Tensor[[M], T.int32]
    mIndex: T.Tensor[[M], T.int32]

    m_split_num = BM // TM
    k_split_size = T.ceildiv(K, SplitK)

    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), SplitK, threads=128) as (bidx, bidy, bidz):
        sA = T.alloc_shared([m_split_num, TM, BK], dtype)
        sB = T.alloc_shared([BN, BK], dtype)

        rD = T.alloc_fragment([BM, BN], accum_dtype)
        rMask = T.alloc_fragment([BM], T.int32)
        rMask_2d = T.alloc_fragment([m_split_num, TM], T.int32)
        rMask_tile = T.alloc_fragment([m_split_num], T.int32)
        rMask_block = T.alloc_fragment([1], T.int32)
        rIndex = T.alloc_fragment([BM], T.int32)

        m_base = bidx * BM
        n_base = bidy * BN
        k_base = bidz * BK

        T.clear(rD)
        T.use_swizzle(panel_size=10, enable=True)

        T.copy(mMask[m_base], rMask)
        for i in T.Parallel(BM):
            rMask[i] = T.if_then_else(m_base + i < M, rMask[i], 0)
        rMask_2d = T.reshape(rMask, [m_split_num, TM])
        T.reduce_bitor(rMask_2d, rMask_tile, dim=-1)
        T.reduce_bitor(rMask_tile, rMask_block, dim=-1)

        if rMask_block[0] != 0:
            T.copy(mIndex[m_base], rIndex)
            rIndex = T.reshape(rIndex, [m_split_num, TM])
            for i, j in T.Parallel(m_split_num, TM):
                rIndex[i, j] = T.if_then_else(rMask_2d[i, j] != 0, rIndex[i, j], rIndex[i, j] % M)
            
            for k in T.Pipelined(0, T.ceildiv(k_split_size, BK), num_stages=num_stages):
                for i, j in T.Parallel(m_split_num, TM):
                    idx = rIndex[i, j]
                    if rMask_tile[i] != 0: sA[i, j, 0:BK] = mA[idx, k_base + k * BK * SplitK:k_base + k * BK * SplitK + BK]
                T.copy(mB[n_base, k_base + k * BK * SplitK], sB)

                for i in T.Serial(0, m_split_num):
                    if rMask_tile[i] != 0: T.gemm(sA[i, :, :], sB, rD[i * TM:(i + 1) * TM, :], transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

            rD = T.reshape(rD, [m_split_num, TM, BN])
            if SplitK == 1:
                if activation == 'silu':
                    for i, j, k in T.Parallel(m_split_num, TM, BN):
                        if rMask_tile[i] != 0: rD[i, j, k] *= T.sigmoid(rD[i, j, k])
                
                if activation == 'relu':
                    for i, j, k in T.Parallel(m_split_num, TM, BN):
                        if rMask_tile[i] != 0: rD[i, j, k] = T.if_then_else(rD[i, j, k] > 0, rD[i, j, k], 0)
                
                for i, j in T.Parallel(m_split_num, TM):
                    idx = rIndex[i, j]
                    if rMask_tile[i] != 0:
                        if rMask_2d[i, j] != 0:
                            mD[idx, n_base:n_base + BN] = rD[i, j, 0:BN]
            
            else:
                for i in T.Serial(m_split_num):
                    for jj in T.Serial(TM // 4):
                        warp_idx = T.get_warp_idx()
                        j = warp_idx * 4 + jj
                        idx = rIndex[i, j]
                        if rMask_tile[i] != 0:
                            if rMask_2d[i, j] != 0:
                                T.atomic_add(rD[i, j, 0:BN], mD[idx, n_base:n_base + BN], memory_order='relaxed')





#########################
# Host Implementation
#########################
class BMSparseMLP:

    support_kernel = [
        'dense',
        'indexing',
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
    def _sort_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        SPLIT_K: Optional[int]=1,
        GROUP_SIZE: Optional[int]=4,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
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

            if kwargs.get('is_offline', False):
                # offline calculate skipping
                m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)
                m_sort = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(-1, BLOCK_M)
                m_sort = m_sort.any(dim=-1)
        
        out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
        bm_sort_mlp_impl(
            x, w, out, m_sort, m_sort_indices,
            BLOCK_M, BLOCK_N, BLOCK_K, 16, SPLIT_K,
            activation=kwargs.get('activation', 'identity'),
            dtype=getattr(T, str(x.dtype).split('.')[1]),
            num_stages=num_stages
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

        estimated_sparsity = kwargs.pop('estimated_sparsity', 1)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        
        if estimated_sparsity == 0: estimated_sparsity = 1
        
        if impl == 'auto': # auto dispatch
            if route_mask is None: impl = 'dense'
            else: impl = 'sort_online'

        if impl in ['sort_offline', 'sort_online']:
            BLOCK_M = min(128, max(16, tilelang.next_power_of_2(M)))
            BLOCK_N = min(128, max(64, tilelang.next_power_of_2(N)))
            BLOCK_K = min(64, max(32, tilelang.next_power_of_2(K)))

            SPLIT_K = 1
            SPLIT_K_list = [1]

            while not check_shared_memory_gemm(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, dtype.itemsize):
                if BLOCK_K > 32: BLOCK_K >>= 1
                elif BLOCK_N > 64: BLOCK_N >>= 1
                elif BLOCK_M > 32: BLOCK_M >>= 1
                else: num_stages -= 1
            
            # split-k active
            if tilelang.cdiv(M, BLOCK_M) * tilelang.cdiv(N, BLOCK_N) < num_sm:
                SPLIT_K_list = [2, 3, 4, 5, 6]
                min_waste = 1.0
                best_split_k = 2
                for split_k in SPLIT_K_list:
                    waste = float(num_sm - ((tilelang.cdiv(M, BLOCK_M) * tilelang.cdiv(N, BLOCK_N) * split_k) % num_sm)) / float(num_sm)
                    if (min_waste > 0 and waste < min_waste):
                        min_waste = waste
                        best_split_k = split_k
                
                SPLIT_K = best_split_k
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
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
            SPLIT_K=SPLIT_K,
            GROUP_SIZE=group_size,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            is_offline=is_offline,
            BLOCK_M_list=[16, 32, 64, 128],
            BLOCK_N_list=[64, 128],
            BLOCK_K_list=[32, 64],
            SPLIT_K_list=SPLIT_K_list,
            num_stages_list=[2, 3],
            **kwargs
        )
