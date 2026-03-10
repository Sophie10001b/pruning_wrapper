#######################################################################################
# sm8x, sm9x, sm12x structured sparse gemm kernel & linear wrapper (tilelang)
#######################################################################################
import torch
import tilelang
import tilelang.language as T

from tilelang.utils.sparse import compress_sm80, compress_sm90
from tilelang.layout.gemm_sp import make_cutlass_metadata_layout_sm8x, make_cutlass_metadata_layout_sm90

@tilelang.jit
def structured_sparse_gemm(
    A, B, E,
    BM: int,
    BN: int,
    BK: int,
    dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
):
    N, K = T.const('N, K')
    M = T.dynamic('M')
    A: T.Tensor[[M, K // 2], dtype]
    B: T.Tensor[[N, K], dtype]
    E: T.Tensor[[M, K // 8], T.uint8]
    
    D = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (bx, by):
        sA = T.alloc_shared([BM, BK // 2], dtype)