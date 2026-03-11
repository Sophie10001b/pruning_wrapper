#######################################################################################
# sm8x, sm9x, sm12x structured sparse gemm kernel & linear wrapper (tilelang)
#######################################################################################
import torch
import tilelang
import tilelang.language as T

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange
from tilelang.contrib import nvcc
from tilelang.utils.sparse import compress_sm80, compress_sm90
from tilelang.layout.gemm_sp import make_cutlass_metadata_layout_sm8x, make_cutlass_metadata_layout_sm90

def make_cutlass_metadata_layout(buffer, mma_dtype: str=T.bfloat16, arch: str | None=None, **extra_args):
    if arch is None:
        arch = nvcc.get_target_compute_version()

    compute_version = nvcc.parse_compute_version(arch)

    if compute_version[0] == 9:
        return make_cutlass_metadata_layout_sm90(buffer=buffer, mma_dtype=mma_dtype, **extra_args)
    elif compute_version[0] in [8, 12]:
        return make_cutlass_metadata_layout_sm8x(buffer=buffer, mma_dtype=mma_dtype)
    else:
        raise NotImplementedError(f"Unsupported architecture: {arch}")

def structured_compress(src: torch.Tensor, block_k: int, transpose: bool=False):
    cc = torch.cuda.get_device_capability("cuda")
    if cc[0] in [8, 12]: 
        return compress_sm80(src, transpose)
    elif cc[0] == 9:
        return compress_sm90(src, block_k, transpose)
    else:
        raise NotImplementedError(f"Unsupported architecture: {cc}")


def generate_2_to_4_sparse_tensor(shape, dtype=torch.float32, device="cpu"):
    if shape[-1] % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4 for 2:4 sparsity.")

    full_tensor = torch.randn(shape, dtype=dtype, device=device)
    group_count = shape[-1] // 4
    group_shape = shape[:-1] + (group_count, 4)

    rand_vals = torch.rand(group_shape, device=device)
    topk_indices = rand_vals.topk(k=2, dim=-1).indices
    mask = torch.zeros(group_shape, dtype=torch.bool, device=device)
    mask.scatter_(-1, topk_indices, True)
    mask = mask.view(shape)

    sparse_tensor = full_tensor * mask
    return sparse_tensor

def get_arch_e_factor(arch: str) -> Tuple:
    compute_version = nvcc.parse_compute_version(arch)
    if compute_version[0] in [8, 12]:
        return T.uint16, 16
    elif compute_version[0] == 9:
        return T.uint8, 8
    else:
        raise NotImplementedError(f"Unsupported architecture: {arch}")


@tilelang.jit
def structured_sparse_gemm(
    A, B, E,
    BM: int,
    BN: int,
    BK: int,
    activation: str='identity',
    dtype: T.dtype=T.bfloat16,
    accum_dtype: T.dtype=T.float32,
    num_stages: int=3,
    arch: str='8.9',
):
    N, K = T.const('N, K')
    M = T.dynamic('M')
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[N, K // 2], dtype]

    e_dtype, e_factor = get_arch_e_factor(arch)
    E: T.Tensor[[M, K // e_factor], e_dtype]
    
    D = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (bx, by):
        sA = T.alloc_shared([BM, BK], dtype)
        sB = T.alloc_shared([BN, BK // 2], dtype)
        sE = T.alloc_shared([BM, BK // e_factor], e_dtype)
        sDT = T.alloc_shared([BM, BN], dtype)

        rD = T.alloc_fragment([BN, BM], accum_dtype)
        # rDT = T.alloc_fragment([BM, BN], dtype)
        T.clear(rD)

        T.use_swizzle(panel_size=10, enable=True)
        T.annotate_layout(
            {
                E: make_cutlass_metadata_layout(E, mma_dtype=dtype, arch=arch, block_k=BK),
                sE: make_cutlass_metadata_layout(sE, mma_dtype=dtype, arch=arch, block_k=BK),
            }
        )

        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
            T.copy(E[bx * BM, k * BK // e_factor], sE, disable_tma=True)
            T.copy(A[bx * BM, k * BK], sA)
            T.copy(B[by * BN, k * BK // 2], sB)

            T.gemm_sp(sB, sE, sA, rD, transpose_B=True) # output [N, M] layout
        
        if activation == 'relu':
            for i, j in T.Parallel(BM, BN):
                rD[i, j] = T.if_then_else(rD[i, j] > 0, rD[i, j], 0)
        
        if activation == 'silu':
            for i, j in T.Parallel(BM, BN):
                rD[i, j] = T.sigmoid(rD[i, j]) * rD[i, j]
        
        # transpose inplace
        for i, j in T.Parallel(BM, BN):
            sDT[i, j] = rD[j, i]

        T.copy(sDT, D[bx * BM, by * BN])
    
    return D


class StructuredSparseMLP:
    @classmethod
    def _spmmv2_kernel(
        cls,
        x: torch.Tensor,
        w: torch.Tensor,
        e: torch.Tensor,
        ACTIVATION: Optional[str]='identity',
        BLOCK_M: Optional[int]=128,
        BLOCK_N: Optional[int]=128,
        BLOCK_K: Optional[int]=64,
        num_stages: Optional[int]=3,
        **kwargs  
    ):
        B, L, D = x.shape

        x_flat = x.flatten(0, 1)
        out = structured_sparse_gemm(
            x_flat, w, e,
            BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K,
            activation=ACTIVATION,
            dtype=T.float16,
            num_stages=num_stages,
            arch=kwargs.get('arch', '8.9'),
        )

        return out.reshape(B, L, D)

    @classmethod
    def kernel(
        cls,
        **kwargs
    ):
        x = kwargs.get('x')
        w = kwargs.get('w')
        dtype = x.dtype
        device = x.device

        B, L, D = x.shape
        M, N, K = B * L, w.shape[0], D

        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")
        num_stages = 3

        BLOCK_M = tilelang.next_power_of_2(M)
        BLOCK_M = min(128, max(16, BLOCK_M))

        BLOCK_N = tilelang.next_power_of_2(N)
        BLOCK_N = min(128, max(16, BLOCK_N))

        BLOCK_K = 64

        return cls._spmmv2_kernel(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_stages=num_stages,
            arch='.'.join(map(str, cc)),
            **kwargs
        )

if __name__ == '__main__':
    device = 'cuda:0'
    dtype = torch.float16

    M = 4096
    N = 4096
    K = 4096

    A = torch.rand((1, M, K), dtype=dtype, device=device)
    B = generate_2_to_4_sparse_tensor((M, K), dtype=dtype, device=device)
    BS, E = structured_compress(B, block_k=32, transpose=False)

    spmm_res = StructuredSparseMLP.kernel(
        x=A,
        w=BS,
        e=E,
        ACTIVATION='identity',
    )
    res = (A.flatten(0, 1) @ B.T).squeeze(0)

    assert torch.allclose(spmm_res, res, atol=1e-2, rtol=1e-2)
    pass