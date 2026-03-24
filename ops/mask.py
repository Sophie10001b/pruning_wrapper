import random
import torch
import torch.nn as nn

from functools import lru_cache
from collections import namedtuple

from einops import rearrange
from typing import Optional, Dict, Sequence
from torch.sparse.semi_structured import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS

_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG",
    "sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols",
)

def rand_sparse_semi_structured(r, c, dtype, device, choice=None):
    pattern = '2by4' if dtype != torch.float32 else '1by2'
    if pattern == '1by2':
        ksparse = 2
        choices = [
            [0, 1],
            [1, 0]
        ]
    elif pattern == '2by4':
        ksparse = 4
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // ksparse)]
    mask = torch.tensor(mask_entries, dtype=torch.bool).view(r, c).to(device)
    dense = torch.testing.make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1  # To prevent zeros except where mask applied.
    dense = dense.masked_fill(~mask, 0)
    return dense

@lru_cache
def search_for_alg_id(
    shape: Sequence[int],
    device: Optional[torch.device]=None,
    dtype: Optional[torch.dtype]=torch.float16,
    m_size: Optional[int]=1024,
) -> Sequence[int]:
    A = rand_sparse_semi_structured(shape[0], shape[1], dtype, device)
    B = torch.randn((m_size, shape[1]), device=device, dtype=dtype)
    A_compress = torch._cslt_compress(A)

    alg_id, split_k, split_k_mode, _ = torch._C._cusparselt.mm_search(A_compress, B.t(), None, None, None, False)
    print(f"[INFO] spmm ALG_ID = {alg_id}, split_k = {split_k}, split_k_mode = {split_k_mode} for shape A{B.shape} @ B{A.shape}")
    return alg_id

class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor):
    """
    The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:
    packed = [ specified elements of original tensor | metadata ]
    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata. Since there is only one tensor, we only use the packed and packed_t
    attributes respectively.

    cuSPARSELt also supports transposition fusion, which is necessary for performant 2:4 sparse training, as well
    as specifying alg_id, a config that affects the performance of the matmul depending on matmul sizes.
    """

    BACKEND = "cusparselt"
    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.float8_e4m3fn: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
    }
    _ALG_CONFIG = {
        'alg_id': 0,
        'split_k': 1,
        'split_k_mode': -1,
    }

    @classmethod
    def from_dense(
        cls, original_tensor: torch.Tensor, m_search_size: Optional[int]=1024
    ) -> "SparseSemiStructuredTensorCUSPARSELT":
        cls._validate_device_dim_dtype_shape(original_tensor)

        # search for best ALG_ID for spmm
        # from https://github.com/pytorch/pytorch/issues/153825
        alg_id = search_for_alg_id(original_tensor.shape, device=original_tensor.device, dtype=original_tensor.dtype, m_size=max(m_search_size, 16))
        cls._ALG_CONFIG['alg_id'] = alg_id

        return cls(
            shape=original_tensor.shape,
            packed=torch._cslt_compress(original_tensor),
            meta=None,
            packed_t=None,
            meta_t=None,
            compressed_swizzled_bitmask=None,
            fuse_transpose_cusparselt=SparseSemiStructuredTensor._FUSE_TRANSPOSE,
            alg_id_cusparselt=cls._ALG_CONFIG['alg_id'],
            requires_grad=original_tensor.requires_grad,
        )

    @classmethod
    def prune_dense_static_sort(
        cls, original_tensor: torch.Tensor, algorithm=""
    ) -> "SparseSemiStructuredTensor":
        """
        This function does the same thing as described in SparseSemiStructuredCUTLASS, but uses the cuSPASRELt metadata
        layout and sparse matmul.

        The only functional difference is that cuSPARSELt stores `metadata` and `packed` together into a single tensor.

        [9 1 7 4]                       [9 0 7 0]
        [1 2 3 0]                       [0 2 0 0]
        [8 3 5 4] -> prune 4x4 tile  -> [8 0 0 4] -> pack to cuSPARSELT semi-structured -> packed
        [1 2 6 2]                       [0 0 6 2]

                                                  -> pack to transposed cuSPARSELt      -> packed_t
                                                     semi-structured representation

                                                  -> compute swizzled bitmask           -> compressed_swizzled_bitmask


        The equivalent PyTorch code to create the same three outputs from the dense tensor can be found below:
        ```
        from torch.sparse import SparseSemiStructuredTensorCUSPARSELT
        from torch.sparse._semi_structured_conversions import (
            _sparse_semi_structured_tile,
            _compute_compressed_swizzled_bitmask,
        )

        pruned = _sparse_semi_structured_tile(dense)
        packed_cusparselt = torch._cslt_compress(pruned)
        packed_t_cusparselt = torch._cslt_compress(pruned.t().contiguous())
        bitmask = _compute_compressed_swizzled_bitmask(pruned)

        SparseSemiStructuredTensorCUSPARSELT(
            dense.shape, packed_cutlass, None, packed_t_cutlass, None, bitmask
        )
        ```
        """
        (
            packed,
            meta,
            packed_t,
            meta_t,
            compressed_swizzled_bitmask,
        ) = torch._sparse_semi_structured_tile(
            original_tensor, algorithm=algorithm, use_cutlass=False
        )

        return cls(
            original_tensor.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=compressed_swizzled_bitmask,
            requires_grad=False,
        )

    def _mm(
        self, B: torch.Tensor, *, bias: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        if isinstance(B, SparseSemiStructuredTensor):
            raise ValueError(
                "`SparseSemiStructuredTensor @ SparseSemiStructuredTensor` is not supported by the hardware"
            )
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: Broadcasting is not implemented"
            )
        if B.dtype != self.dtype:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, "
                f"with A.dtype={self.dtype} and B.dtype={B.dtype}. "
                "This operation is only supported when A and B have the same data type."
            )
        if bias is not None and bias.dtype != self.dtype:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)} + C`, "
                f"with A.dtype=B.dtype={self.dtype} and C.dtype={B.dtype}. "
                "This operation is only supported when A, B and C have the same data type."
            )
        # Force fp8 mm to error to be consistent with torch
        if self.dtype == torch.float8_e4m3fn:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, "
                f"with A.dtype=B.dtype={self.dtype}. "
                "mm is not supported for float8_e4m3fn, please use `torch._scaled_mm` instead."
            )
        if self.packed is None:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: operation is not supported"
            )
        else:
            res = torch._cslt_sparse_mm(
                self.packed,
                B,
                bias=bias,
                transpose_result=self.fuse_transpose_cusparselt,
                alg_id=self._ALG_CONFIG['alg_id'],
                split_k=self._ALG_CONFIG['split_k'],
                split_k_mode=self._ALG_CONFIG['split_k_mode'],
            )
            return res.t() if self.fuse_transpose_cusparselt else res


###########################
#   Mask for Unstructured
###########################

class BaseMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        pass
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        pass


class UnstructuredMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        return torch.rand(shape, device=device) < sparsity
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        if mask is None:
            mask = cls.random_sample(module.weight.shape, device=module.weight.device, **kwargs)
        
        module.weight *= mask.to(module.weight.dtype)

class SemiStructuredMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        block_size: Optional[int]=4,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        left = int(sparsity * block_size)
        right = block_size
        assert left > 0

        mask = torch.ones(shape, device=device).flatten()
        mask = rearrange(mask, '(a b) -> a b', b=right)
        indices = torch.multinomial(mask, left, replacement=False)

        rows = torch.arange(mask.shape[0], device=device).view(-1, 1).expand(-1, left)
        mask[rows, indices] = False
        mask = mask.reshape(shape)
        return mask
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        if isinstance(module.weight.data, SparseSemiStructuredTensor): return
        if mask is None:
            mask = cls.random_sample(module.weight.shape, device=module.weight.device, **kwargs)
        
        module.weight *= mask.to(module.weight.dtype)
        module.weight = nn.Parameter(SparseSemiStructuredTensorCUSPARSELT.from_dense(module.weight, m_search_size=kwargs.get('num_tokens', 1024)))
