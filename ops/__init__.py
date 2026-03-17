from .cache import DynamicCacheSeqFirst
from .utils import triton_rmsnorm, triton_rope_qk_align

from .router import LinearRouter, BottleneckRouter
from .approximator import BottleneckApproximator

from .mask import BaseMask, UnstructuredMask, SemiStructuredMask

from .index import BaseIndex, StructuredIndex

from .attention_threshold import BaseThreshold, BlasstThreshold

from .attention.base import DenseAttentionKernel
from .attention.query_pruning import QuerySparseAttentionKernel
from .attention.query_head_group_pruning import GroupSparseAttentionKernel
from .attention.query_head_pruning import HeadSparseAttentionKernel

from .attention.pv_pruning import BlasstAttentionKernel

from .mlp.base import DenseMLPKernel
from .mlp.bm_pruning import BMSparseMLPKernel
from .mlp.bn_pruning import BNSparseMLPKernel
from .mlp.bk_pruning import BKSparseMLPKernel

__all__ = [
    "DynamicCacheSeqFirst",
    "triton_rmsnorm",
    "triton_rope_qk_align",
    "DenseAttentionKernel",
    "QuerySparseAttentionKernel",
    "GroupSparseAttentionKernel",
    "HeadSparseAttentionKernel",
    "BlasstAttentionKernel",
    "DenseMLPKernel",
    "BMSparseMLPKernel",
    "BNSparseMLPKernel",
    "BKSparseMLPKernel",
]

__ROUTER__ = {
    "linear": LinearRouter,
    "bottleneck": BottleneckRouter,
}

__APPROXIMATOR__ = {
    "bottleneck": BottleneckApproximator,
}

__KV_CACHE__ = {
    "base": DynamicCacheSeqFirst,
}

__MASK__ = {
    "base": BaseMask,
    "unstructured": UnstructuredMask,
    "semi_structured": SemiStructuredMask,
}

__INDEX__ = {
    "base": BaseIndex,
    "structured": StructuredIndex,
}

__THRESHOLD__ = {
    "base": BaseThreshold,
    "blasst": BlasstThreshold,
}

__ATTENTION__ = {
    "base": DenseAttentionKernel,
    "query": QuerySparseAttentionKernel,
    "group": GroupSparseAttentionKernel,
    "head": HeadSparseAttentionKernel,
}

__SPARSE_ATTENTION__ = {
    "base": DenseAttentionKernel,
    "blasst": BlasstAttentionKernel,
}

__MLP__ = {
    "base": DenseMLPKernel,
    "bm": BMSparseMLPKernel,
    "bn": BNSparseMLPKernel,
    "bk": BKSparseMLPKernel,
}