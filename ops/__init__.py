from .cache import DynamicCacheSeqFirst
from .utils import triton_rmsnorm, triton_rope_qk_align

from .router import LinearRouter, BottleneckRouter
from .approximator import BottleneckApproximator

from .attention.base import DenseAttentionKernel
from .attention.query_pruning import QuerySparseAttentionKernel
from .attention.query_head_group_pruning import GroupSparseAttentionKernel
from .attention.query_head_pruning import HeadSparseAttentionKernel

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

__ATTENTION__ = {
    "base": DenseAttentionKernel,
    "query": QuerySparseAttentionKernel,
    "group": GroupSparseAttentionKernel,
    "head": HeadSparseAttentionKernel,
}

__MLP__ = {
    "base": DenseMLPKernel,
    "bm": BMSparseMLPKernel,
    "bn": BNSparseMLPKernel,
    "bk": BKSparseMLPKernel,
}