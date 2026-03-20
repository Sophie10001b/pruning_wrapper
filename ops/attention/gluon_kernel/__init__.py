"""
Gluon kernel implementations for attention operations on NVIDIA GPUs.

This module provides optimized attention kernels using the Gluon DSL:
- SM80 (Ampere): query_pruning_prefill with async copy and mma_v2

All kernels support:
- Query pruning via route_mask and route_indices
- Flash Attention v2 algorithm with online softmax
- GQA/MQA via group size parameter
- Causal masking with FA3-style bitmasks
"""

from .prefill import (
    QueryPruningPrefill,
    query_pruning_prefill,
    # Core functions (for advanced users)
    swizzle_l2,
    _mask_scalar,
    _apply_causal_mask,
    issue_load_Q,
    issue_load_KV,
    issue_mma_QK,
    issue_mma_PV,
    online_softmax_update,
    query_pruning_prefill_impl,
)

__all__ = [
    # Main API
    'QueryPruningPrefill',
    'query_pruning_prefill',
    # Core functions (exposed for flexibility)
    'swizzle_l2',
    '_mask_scalar',
    '_apply_causal_mask',
    'issue_load_Q',
    'issue_load_KV',
    'issue_mma_QK',
    'issue_mma_PV',
    'online_softmax_update',
    'query_pruning_prefill_impl',
]
