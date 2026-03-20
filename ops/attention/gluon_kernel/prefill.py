"""
SM80 Query Pruning Prefill Attention Kernel (Gluon Implementation)

This module implements Flash Attention v2 with query pruning support for NVIDIA sm80 (Ampere) GPUs.
Features:
- Query pruning via gather/scatter with route_mask and route_indices
- FA3-style bitmask causal masking for efficient attention masking
- Software pipelining with async copy and mma_v2 tensor cores
- Support for MHA, GQA, and MQA attention patterns
- Both online and offline pruning modes

Author: Auto-generated for LMU pruning project
"""

import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

from ops.utils import get_autotune_config, get_autotune_cache


# =============================================================================
# Layout and Helper Functions
# =============================================================================

@gluon.jit
def swizzle_l2(i, j, size_i, size_j, size_g):
    """
    L2 cache swizzling for better memory locality.
    Groups iterations to improve cache hit rate.
    
    Args:
        i, j: Current grid indices
        size_i, size_j: Grid dimensions
        size_g: Group size for swizzling
    
    Returns:
        Swizzled (new_i, new_j) indices
    """
    # Unrolled index in array
    ij = i * size_j + j
    # Number of elements in size_g groups of size_j columns
    size_gj = size_g * size_j
    # Index of the group in which (i,j) is
    group_id = ij // size_gj
    # Row-index of the first element of this group
    off_i = group_id * size_g
    # Last group may have fewer rows
    size_g = gl.minimum(size_i - off_i, size_g)
    # Linear index with respect to the first element in this group
    ij = ij % size_gj
    # New row and column indices
    new_i = off_i + ij % size_g
    new_j = ij // size_g
    return new_i, new_j


# =============================================================================
# FA3 Bitmask Causal Masking
# =============================================================================

@gluon.jit
def _mask_scalar(qk, col_limit_right, s, i):
    """
    FA3-style scalar masking using bitwise operations.
    
    Uses bitmask to efficiently mask attention scores:
    - col_limit_right: number of columns visible from the right
    - s: group start (every 16 columns)
    - i: index within group (0-15)
    
    Example for col_limit_right=21:
    - Columns 0-20: visible (mask bit = 0)
    - Columns 21+: masked (mask bit = 1, set to -inf)
    """
    col_lim_right_s = col_limit_right - s        # Visible columns in this group
    col_lim_right_cur = gl.maximum(col_lim_right_s, 0)  # At least 0
    mask = -1 << col_lim_right_cur               # Bitmask: low bits = 0 (visible)
    mask_i_bit = (mask & (1 << i)) == 0          # Check if bit i is 0 (visible)
    return gl.where(mask_i_bit, qk, -float("inf"))


@gluon.jit
def _apply_causal_mask(qk, col_limit_right):
    """
    Apply causal mask using FA3 bitmask technique.
    
    Processes 16 columns at a time for efficient bit manipulation.
    Uses map_elementwise for better PTX optimization.
    
    Args:
        qk: Attention scores tensor [BLOCK_M, BLOCK_N]
        col_limit_right: Rightmost visible column for each query
    
    Returns:
        Masked attention scores
    """
    # Column indices within the KV tile
    offs_n = gl.arange(0, qk.shape[1])[None, :]
    # Group start (every 16 columns)
    s = offs_n & ~0xf
    # Index within group (0-15)
    i = offs_n & 0xf
    
    # Apply mask elementwise with efficient bit operations
    return gl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)


# =============================================================================
# Async Load Functions with Query Pruning Support
# =============================================================================

@gluon.jit
def issue_load_Q(
    producer: gl.uint32,
    mQ: gl.tensor,
    sQ: gl.shared_memory_descriptor,
    indices: gl.tensor,
    q_base: gl.int64,
    HQ: gl.constexpr,
    D: gl.constexpr,
    LQ: gl.int64,
    BLOCK_M: gl.constexpr,
    num_stages: gl.constexpr,
    layout: gl.constexpr,
):
    """
    Issue async load for query tile with gather support.
    
    Loads Q[indices] into shared memory for pruned queries.
    If indices is None or pruning disabled, loads contiguous block.
    
    Args:
        producer: Current pipeline stage index
        mQ: Query tensor pointer [B, LQ, HQ, D]
        sQ: Query shared memory descriptor
        indices: Gather indices [BLOCK_M] or None for dense
        q_base: Base offset for query batch/head
        HQ: Number of query heads
        D: Head dimension
        LQ: Query sequence length
        BLOCK_M: Query block size
        num_stages: Pipeline depth
        layout: Load layout
    
    Returns:
        Updated producer index
    """
    index = producer % num_stages
    m_start = producer * BLOCK_M
    
    # Create mask for valid queries
    offs_m = m_start + gl.arange(0, BLOCK_M, layout=layout)
    mask = offs_m < LQ
    
    if indices is not None:
        # Gather mode: use indices to load specific queries
        gather_mask = indices < LQ
        valid_indices = gl.where(gather_mask, indices, 0)
        
        # Load with gather: Q[valid_indices, :, :]
        src_ptr = mQ + q_base + valid_indices[:, None] * HQ * D + gl.arange(0, D)[None, :]
        async_copy.async_copy_global_to_shared(
            sQ.index(index), 
            src_ptr, 
            mask=gather_mask[:, None] & mask[:, None]
        )
    else:
        # Dense mode: contiguous load
        src_ptr = mQ + q_base + offs_m[:, None] * HQ * D + gl.arange(0, D)[None, :]
        async_copy.async_copy_global_to_shared(
            sQ.index(index),
            src_ptr,
            mask=mask[:, None]
        )
    
    async_copy.commit_group()
    return producer + 1


@gluon.jit
def issue_load_KV(
    producer: gl.uint32,
    mK: gl.tensor,
    mV: gl.tensor,
    sK: gl.shared_memory_descriptor,
    sV: gl.shared_memory_descriptor,
    kv_base: gl.int64,
    HK: gl.constexpr,
    D: gl.constexpr,
    LK: gl.int64,
    pad_offset: gl.int64,
    BLOCK_N: gl.constexpr,
    num_stages: gl.constexpr,
    layout: gl.constexpr,
):
    """
    Issue async load for key and value tiles.
    
    Args:
        producer: Current pipeline stage index
        mK, mV: Key and value tensor pointers
        sK, sV: Shared memory descriptors
        kv_base: Base offset for KV batch/head
        HK: Number of key/value heads
        D: Head dimension
        LK: Key sequence length
        pad_offset: Left padding offset
        BLOCK_N: KV block size
        num_stages: Pipeline depth
        layout: Load layout
    
    Returns:
        Updated producer index
    """
    index = producer % num_stages
    kv_start = producer * BLOCK_N
    
    # Actual key positions (accounting for padding)
    actual_kv_start = kv_start + pad_offset
    offs_n = actual_kv_start + gl.arange(0, BLOCK_N, layout=layout)
    mask = offs_n < LK
    
    # Load Key: K[offs_n, :, :]
    src_k = mK + kv_base + offs_n[:, None] * HK * D + gl.arange(0, D)[None, :]
    async_copy.async_copy_global_to_shared(
        sK.index(index),
        src_k,
        mask=mask[:, None]
    )
    
    # Load Value: V[offs_n, :, :]
    src_v = mV + kv_base + offs_n[:, None] * HK * D + gl.arange(0, D)[None, :]
    async_copy.async_copy_global_to_shared(
        sV.index(index),
        src_v,
        mask=mask[:, None]
    )
    
    async_copy.commit_group()
    return producer + 1


# =============================================================================
# MMA Operations with mma_v2
# =============================================================================

@gluon.jit
def issue_mma_QK(
    consumer: gl.uint32,
    sQ: gl.shared_memory_descriptor,
    sK: gl.shared_memory_descriptor,
    rQ_layout: gl.constexpr,
    rK_layout: gl.constexpr,
    rS: gl.tensor,
    num_stages: gl.constexpr,
):
    """
    Issue mma_v2 for Q @ K^T.
    
    Loads Q and K from shared memory, computes attention scores.
    
    Args:
        consumer: Current pipeline stage
        sQ, sK: Shared memory descriptors
        rQ_layout, rK_layout: Register layouts for dot operands
        rS: Score accumulator (output)
        num_stages: Pipeline depth
    
    Returns:
        Updated consumer index and new accumulator
    """
    index = consumer % num_stages
    
    # Load from shared to registers
    rQ = sQ.index(index).load(rQ_layout)
    rK = sK.index(index).permute((1, 0)).load(rK_layout)  # Transpose K
    
    # MMA: S = Q @ K^T
    rS = mma_v2(rQ, rK, rS)
    
    return consumer + 1, rS


@gluon.jit
def issue_mma_PV(
    consumer: gl.uint32,
    sP: gl.shared_memory_descriptor,
    sV: gl.shared_memory_descriptor,
    rP_layout: gl.constexpr,
    rV_layout: gl.constexpr,
    rO: gl.tensor,
    num_stages: gl.constexpr,
):
    """
    Issue mma_v2 for P @ V.
    
    Loads P (softmax probabilities) and V from shared memory.
    
    Args:
        consumer: Current pipeline stage
        sP, sV: Shared memory descriptors
        rP_layout, rV_layout: Register layouts
        rO: Output accumulator (in/out)
        num_stages: Pipeline depth
    
    Returns:
        Updated consumer index and new accumulator
    """
    index = consumer % num_stages
    
    # Load from shared to registers
    rP = sP.index(index).load(rP_layout)
    rV = sV.index(index).load(rV_layout)
    
    # MMA: O = P @ V
    rO = mma_v2(rP, rV, rO)
    
    return consumer + 1, rO


# =============================================================================
# Online Softmax Update
# =============================================================================

@gluon.jit
def online_softmax_update(
    rS: gl.tensor,
    score_max: gl.tensor,
    score_sum: gl.tensor,
    acc: gl.tensor,
    sV: gl.shared_memory_descriptor,
    rP_layout: gl.constexpr,
    rV_layout: gl.constexpr,
    rO_layout: gl.constexpr,
    score_layout: gl.constexpr,
    qk_scale: gl.float32,
    consumer: gl.uint32,
    num_stages: gl.constexpr,
):
    """
    Online softmax update with attention output computation.
    
    Implements Flash Attention v2 online softmax:
    1. Track running maximum m_i
    2. Compute correction factor alpha = exp2(m_old - m_new)
    3. Rescale accumulator and running sum
    4. Compute P = exp2(S - m_new) and accumulate P @ V
    
    Args:
        rS: Attention scores [BLOCK_M, BLOCK_N]
        score_max: Running max [BLOCK_M]
        score_sum: Running sum [BLOCK_M]
        acc: Output accumulator [BLOCK_M, D]
        sV: Value shared memory
        rP_layout, rV_layout: Register layouts
        rO_layout: Output layout
        qk_scale: Scale factor (1/sqrt(D))
        consumer: Pipeline stage
        num_stages: Pipeline depth
    
    Returns:
        Updated (score_max, score_sum, acc, consumer)
    """
    # Scale scores
    rS_scaled = rS * qk_scale
    
    # Compute new max and ensure layout consistency
    score_max_temp = gl.maximum(score_max, gl.max(rS_scaled, axis=1))
    score_max_new = gl.convert_layout(score_max_temp, score_max.type.layout)
    
    # Correction factor: exp2(old_max - new_max)
    correction = gl.exp2(score_max - score_max_new)
    
    # Rescale softmax denominator
    score_sum = score_sum * correction
    
    # Compute P = exp2(S - new_max) for this tile
    rP = gl.exp2(rS_scaled - score_max_new[:, None])
    
    # Update running sum and ensure layout consistency
    score_sum_temp = score_sum + gl.sum(rP, axis=1)
    score_sum = gl.convert_layout(score_sum_temp, score_sum.type.layout)
    
    # Rescale accumulator
    # Convert correction to output layout for per-element scaling
    # correction_2d = gl.broadcast_to(correction[:, None], acc.shape)
    # correction_2d = gl.convert_layout(correction_2d, acc.type.layout)
    acc = acc * correction[:, None]
    
    # Load P to shared memory for MMA (convert layout first)
    # Note: In practice, we might keep P in registers if possible
    # Here we assume sP is pre-allocated
    
    # For simplicity, accumulate PV directly
    # In full implementation, this would be:
    # consumer, acc = issue_mma_PV(consumer, sP, sV, rP_layout, rV_layout, acc, num_stages)
    
    # Placeholder: actual PV MMA would go here
    # For now, accumulate manually (less efficient but functional)
    index = consumer % num_stages
    rV = sV.index(index).load(rV_layout)
    rP_converted = gl.convert_layout(rP.to(rV.dtype), rP_layout)
    acc = mma_v2(rP_converted, rV, acc)
    
    return score_max_new, score_sum, acc, consumer + 1


# =============================================================================
# Main Kernel Implementation
# =============================================================================

def query_pruning_prefill_impl(
    mQ: gl.tensor,
    mK: gl.tensor,
    mV: gl.tensor,
    route_mask: gl.tensor,
    route_indices: gl.tensor,
    pad_offset: gl.tensor,
    mO: gl.tensor,
    B: gl.int64,
    LQ: gl.int64,
    LK: gl.int64,
    HQ: gl.constexpr,
    HK: gl.constexpr,
    D: gl.constexpr,
    G: gl.constexpr,
    qk_scale: gl.float32,
    # Block sizes
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    # Pipelining
    num_stages: gl.constexpr,
    num_warps: gl.constexpr,
    # Pruning mode
    IS_OFFLINE: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    # Layouts
    sQ_layout: gl.constexpr,
    sK_layout: gl.constexpr,
    sV_layout: gl.constexpr,
    rQ_layout: gl.constexpr,
    rK_layout: gl.constexpr,
    rP_layout: gl.constexpr,
    rV_layout: gl.constexpr,
    rO_layout: gl.constexpr,
):
    """
    Main query pruning prefill attention kernel for sm80.
    
    Implements Flash Attention v2 with:
    - Query pruning via route_mask/route_indices
    - Software pipelining with async copy
    - mma_v2 tensor core operations
    - FA3-style bitmask causal masking
    
    Args:
        mQ, mK, mV: Input tensors [B, L, H, D]
        route_mask: Pruning mask [B, LQ] or [B, cdiv(LQ, BLOCK_M)]
        route_indices: Gather indices [B, LQ]
        pad_offset: Left padding per batch [B]
        mO: Output tensor [B, LQ, HQ, D]
        B, LQ, LK: Batch size, query/key lengths
        HQ, HK: Number of query/key heads
        D: Head dimension
        G: Group size (HQ // HK) for GQA
        qk_scale: Attention scale factor
        BLOCK_M, BLOCK_N: Tile sizes
        num_stages, num_warps: Pipeline and parallelism config
        IS_OFFLINE: Use block-level skip if True
        IS_CAUSAL: Enable causal masking if True
        *_layout: Memory layouts
    """
    # Grid: (num_blocks_m, HQ, B)
    pid_m = gl.program_id(0)
    pid_h = gl.program_id(1)
    pid_b = gl.program_id(2)
    
    num_blocks_m = gl.num_programs(0)
    
    # Swizzle for better L2 locality
    # pid_m, pid_h = swizzle_l2(pid_m, pid_h, num_blocks_m, HQ, 4)
    
    # Compute head mappings for GQA
    key_head_id = pid_h // G
    query_head_id = pid_h
    
    # Base offsets
    q_batch_offset = pid_b * LQ * HQ * D
    q_head_offset = query_head_id * D
    kv_batch_offset = pid_b * LK * HK * D
    kv_head_offset = key_head_id * D
    
    # Get padding offset for this batch
    pad_offset_kv = gl.load(pad_offset + pid_b)
    key_length = LK - pad_offset_kv
    
    # Query block start
    m_start = pid_m * BLOCK_M
    
    # Check if this block should be skipped (pruning)
    skip_flag = 0
    if IS_OFFLINE:
        # Offline mode: check block-level mask
        skip_flag = gl.load(route_mask + pid_b * num_blocks_m + pid_m)
    else:
        # Online mode: check if any query in block is active
        query_mask = gl.load(
            route_mask + pid_b * LQ + m_start + gl.arange(0, BLOCK_M),
            mask=m_start + gl.arange(0, BLOCK_M) < LQ,
            other=0,
        )
        skip_flag = gl.reduce_or(query_mask, axis=-1)
    
    if skip_flag > 0:
        # Load gather indices for pruned queries
        if IS_OFFLINE:
            indices = gl.load(
                route_indices + pid_b * LQ + m_start + gl.arange(0, BLOCK_M),
                mask=m_start + gl.arange(0, BLOCK_M) < LQ,
                other=-1,
            )
            valid_mask = indices >= 0
            indices = gl.where(valid_mask, indices, 0)
        else:
            valid_mask = query_mask > 0
            indices = gl.load(
                route_indices + pid_b * LQ + m_start + gl.arange(0, BLOCK_M),
                mask=valid_mask,
                other=0,
            )
        
        # Allocate shared memory (multi-buffered)
        dtype: gl.constexpr = mQ.dtype.element_ty
        sQ = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_M, D], sQ_layout)
        sK = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_N, D], sK_layout)
        sV = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_N, D], sV_layout)
        
        # Allocate accumulator
        rO = gl.zeros([BLOCK_M, D], gl.float32, layout=rO_layout)
        
        # Initialize softmax state with consistent layout
        # Use BlockedLayout for 1D reduction results
        score_layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
        score_max = gl.full([BLOCK_M], -float('inf'), dtype=gl.float32, layout=score_layout)
        score_sum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=score_layout)
        
        # Pipeline indices
        producer: gl.uint32 = 0
        consumer: gl.uint32 = 0
        
        # Layout for indices
        indices_layout: gl.constexpr = gl.SliceLayout(1, gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0]))
        
        # Compute number of KV tiles
        # For causal attention, only process up to max query position
        max_query_idx = gl.max(indices)
        if IS_CAUSAL:
            num_kv_tiles = gl.cdiv(max_query_idx - pad_offset_kv + 1, BLOCK_N)
        else:
            num_kv_tiles = gl.cdiv(key_length, BLOCK_N)
        
        # Prologue: issue first (num_stages-1) KV loads
        for _ in gl.static_range(num_stages - 1):
            if producer < num_kv_tiles:
                producer = issue_load_KV(
                    producer, mK, mV, sK, sV,
                    kv_batch_offset + kv_head_offset,
                    HK, D, LK, pad_offset_kv,
                    BLOCK_N, num_stages, indices_layout
                )
        
        # Issue first Q load (with gather)
        q_base = q_batch_offset + q_head_offset
        producer_q = issue_load_Q(
            0, mQ, sQ, indices, q_base,
            HQ, D, LQ, BLOCK_M, 1, indices_layout
        )
        
        # Main loop: pipelined compute
        for kv_tile in range(num_kv_tiles - (num_stages - 1)):
            # Prefetch next KV
            producer = issue_load_KV(
                producer, mK, mV, sK, sV,
                kv_batch_offset + kv_head_offset,
                HK, D, LK, pad_offset_kv,
                BLOCK_N, num_stages, indices_layout
            )
            
            # Wait for KV load
            async_copy.wait_group(num_stages - 1)
            
            # Compute QK
            consumer, rS = issue_mma_QK(
                consumer, sQ, sK,
                rQ_layout, rK_layout,
                gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=rO_layout),
                num_stages
            )
            
            # Apply causal mask if needed
            if IS_CAUSAL:
                kv_start = kv_tile * BLOCK_N
                col_limit_right = (indices - (kv_start + pad_offset_kv) + 1)[:, None]
                rS = _apply_causal_mask(rS, col_limit_right)
            
            # Online softmax and PV accumulation
            score_max, score_sum, rO, consumer = online_softmax_update(
                rS, score_max, score_sum, rO, sV,
                rP_layout, rV_layout, rO_layout, score_layout,
                qk_scale, consumer, num_stages
            )
        
        # Epilogue: drain pipeline
        for i in gl.static_range(num_stages - 1):
            async_copy.wait_group(num_stages - 2 - i)
            
            consumer, rS = issue_mma_QK(
                consumer, sQ, sK,
                rQ_layout, rK_layout,
                gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=rO_layout),
                num_stages
            )
            
            if IS_CAUSAL:
                kv_start = (num_kv_tiles - (num_stages - 1) + i) * BLOCK_N
                col_limit_right = (indices - (kv_start + pad_offset_kv) + 1)[:, None]
                rS = _apply_causal_mask(rS, col_limit_right)
            
            score_max, score_sum, rO, consumer = online_softmax_update(
                rS, score_max, score_sum, rO, sV,
                rP_layout, rV_layout, rO_layout, score_layout,
                qk_scale, consumer, num_stages
            )
        
        # Normalize output
        rO = rO / score_sum[:, None]
        
        # Convert to output dtype and store with scatter
        rO_out = rO.to(dtype)
        out_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
        rO_out = gl.convert_layout(rO_out, out_layout)
        
        out_ptr = mO + q_batch_offset + indices[:, None] * HQ * D + q_head_offset + gl.arange(0, D)[None, :]
        gl.store(
            out_ptr,
            rO_out,
            mask=valid_mask[:, None] & (indices < LQ)[:, None]
        )


# =============================================================================
# Host Wrapper Class
# =============================================================================

class QueryPruningPrefill:
    """
    Host wrapper for sm80 query pruning prefill attention kernel.
    
    Supports three modes:
    - 'dense': No pruning, standard attention
    - 'sort_online': Token-level pruning with runtime mask check
    - 'sort_offline': Block-level pruning with pre-computed skip flags
    
    Example:
        >>> q = torch.randn(B, LQ, HQ, D, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, LK, HK, D, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, LK, HK, D, dtype=torch.bfloat16, device='cuda')
        >>> route_mask = torch.randint(0, 2, (B, LQ), dtype=torch.bool, device='cuda')
        >>> out = QueryPruningPrefill.kernel(
        ...     'sort_offline', q=q, k=k, v=v,
        ...     route_mask=route_mask, estimated_sparsity=0.5
        ... )
    """
    
    support_kernel = ['dense', 'sort_online', 'sort_offline']
    
    @classmethod
    def _sort_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: Optional[torch.Tensor] = None,
        pad_offset: Optional[torch.Tensor] = None,
        BLOCK_M: int = 64,
        BLOCK_N: int = 64,
        num_stages: int = 2,
        num_warps: int = 4,
        **kwargs
    ):
        """
        Execute kernel with query pruning.
        
        Args:
            q, k, v: Input tensors [B, L, H, D]
            route_mask: Pruning mask [B, LQ] or [B, cdiv(LQ, BLOCK_M)]
            pad_offset: Left padding [B]
            BLOCK_M, BLOCK_N: Tile sizes
            num_stages: Pipeline depth (2 or 3)
            num_warps: Number of warps (4 or 8)
            is_offline: Whether to use block-level skip
        """
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK
        
        device = q.device
        dtype = q.dtype
        
        # Default padding offset
        if pad_offset is None:
            pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        
        # Prepare sorted indices
        is_offline = kwargs.get('is_offline', False)
        
        if route_mask is not None:
            # Sort route_mask descending to group active queries
            m_sort, m_sort_indices = torch.sort(
                route_mask, descending=True, stable=False
            )  # [B, LQ]
            
            if is_offline:
                # Offline mode: compute block-level skip flags
                m_sort_indices = m_sort_indices.masked_fill(~m_sort, -1)
                # Pad to BLOCK_M boundary and reshape
                pad_size = (BLOCK_M - LQ % BLOCK_M) % BLOCK_M
                m_sort = torch.nn.functional.pad(m_sort, (0, pad_size), value=0)
                m_sort = m_sort.reshape(B, -1, BLOCK_M)
                m_sort = m_sort.any(dim=-1)  # [B, num_blocks_m]
        else:
            # Dense mode: all queries active
            m_sort = torch.ones((B, (LQ + BLOCK_M - 1) // BLOCK_M), dtype=torch.bool, device=device)
            m_sort_indices = torch.arange(LQ, device=device).unsqueeze(0).expand(B, -1)
        
        # Output tensor
        out = torch.zeros_like(q)
        
        # Grid dimensions
        grid = lambda meta: (triton.cdiv(LQ, meta['BLOCK_M']), HQ, B)
        
        # Get layouts for given block sizes
        gl_dtype = getattr(gl, str(dtype).split('.')[1])
        
        # Shared memory layouts (NVMMA for efficient MMA access)
        sQ_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, D], gl_dtype)
        sK_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, D], gl_dtype)
        sV_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, D], gl_dtype)
        
        # MMA register layouts
        rO_layout = gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[1, num_warps],
            instr_shape=[16, 8]  # m16n8k8 for mma_v2
        )
        rQ_layout = gl.DotOperandLayout(operand_index=0, parent=rO_layout, k_width=2)
        rK_layout = gl.DotOperandLayout(operand_index=1, parent=rO_layout, k_width=2)
        rP_layout = gl.DotOperandLayout(operand_index=0, parent=rO_layout, k_width=2)
        rV_layout = gl.DotOperandLayout(operand_index=1, parent=rO_layout, k_width=2)
        
        # Autotuning configuration
        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N', 'num_stages'],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            **kwargs
        )
        
        # Get cached kernel
        kernel = get_autotune_cache(
            query_pruning_prefill_impl,
            enable_autotune=True,
            config=config,
            keys=['B', 'LQ', 'LK', 'HQ', 'HK', 'D'],
            do_not_specialize=['qk_scale'],
            is_gluon=True
        )
        
        # Launch kernel
        kernel[grid](
            q, k, v,
            m_sort, m_sort_indices.to(torch.int32),
            pad_offset, out,
            B, LQ, LK, HQ, HK, D, G,
            D ** -0.5,  # qk_scale
            IS_OFFLINE=is_offline,
            IS_CAUSAL=kwargs.get('causal', True),
            sQ_layout=sQ_layout,
            sK_layout=sK_layout,
            sV_layout=sV_layout,
            rQ_layout=rQ_layout,
            rK_layout=rK_layout,
            rP_layout=rP_layout,
            rV_layout=rV_layout,
            rO_layout=rO_layout,
        )
        
        return out
    
    @classmethod
    def _dense_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_offset: Optional[torch.Tensor] = None,
        BLOCK_M: int = 64,
        BLOCK_N: int = 64,
        num_stages: int = 2,
        num_warps: int = 4,
        **kwargs
    ):
        """Execute dense attention without pruning."""
        B, LQ, HQ, D = q.shape
        device = q.device
        
        # Create dummy mask (all ones)
        route_mask = torch.ones((B, LQ), dtype=torch.bool, device=device)
        
        return cls._sort_kernel(
            q, k, v,
            route_mask=route_mask,
            pad_offset=pad_offset,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
            is_offline=False,
            **kwargs
        )
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Main entry point for query pruning prefill attention.
        
        Args:
            impl: Implementation mode - 'dense', 'sort_online', or 'sort_offline'
            q: Query tensor [B, LQ, HQ, D]
            k: Key tensor [B, LK, HK, D]
            v: Value tensor [B, LK, HK, D]
            route_mask: Optional pruning mask [B, LQ] (bool)
            pad_offset: Optional left padding [B] (int32)
            estimated_sparsity: Expected fraction of active queries (0-1)
            causal: Whether to apply causal masking (default True)
            BLOCK_M, BLOCK_N: Tile sizes (auto-selected if not provided)
            num_stages: Pipeline depth (auto-selected if not provided)
            num_warps: Number of warps (auto-selected if not provided)
        
        Returns:
            Output tensor [B, LQ, HQ, D]
        """
        q = kwargs.get('q')
        k = kwargs.get('k')
        v = kwargs.get('v')
        
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        
        # Get or create padding offset
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None:
            pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        
        # Get pruning mask
        route_mask = kwargs.pop('route_mask', None)
        estimated_sparsity = kwargs.pop('estimated_sparsity', 1.0)
        
        # Auto-select implementation
        if impl == 'auto':
            if route_mask is None:
                impl = 'dense'
            else:
                impl = 'sort_offline'  # Default to offline for efficiency
        
        # Validate implementation
        if impl not in cls.support_kernel:
            raise ValueError(f"Implementation '{impl}' not supported. Use: {cls.support_kernel}")
        
        # Auto-configure block sizes based on workload
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        
        # Default values
        num_stages = kwargs.pop('num_stages', 2)
        num_warps = kwargs.pop('num_warps', 4)
        
        # Select BLOCK_M based on estimated sparsity
        if 'BLOCK_M' not in kwargs:
            if impl in ['sort_offline', 'sort_online'] and estimated_sparsity < 1.0:
                # Smaller blocks for better pruning granularity
                BLOCK_M = triton.next_power_of_2(int(LQ * estimated_sparsity))
                if BLOCK_M >= int(LQ * estimated_sparsity):
                    BLOCK_M = BLOCK_M // 2
                BLOCK_M = min(128, max(16, BLOCK_M))
            else:
                BLOCK_M = 64  # Default for dense
        else:
            BLOCK_M = kwargs.pop('BLOCK_M')
        
        # Select BLOCK_N
        if 'BLOCK_N' not in kwargs:
            BLOCK_N = min(64, max(16, triton.next_power_of_2(LK)))
        else:
            BLOCK_N = kwargs.pop('BLOCK_N')
        
        # Validate head dimension
        assert D <= 128, f"Head dimension {D} exceeds sm80 limit of 128"
        
        # Check shared memory constraints
        from ops.utils import check_shared_memory_attn
        while not check_shared_memory_attn(BLOCK_M, BLOCK_N, D, num_stages, dtype.itemsize):
            if BLOCK_N > 32:
                BLOCK_N = BLOCK_N // 2
            elif BLOCK_M > 32:
                BLOCK_M = BLOCK_M // 2
            else:
                num_stages = num_stages - 1
                if num_stages < 1:
                    raise RuntimeError("Cannot fit attention kernel in shared memory")
        
        # Ensure sufficient parallelism
        while HQ * B * triton.cdiv(LQ, BLOCK_M) < num_sm // 2:
            if BLOCK_M > 16:
                BLOCK_M = BLOCK_M // 2
            else:
                break
        
        # Determine offline mode
        is_offline = impl == 'sort_offline'
        
        # Extract kernel function name
        kernel_fn = impl.split('_')[0] if '_' in impl else impl
        
        # Launch appropriate kernel
        return getattr(cls, f'_{kernel_fn}_kernel')(
            route_mask=route_mask if impl != 'dense' else None,
            pad_offset=pad_offset,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
            is_offline=is_offline,
            **kwargs
        )


# =============================================================================
# Convenience Function
# =============================================================================

def query_pruning_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    route_mask: Optional[torch.Tensor] = None,
    impl: str = 'auto',
    **kwargs
) -> torch.Tensor:
    """
    Convenience function for query pruning prefill attention.
    
    Args:
        q, k, v: Input tensors [B, L, H, D]
        route_mask: Optional pruning mask [B, LQ]
        impl: 'dense', 'sort_online', 'sort_offline', or 'auto'
        **kwargs: Additional arguments passed to kernel
    
    Returns:
        Attention output [B, LQ, HQ, D]
    """
    return QueryPruningPrefill.kernel(impl, q=q, k=k, v=v, route_mask=route_mask, **kwargs)
