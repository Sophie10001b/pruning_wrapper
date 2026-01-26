import os
import itertools
import torch
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, Any
from einops import rearrange

os.environ['TRITON_PRINT_AUTOTUNING']='0'

def generate_autotune_config(tuning_dict: Dict):
    keys = list(tuning_dict.keys())
    value_lists = [tuning_dict[key] for key in keys]
    
    combinations = []
    for value_combination in itertools.product(*value_lists):
        combination_dict = dict(zip(keys, value_combination))
        num_stages = combination_dict.pop('num_stages', 4)
        num_warps = combination_dict.pop('num_warps', 4)
        combinations.append(triton.Config(combination_dict, num_stages=num_stages, num_warps=num_warps))
    
    return combinations

@triton.autotune(
    configs=generate_autotune_config(
        dict(
            BLOCK_M=[1, 2, 4, 8],
            num_stages=[3, 4],
        )
    ),
    key=['HQ', 'HK', 'N'],
)
@triton.jit(do_not_specialize=['B', 'L', 'M'])
def triton_rope_qk_align_fwd(
    q: tl.tensor,
    k: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    out_q_cos: tl.tensor,
    out_q_sin: tl.tensor,
    out_k_cos: tl.tensor,
    out_k_sin: tl.tensor,
    B: tl.int64,
    L: tl.int64,
    M: tl.int64,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    mid = tl.program_id(0)
    head_id_q = tl.program_id(1)
    head_id_k = head_id_q // G

    offset_m = (mid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    batch_id = offset_m // L
    pos_id = offset_m % L

    offset_emb = pos_id * N
    offset_q = offset_m * HQ * N + head_id_q * N
    offset_k = offset_m * HK * N + head_id_k * N
    emb_mask = (batch_id < B) & (pos_id < L)

    cos_left = tl.load(cos + offset_emb[:, None] + tl.arange(0, N // 2)[None, :], mask=emb_mask[:, None], other=0)
    cos_right = tl.load(cos + offset_emb[:, None] + tl.arange(N // 2, N)[None, :], mask=emb_mask[:, None], other=0)
    sin_left = tl.load(sin + offset_emb[:, None] + tl.arange(0, N // 2)[None, :], mask=emb_mask[:, None], other=0)
    sin_right = tl.load(sin + offset_emb[:, None] + tl.arange(N // 2, N)[None, :], mask=emb_mask[:, None], other=0)

    q_data_left = tl.load(q + offset_q[:, None] + tl.arange(0, N // 2)[None, :], mask=(offset_m < M)[:, None], other=0)
    q_data_right = tl.load(q + offset_q[:, None] + tl.arange(N // 2, N)[None, :], mask=(offset_m < M)[:, None], other=0)
    k_data_left = tl.load(k + offset_k[:, None] + tl.arange(0, N // 2)[None, :], mask=(offset_m < M)[:, None], other=0)
    k_data_right = tl.load(k + offset_k[:, None] + tl.arange(N // 2, N)[None, :], mask=(offset_m < M)[:, None], other=0)

    q_left_cos = q_data_left * cos_left
    q_left_sin = -q_data_right * sin_left
    q_right_cos = q_data_right * cos_right
    q_right_sin = q_data_left * sin_right

    k_left_cos = k_data_left * cos_left
    k_left_sin = -k_data_right * sin_left
    k_right_cos = k_data_right * cos_right
    k_right_sin = k_data_left * sin_right

    store_mask = (offset_m < M)[:, None]

    tl.store(out_q_cos + offset_q[:, None] + tl.arange(0, N // 2)[None, :], q_left_cos.to(out_q_cos.dtype.element_ty), mask=store_mask)
    tl.store(out_q_cos + offset_q[:, None] + tl.arange(N // 2, N)[None, :], q_right_cos.to(out_q_cos.dtype.element_ty), mask=store_mask)
    tl.store(out_q_sin + offset_q[:, None] + tl.arange(0, N // 2)[None, :], q_left_sin.to(out_q_sin.dtype.element_ty), mask=store_mask)
    tl.store(out_q_sin + offset_q[:, None] + tl.arange(N // 2, N)[None, :], q_right_sin.to(out_q_sin.dtype.element_ty), mask=store_mask)
    
    tl.store(out_k_cos + offset_k[:, None] + tl.arange(0, N // 2)[None, :], k_left_cos.to(out_k_cos.dtype.element_ty), mask=store_mask)
    tl.store(out_k_cos + offset_k[:, None] + tl.arange(N // 2, N)[None, :], k_right_cos.to(out_k_sin.dtype.element_ty), mask=store_mask)
    tl.store(out_k_sin + offset_k[:, None] + tl.arange(0, N // 2)[None, :], k_left_sin.to(out_k_sin.dtype.element_ty), mask=store_mask)
    tl.store(out_k_sin + offset_k[:, None] + tl.arange(N // 2, N)[None, :], k_right_sin.to(out_k_sin.dtype.element_ty), mask=store_mask)

# @torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False)
def triton_rope_qk_align(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to qk, avoiding apply addition in triton kernel to match the pytorch precision.
    Args:
        q: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
        k: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
        sin: torch.Tensor with shape [1, seqlen, head_dim]
        cos: torch.Tensor with shape [1, seqlen, head_dim]
    Returns:
        out_q: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]\\
        out_k: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
    """

    out_q_cos, out_q_sin = torch.empty_like(q), torch.empty_like(q)
    out_k_cos, out_k_sin = torch.empty_like(k), torch.empty_like(k)

    assert q.dim() == k.dim() == 3
    assert sin.dim() == cos.dim()

    if sin.dim() == 3: assert sin.shape[0] == 1
    else: assert sin.dim() == 2

    B, L, HQ, N = q.shape
    HK = k.shape[-2]
    G = HQ // HK
    M = q.shape[0] * q.shape[1]
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), HQ)
    
    triton_rope_qk_align_fwd[grid](
        q, k,
        cos,
        sin,
        out_q_cos, out_q_sin,
        out_k_cos, out_k_sin,
        B, L, M, HQ, HK, N, G,
    )
    return out_q_cos + out_q_sin, out_k_cos + out_k_sin

@triton.autotune(
    configs=generate_autotune_config(
        dict(
            BLOCK_M=[1, 2, 4, 8],
            num_stages=[3, 4],
        )
    ),
    key=['N', 'BLOCK_N'],
)
@triton.jit(do_not_specialize=['eps', 'M'])
def triton_rmsnorm_fwd(
    x: tl.tensor,
    out: tl.tensor,
    w: tl.tensor,
    eps: tl.float32,
    M: tl.int64,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    mid = tl.program_id(0)
    x_ptr = tl.make_block_ptr(
        x,
        shape=(M, N),
        strides=(N, 1),
        offsets=(mid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x_data = tl.load(tl.advance(x_ptr, (0, i * BLOCK_N)), boundary_check=(0, 1), padding_option='zero')
        accum += x_data * x_data
    
    x_mean = tl.sum(accum, axis=1, keep_dims=True) / N + eps

    w_ptr = tl.make_block_ptr(
        w,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    out_ptr = tl.make_block_ptr(
        out,
        shape=(M, N),
        strides=(N, 1),
        offsets=(mid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    x_rsqrt = tl.math.rsqrt(x_mean)
    for i in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x_data = tl.load(tl.advance(x_ptr, (0, i * BLOCK_N)), boundary_check=(0, 1), padding_option='zero')
        w_data = tl.load(tl.advance(w_ptr, (i * BLOCK_N,)), boundary_check=(0,), padding_option='zero')
        x_norm = x_data * x_rsqrt * w_data[None, :]

        tl.store(tl.advance(out_ptr, (0, i * BLOCK_N)), x_norm.to(x.dtype.element_ty), boundary_check=(0, 1))

def triton_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    An naive implementation of RMSNorm for the final dimension
    """
    original_shape = x.shape
    if x.dim() > 2: x_flatten = x.flatten(0, -2)
    else: x_flatten = x

    M, N = x_flatten.shape
    assert N == w.shape[0]

    out = torch.empty_like(x_flatten)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    triton_rmsnorm_fwd[grid](
        x,
        out,
        w,
        eps,
        M, N,
        BLOCK_N=min(triton.next_power_of_2(N), 1024)
    )
    return out.reshape(original_shape)