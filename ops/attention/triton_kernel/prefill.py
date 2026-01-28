import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, List, Any

def generate_autotune_config(tuning_dict: Dict):
    keys = list(tuning_dict.keys())
    value_lists = [tuning_dict[key] for key in keys]
    
    combinations = []
    for value_combination in itertools.product(*value_lists):
        combination_dict = dict(zip(keys, value_combination))
        num_stages = combination_dict.pop('num_stages', 3)
        num_warps = combination_dict.pop('num_warps', 4)
        combinations.append(triton.Config(combination_dict, num_stages=num_stages, num_warps=num_warps))
    
    return combinations


class QuerySparsePrefill:

    support_kernel = [
        'ragged',
        'dense',
        'sort_offline',
        'sort_online',
    ]

    @classmethod
    def _ragged_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit
        def chunking(
            chunk_num: tl.tensor,
            chunk_cumsum: tl.tensor,
            out: tl.tensor,
        ):
            bid = tl.program_id(0)

            num = tl.load(chunk_num + bid)
            prev_count = tl.load(chunk_cumsum + bid - 1, mask=bid > 0, other=0)

            for i in tl.range(num):
                tl.store(out + (prev_count + i) * 2, bid)
                tl.store(out + (prev_count + i) * 2 + 1, i)
        

        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            cu_seqlens_q: tl.tensor,
            pos_indices: tl.tensor,
            chunk_indices: tl.tensor,
            pad_offset: tl.tensor,
            out: tl.tensor,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            query_head_id, chunk_id = tl.program_id(0), tl.program_id(1)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            qk_scale *= 1.44269504 # 1/log(2)
            key_head_id = query_head_id // G

            batch_id = tl.load(chunk_indices + chunk_id * 2)
            chunk_id = tl.load(chunk_indices + chunk_id * 2 + 1)

            start_q, end_q = tl.load(cu_seqlens_q + batch_id), tl.load(cu_seqlens_q + batch_id + 1)
            seq_offset_q = start_q + chunk_id * BLOCK_M

            pad_offset_k = tl.load(pad_offset + batch_id)
            start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
            key_length = end_k - start_k

            pos_idx_q = tl.load(pos_indices + seq_offset_q + tl.arange(0, BLOCK_M), mask=(seq_offset_q + tl.arange(0, BLOCK_M)) < end_q, other=-1)
            max_pos_q = tl.max(pos_idx_q)

            query_range_mask = (seq_offset_q + tl.arange(0, BLOCK_M)) < end_q
            query_data = tl.load(
                q + ((seq_offset_q + tl.arange(0, BLOCK_M)) * HQ * D + query_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                mask=query_range_mask[:, None],
                other=0.0
            )

            split_n_range = tl.cdiv(max_pos_q - pad_offset_k + 1, BLOCK_N)
            for tile_n in tl.range(0, split_n_range):
                key_range_mask = ((tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
                key_data = tl.load(
                    k + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=key_range_mask[:, None],
                    other=0.0
                )
                value_data = tl.load(
                    v + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=key_range_mask[:, None],
                    other=0.0
                )

                qk = tl.dot(query_data, key_data.T) * qk_scale
                
                # causal mask for each query pos
                causal_mask = pos_idx_q[:, None] >= (tile_n * BLOCK_N + pad_offset_k + tl.arange(0, BLOCK_N))[None, :]
                qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

                score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                score_scale = tl.exp2(score_max - score_max_new)
                qk = tl.exp2(qk - score_max_new[:, None])
                score_sum = score_sum * score_scale + tl.sum(qk, 1)
                acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                score_max = score_max_new
            
            acc /= score_sum[:, None]
            tl.store(
                out + ((seq_offset_q + tl.arange(0, BLOCK_M)) * HQ * D + query_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                acc.to(out.dtype.element_ty),
                mask=query_range_mask[:, None]
            )
        
        if q.dim() == 4:
            all_indices = torch.nonzero(route_mask)
            flatten_q = q[all_indices[:, 0], all_indices[:, 1]]
            seqlens_q = route_mask.sum(-1)
            cu_seqlens_q = torch.nn.functional.pad(seqlens_q.cumsum(-1), (1, 0), mode='constant', value=0)
            pos_indices = all_indices[:, -1]
        else:
            flatten_q = q
            cu_seqlens_q = kwargs.get('cu_seqlens_q')
            pos_indices = kwargs.get('pos_indices')
        
        NQ, HQ, D = flatten_q.shape
        B, LK, HK, _ = k.shape
        G = HQ // HK

        assert cu_seqlens_q[-1] == NQ
        chunk_num = triton.cdiv(cu_seqlens_q[1:] - cu_seqlens_q[:-1], BLOCK_M)
        chunk_cumsum = chunk_num.cumsum(0)
        C = chunk_cumsum[-1].item()

        chunk_indices = torch.empty([C, 2], dtype=torch.int32, device=cu_seqlens_q.device)
        grid = lambda META: (B,)
        chunking[grid](chunk_num, chunk_cumsum, chunk_indices)

        out = torch.zeros_like(flatten_q)
        grid = lambda META: (HQ, C)
        attention[grid](
            flatten_q, k, v,
            cu_seqlens_q, pos_indices, chunk_indices, pad_offset, out,
            LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out

    @classmethod    
    def _dense_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=['C'])
        def chunking(
            cu_seqlens: tl.tensor,
            cu_chunk_num: tl.tensor,
            cu_pos_indices: tl.tensor,
            chunk_indices: tl.tensor,
            C: tl.int64,
        ):
            bid = tl.program_id(0)

            start_idx = tl.load(cu_seqlens + bid - 1, mask=bid > 0, other=0)

            start_chunk_num = tl.load(cu_chunk_num + bid - 1, mask=bid > 0, other=0)
            end_chunk_num = tl.load(cu_chunk_num + bid)
            total_chunk_num = end_chunk_num - start_chunk_num

            for i in tl.range(total_chunk_num):
                tl.store(cu_pos_indices + start_chunk_num + i, start_idx + C * i)
                tl.store(chunk_indices + start_chunk_num + i, bid)

        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            pos_indices: tl.tensor, # [total_active_tokens,] with the exact pos indices of each active tokens
            cu_pos_indices: tl.tensor, # [chunk_num + 1,]
            chunk_indices: tl.tensor, # [chunk_num,] with [0] -> batch_id
            pad_offset: tl.tensor,
            out: tl.tensor,
            LQ: tl.int64,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            query_head_id, cid = tl.program_id(0), tl.program_id(1)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            qk_scale *= 1.44269504 # 1/log(2)

            batch_id = tl.load(chunk_indices + cid)
            key_head_id = query_head_id // G

            start_q, end_q = batch_id * LQ, (batch_id + 1) * LQ

            pad_offset_k = tl.load(pad_offset + batch_id)
            start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
            key_length = end_k - start_k

            start_pos_indices, end_pos_indices = tl.load(cu_pos_indices + cid), tl.load(cu_pos_indices + cid + 1)
            pos_idx_q = tl.load(
                pos_indices + start_pos_indices + tl.arange(0, BLOCK_M),
                mask=(start_pos_indices + tl.arange(0, BLOCK_M)) < end_pos_indices,
                other=-1
            )
            max_pos_q = tl.max(pos_idx_q)

            query_range_mask = pos_idx_q >= 0
            query_data = tl.load(
                q + ((start_q + pos_idx_q) * HQ * D + query_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                mask=query_range_mask[:, None],
                other=0.0
            )

            split_n_range = tl.cdiv(max_pos_q - pad_offset_k + 1, BLOCK_N)
            for tile_n in tl.range(0, split_n_range):
                key_range_mask = ((tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
                key_data = tl.load(
                    k + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=key_range_mask[:, None],
                    other=0.0
                )
                value_data = tl.load(
                    v + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=key_range_mask[:, None],
                    other=0.0
                )

                qk = tl.dot(query_data, key_data.T) * qk_scale
                
                # causal mask for each query pos
                causal_mask = pos_idx_q[:, None] >= (tile_n * BLOCK_N + pad_offset_k + tl.arange(0, BLOCK_N))[None, :]
                qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

                score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                score_scale = tl.exp2(score_max - score_max_new)
                qk = tl.exp2(qk - score_max_new[:, None])
                score_sum = score_sum * score_scale + tl.sum(qk, 1)
                acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                score_max = score_max_new
            
            acc /= score_sum[:, None]
            tl.store(
                out + ((start_q + pos_idx_q) * HQ * D + query_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                acc.to(out.dtype.element_ty),
                mask=query_range_mask[:, None]
            )
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        route_indices = route_mask.nonzero(as_tuple=False)
        valid_token_num = route_indices[:, 0].bincount()
        valid_token_num = valid_token_num[valid_token_num.ne(0)]
        cu_valid_token_num = torch.cumsum(valid_token_num, 0)
        chunk_num = triton.cdiv(valid_token_num, BLOCK_M)
        cu_chunk_num = torch.cumsum(chunk_num, 0)

        C = cu_chunk_num[-1].item()
        cu_pos_indices = torch.empty((C+1,), dtype=torch.int64, device=route_mask.device)
        chunk_indices = torch.empty((C,), dtype=torch.int32, device=route_mask.device)
        grid = lambda META: (B,)
        chunking[grid](cu_valid_token_num, cu_chunk_num, cu_pos_indices, chunk_indices, BLOCK_M)
        cu_pos_indices[-1] = cu_valid_token_num[-1]
        pos_indices = route_indices[:, -1]

        out = torch.zeros_like(q)
        grid = lambda META: (HQ, C)
        attention[grid](
            q, k, v,
            pos_indices, cu_pos_indices, chunk_indices, pad_offset, out,
            LQ, LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
    
    @classmethod    
    def _sort_offline_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor, # [B, cdiv(LQ, BLOCK_M)]
            route_indices: tl.tensor, # [B, LQ]
            pad_offset: tl.tensor, # [B]
            out: tl.tensor,
            LQ: tl.int64,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            query_id, query_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            BLOCK_NUM = tl.num_programs(0)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            qk_scale *= 1.44269504 # 1/log(2)
            key_head_id = query_head_id // G

            skip_flag = tl.load(route_mask + batch_id * BLOCK_NUM + query_id)
            if skip_flag > 0:
                query_batch_offset = batch_id * LQ * HQ * D
                query_head_offset = query_head_id * D
                key_batch_offset = batch_id * LK * HK * D
                key_head_offset = key_head_id * D

                pad_offset_kv = tl.load(pad_offset + batch_id)
                key_length = LK - pad_offset_kv

                query_indices = tl.load(
                    route_indices + batch_id * LQ + query_id * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=query_id * BLOCK_M + tl.arange(0, BLOCK_M) < LQ,
                    other=-1,
                )
                query_range_mask = query_indices >= 0
                query_indices = tl.where(query_range_mask, query_indices, 0)

                query_data = tl.load(
                    q + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]),
                )

                max_pos_q = tl.max(query_indices)
                num_kv_iter = tl.cdiv(max_pos_q - pad_offset_kv + 1, BLOCK_N)
                for tile_kv in tl.range(0, num_kv_iter):
                    key_range_mask = ((tile_kv * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
                    key_data = tl.load(
                        k + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                        mask=key_range_mask[:, None],
                        other=0.0,
                    )
                    value_data = tl.load(
                        v + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                        mask=key_range_mask[:, None],
                        other=0.0,
                    )

                    qk = tl.dot(query_data, key_data.T) * qk_scale

                    # causal mask for each query pos
                    causal_mask = query_indices[:, None] >= (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[None, :]
                    qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

                    score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new[:, None])
                    score_sum = score_sum * score_scale + tl.sum(qk, 1)
                    acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                    score_max = score_max_new

                acc /= score_sum[:, None]
                tl.store(
                    out + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]),
                    acc.to(out.dtype.element_ty),
                    mask=query_range_mask[:, None],
                )
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        m_sort, m_sort_indices = torch.sort(route_mask, descending=True, stable=False) # [B, LQ]
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - LQ % BLOCK_M), value=0).reshape(B, -1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = torch.zeros_like(q)
        grid = lambda META: (triton.cdiv(LQ, BLOCK_M), HQ, B)
        attention[grid](
            q, k, v,
            m_sort_pad, m_sort_indices, pad_offset, out,
            LQ, LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
    
    @classmethod    
    def _sort_online_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor, # [B, LQ]
            route_indices: tl.tensor, # [B, LQ]
            pad_offset: tl.tensor, # [B]
            out: tl.tensor,
            LQ: tl.int64,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            query_id, query_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            qk_scale *= 1.44269504 # 1/log(2)
            key_head_id = query_head_id // G

            query_mask = tl.load(
                route_mask + batch_id * LQ + query_id * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=query_id * BLOCK_M + tl.arange(0, BLOCK_M) < LQ,
                other=0,
            )
            skip_flag = tl.reduce_or(query_mask, axis=-1)
            if skip_flag > 0:
                query_batch_offset = batch_id * LQ * HQ * D
                query_head_offset = query_head_id * D
                key_batch_offset = batch_id * LK * HK * D
                key_head_offset = key_head_id * D

                pad_offset_kv = tl.load(pad_offset + batch_id)
                key_length = LK - pad_offset_kv

                query_range_mask = query_mask > 0
                query_indices = tl.load(
                    route_indices + batch_id * LQ + query_id * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=query_range_mask,
                    other=0,
                )

                query_data = tl.load(
                    q + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]),
                )

                max_pos_q = tl.max(query_indices)
                num_kv_iter = tl.cdiv(max_pos_q - pad_offset_kv + 1, BLOCK_N)
                for tile_kv in tl.range(0, num_kv_iter):
                    key_range_mask = ((tile_kv * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
                    key_data = tl.load(
                        k + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                        mask=key_range_mask[:, None],
                        other=0.0,
                    )
                    value_data = tl.load(
                        v + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                        mask=key_range_mask[:, None],
                        other=0.0,
                    )

                    qk = tl.dot(query_data, key_data.T) * qk_scale

                    # causal mask for each query pos
                    causal_mask = query_indices[:, None] >= (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[None, :]
                    qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

                    score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new[:, None])
                    score_sum = score_sum * score_scale + tl.sum(qk, 1)
                    acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                    score_max = score_max_new

                acc /= score_sum[:, None]
                tl.store(
                    out + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]),
                    acc.to(out.dtype.element_ty),
                    mask=query_range_mask[:, None],
                )
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        m_sort, m_sort_indices = torch.sort(route_mask, descending=True, stable=False) # [B, LQ]

        out = torch.zeros_like(q)
        grid = lambda META: (triton.cdiv(LQ, BLOCK_M), HQ, B)
        attention[grid](
            q, k, v,
            m_sort, m_sort_indices, pad_offset, out,
            LQ, LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for query sparse attention (prefill)\\
        **ragged:**\\
        Return [nnz, num_query_heads, head_dim]\\
        **dense & sort**\\
        Return [batch_size, query_length, num_query_heads, head_dim]
        Args:
            q, k, v: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
            route_mask: Optional[torch.Tensor] with shape [batch_size, seqlen], with 1 for active tokens
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        
        q = kwargs.get('q')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        route_mask = kwargs.pop('route_mask', None)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]
        num_stages = 2
        num_warps = 4

        if impl == 'auto': impl = 'sort_online'

        estimated_sparsity = kwargs.pop('estimated_sparsity', 0)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, LQ), dtype=torch.bool, device=device)

        BLOCK_M = triton.next_power_of_2(int(LQ * estimated_sparsity))
        if BLOCK_M > int(LQ * estimated_sparsity): BLOCK_M = BLOCK_M >> 1
        BLOCK_M = min(128, max(16, BLOCK_M))
        BLOCK_N = min(32, max(16, triton.next_power_of_2(LQ)))
        
        BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)

        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            pad_offset=pad_offset,
            **kwargs
        )

class GroupSparsePrefill:

    support_kernel = [
        'dense',
        'sort',
    ]

    @classmethod    
    def _dense_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=['C'])
        def chunking(
            cu_seqlens: tl.tensor,
            cu_chunk_num: tl.tensor,
            cu_pos_indices: tl.tensor,
            chunk_indices: tl.tensor,
            H: tl.constexpr,
            C: tl.int64,
        ):
            nid = tl.program_id(0)
            gid, bid = nid % H, nid // H

            start_idx = tl.load(cu_seqlens + nid - 1, mask=nid > 0, other=0)

            start_chunk_num = tl.load(cu_chunk_num + nid - 1, mask=nid > 0, other=0)
            end_chunk_num = tl.load(cu_chunk_num + nid)
            total_chunk_num = end_chunk_num - start_chunk_num

            for i in tl.range(total_chunk_num):
                tl.store(cu_pos_indices + start_chunk_num + i, start_idx + C * i)
                tl.store(chunk_indices + (start_chunk_num + i) * 2, bid)
                tl.store(chunk_indices + (start_chunk_num + i) * 2 + 1, gid)

        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            pos_indices: tl.tensor, # [total_active_heads,] with the exact pos indices of each active token
            cu_pos_indices: tl.tensor, # [chunk_num + 1,]
            chunk_indices: tl.tensor, # [chunk_num, 2] with [0] -> batch_id, [1] -> head_id
            pad_offset: tl.tensor,
            out: tl.tensor,
            LQ: tl.int64,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            cid = tl.program_id(0)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            qk_scale *= 1.44269504 # 1/log(2)

            batch_id = tl.load(chunk_indices + cid * 2)
            query_head_id = tl.load(chunk_indices + cid * 2 + 1)
            key_head_id = query_head_id // G

            start_q, end_q = batch_id * LQ, (batch_id + 1) * LQ

            pad_offset_k = tl.load(pad_offset + batch_id)
            start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
            key_length = end_k - start_k

            start_pos_indices, end_pos_indices = tl.load(cu_pos_indices + cid), tl.load(cu_pos_indices + cid + 1)
            pos_idx_q = tl.load(
                pos_indices + start_pos_indices + tl.arange(0, BLOCK_M),
                mask=(start_pos_indices + tl.arange(0, BLOCK_M)) < end_pos_indices,
                other=-1
            )
            max_pos_q = tl.max(pos_idx_q)

            query_range_mask = pos_idx_q >= 0
            query_data = tl.load(
                q + ((start_q + pos_idx_q) * HQ * D + query_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                mask=query_range_mask[:, None],
                other=0.0
            )

            split_n_range = tl.cdiv(max_pos_q - pad_offset_k + 1, BLOCK_N)
            for tile_n in tl.range(0, split_n_range):
                key_range_mask = ((tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
                key_data = tl.load(
                    k + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=key_range_mask[:, None],
                    other=0.0
                )
                value_data = tl.load(
                    v + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=key_range_mask[:, None],
                    other=0.0
                )

                qk = tl.dot(query_data, key_data.T) * qk_scale
                
                # causal mask for each query pos
                causal_mask = pos_idx_q[:, None] >= (tile_n * BLOCK_N + pad_offset_k + tl.arange(0, BLOCK_N))[None, :]
                qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

                score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                score_scale = tl.exp2(score_max - score_max_new)
                qk = tl.exp2(qk - score_max_new[:, None])
                score_sum = score_sum * score_scale + tl.sum(qk, 1)
                acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                score_max = score_max_new
            
            acc /= score_sum[:, None]
            tl.store(
                out + ((start_q + pos_idx_q) * HQ * D + query_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                acc.to(out.dtype.element_ty),
                mask=query_range_mask[:, None]
            )
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        expanded_route_mask = route_mask.repeat_interleave(G, dim=-1) # expand to head-level for better parallelism
        trans_mask = expanded_route_mask.transpose(1, 2).flatten(0, 1)
        trans_indices = trans_mask.nonzero(as_tuple=False)
        valid_token_num = trans_indices[:, 0].bincount()
        valid_token_num = valid_token_num[valid_token_num.ne(0)]
        cu_valid_token_num = torch.cumsum(valid_token_num, 0)
        chunk_num = triton.cdiv(valid_token_num, BLOCK_M)
        cu_chunk_num = torch.cumsum(chunk_num, 0)

        C = cu_chunk_num[-1].item()
        cu_pos_indices = torch.empty((C+1,), dtype=torch.int64, device=route_mask.device)
        chunk_indices = torch.empty((C, 2), dtype=torch.int32, device=route_mask.device)

        grid = lambda META: (valid_token_num.shape[0],)
        chunking[grid](cu_valid_token_num, cu_chunk_num, cu_pos_indices, chunk_indices, HQ, BLOCK_M)
        cu_pos_indices[-1] = cu_valid_token_num[-1]
        pos_indices = trans_indices[:, -1]

        out = torch.zeros_like(q)
        grid = lambda META: (C,)
        attention[grid](
            q, k, v,
            pos_indices, cu_pos_indices, chunk_indices, pad_offset, out,
            LQ, LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
    
    @classmethod    
    def _sort_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor, # [B, HK, cdiv(LQ, BLOCK_M)]
            route_indices: tl.tensor, # [B, HK, LQ]
            pad_offset: tl.tensor, # [B]
            out: tl.tensor,
            LQ: tl.int64,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            query_id, query_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            BLOCK_NUM = tl.num_programs(0)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            qk_scale *= 1.44269504 # 1/log(2)
            key_head_id = query_head_id // G

            skip_flag = tl.load(
                route_mask + batch_id * HK * BLOCK_NUM + key_head_id * BLOCK_NUM + query_id,
            )
            if skip_flag > 0:
                query_batch_offset = batch_id * LQ * HQ * D
                query_head_offset = query_head_id * D
                key_batch_offset = batch_id * LK * HK * D
                key_head_offset = key_head_id * D

                pad_offset_kv = tl.load(pad_offset + batch_id)
                key_length = LK - pad_offset_kv

                query_indices = tl.load(
                    route_indices + batch_id * HK * LQ + key_head_id * LQ + query_id * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=query_id * BLOCK_M + tl.arange(0, BLOCK_M) < LQ,
                    other=-1,
                )
                query_range_mask = query_indices >= 0
                query_indices = tl.where(query_range_mask, query_indices, 0)

                query_data = tl.load(
                    q + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]),
                )

                max_pos_q = tl.max(query_indices)
                num_kv_iter = tl.cdiv(max_pos_q - pad_offset_kv + 1, BLOCK_N)

                for tile_kv in tl.range(0, num_kv_iter):
                    key_range_mask = ((tile_kv * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
                    key_data = tl.load(
                        k + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                        mask=key_range_mask[:, None],
                        other=0.0,
                    )
                    value_data = tl.load(
                        v + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                        mask=key_range_mask[:, None],
                        other=0.0,
                    )

                    qk = tl.dot(query_data, key_data.T) * qk_scale

                    # causal mask for each query pos
                    causal_mask = query_indices[:, None] >= (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[None, :]
                    qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

                    score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new[:, None])
                    score_sum = score_sum * score_scale + tl.sum(qk, 1)
                    acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                    score_max = score_max_new

                acc /= score_sum[:, None]
                tl.store(
                    out + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]),
                    acc.to(out.dtype.element_ty),
                    mask=query_range_mask[:, None],
                )
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        m_sort, m_sort_indices = torch.sort(route_mask.transpose(1, 2).contiguous(), descending=True, stable=False) # [B, HK, LQ]
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - LQ % BLOCK_M), value=0).reshape(B, HK, -1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = torch.zeros_like(q)
        grid = lambda META: (triton.cdiv(LQ, BLOCK_M), HQ, B)
        attention[grid](
            q, k, v,
            m_sort_pad, m_sort_indices, pad_offset, out,
            LQ, LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for group sparse attention (prefill)\\
        **dense & sort**\\
        Return [batch_size, query_length, num_query_heads, head_dim]
        Args:
            q: torch.Tensor with shape [batch_size, query_length, num_query_heads, head_dim]
            k: torch.Tensor with shape [batch_size, query_length, num_key_heads, head_dim]
            v: torch.Tensor with shape [batch_size, query_length, num_key_heads, head_dim]
            route_mask: Optional[torch.Tensor] with shape [batch_size, seqlen, num_key_heads], with 1 for active group
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        
        q = kwargs.get('q')
        k = kwargs.get('k')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        route_mask = kwargs.pop('route_mask', None)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]
        num_stages = 2
        num_warps = 4

        if impl == 'auto': impl = 'sort'

        estimated_sparsity = kwargs.pop('estimated_sparsity', 0)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, LQ, HK), dtype=torch.bool, device=device)

        BLOCK_M = triton.next_power_of_2(int(LQ * estimated_sparsity))
        if BLOCK_M > int(LQ * estimated_sparsity): BLOCK_M = BLOCK_M >> 1
        BLOCK_M = min(128, max(16, BLOCK_M))
        BLOCK_N = min(32, max(16, triton.next_power_of_2(LQ)))
        
        BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)

        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            pad_offset=pad_offset,
            **kwargs
        )


class HeadSparsePrefill:

    support_kernel = [
        'dense',
        'sort',
    ]

    @classmethod    
    def _dense_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=['C'])
        def chunking(
            cu_seqlens: tl.tensor,
            cu_chunk_num: tl.tensor,
            cu_pos_indices: tl.tensor,
            chunk_indices: tl.tensor,
            H: tl.constexpr,
            C: tl.int64,
        ):
            nid = tl.program_id(0)
            hid, bid = nid % H, nid // H

            start_idx = tl.load(cu_seqlens + nid - 1, mask=nid > 0, other=0)

            start_chunk_num = tl.load(cu_chunk_num + nid - 1, mask=nid > 0, other=0)
            end_chunk_num = tl.load(cu_chunk_num + nid)
            total_chunk_num = end_chunk_num - start_chunk_num

            for i in tl.range(total_chunk_num):
                tl.store(cu_pos_indices + start_chunk_num + i, start_idx + C * i)
                tl.store(chunk_indices + (start_chunk_num + i) * 2, bid)
                tl.store(chunk_indices + (start_chunk_num + i) * 2 + 1, hid)

        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            pos_indices: tl.tensor, # [total_active_heads,] with the exact pos indices of each active token
            cu_pos_indices: tl.tensor, # [chunk_num + 1,]
            chunk_indices: tl.tensor, # [chunk_num, 2] with [0] -> batch_id, [1] -> head_id
            pad_offset: tl.tensor,
            out: tl.tensor,
            LQ: tl.int64,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            cid = tl.program_id(0)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            qk_scale *= 1.44269504 # 1/log(2)

            batch_id = tl.load(chunk_indices + cid * 2)
            query_head_id = tl.load(chunk_indices + cid * 2 + 1)
            key_head_id = query_head_id // G

            start_q, end_q = batch_id * LQ, (batch_id + 1) * LQ

            pad_offset_k = tl.load(pad_offset + batch_id)
            start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
            key_length = end_k - start_k

            start_pos_indices, end_pos_indices = tl.load(cu_pos_indices + cid), tl.load(cu_pos_indices + cid + 1)
            pos_idx_q = tl.load(
                pos_indices + start_pos_indices + tl.arange(0, BLOCK_M),
                mask=(start_pos_indices + tl.arange(0, BLOCK_M)) < end_pos_indices,
                other=-1
            )
            max_pos_q = tl.max(pos_idx_q)

            query_range_mask = pos_idx_q >= 0
            query_data = tl.load(
                q + ((start_q + pos_idx_q) * HQ * D + query_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                mask=query_range_mask[:, None],
                other=0.0
            )

            split_n_range = tl.cdiv(max_pos_q - pad_offset_k + 1, BLOCK_N)
            for tile_n in tl.range(0, split_n_range):
                key_range_mask = ((tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
                key_data = tl.load(
                    k + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=key_range_mask[:, None],
                    other=0.0
                )
                value_data = tl.load(
                    v + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=key_range_mask[:, None],
                    other=0.0
                )

                qk = tl.dot(query_data, key_data.T) * qk_scale
                
                # causal mask for each query pos
                causal_mask = pos_idx_q[:, None] >= (tile_n * BLOCK_N + pad_offset_k + tl.arange(0, BLOCK_N))[None, :]
                qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

                score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                score_scale = tl.exp2(score_max - score_max_new)
                qk = tl.exp2(qk - score_max_new[:, None])
                score_sum = score_sum * score_scale + tl.sum(qk, 1)
                acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                score_max = score_max_new
            
            acc /= score_sum[:, None]
            tl.store(
                out + ((start_q + pos_idx_q) * HQ * D + query_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                acc.to(out.dtype.element_ty),
                mask=query_range_mask[:, None]
            )
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        trans_mask = route_mask.transpose(1, 2).flatten(0, 1)
        trans_indices = trans_mask.nonzero(as_tuple=False)
        valid_token_num = trans_indices[:, 0].bincount()
        valid_token_num = valid_token_num[valid_token_num.ne(0)]
        cu_valid_token_num = torch.cumsum(valid_token_num, 0)
        chunk_num = triton.cdiv(valid_token_num, BLOCK_M)
        cu_chunk_num = torch.cumsum(chunk_num, 0)

        C = cu_chunk_num[-1].item()
        cu_pos_indices = torch.empty((C+1,), dtype=torch.int64, device=route_mask.device)
        chunk_indices = torch.empty((C, 2), dtype=torch.int32, device=route_mask.device)

        grid = lambda META: (valid_token_num.shape[0],)
        chunking[grid](cu_valid_token_num, cu_chunk_num, cu_pos_indices, chunk_indices, HQ, BLOCK_M)
        cu_pos_indices[-1] = cu_valid_token_num[-1]
        pos_indices = trans_indices[:, -1]

        out = torch.zeros_like(q)
        grid = lambda META: (C,)
        attention[grid](
            q, k, v,
            pos_indices, cu_pos_indices, chunk_indices, pad_offset, out,
            LQ, LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
    
    @classmethod    
    def _sort_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor, # [B, HQ, cdiv(LQ, BLOCK_M)]
            route_indices: tl.tensor, # [B, HQ, LQ]
            pad_offset: tl.tensor, # [B]
            out: tl.tensor,
            LQ: tl.int64,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            query_id, query_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            BLOCK_NUM = tl.num_programs(0)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            qk_scale *= 1.44269504 # 1/log(2)
            key_head_id = query_head_id // G

            skip_flag = tl.load(
                route_mask + batch_id * HQ * BLOCK_NUM + query_head_id * BLOCK_NUM + query_id,
            )
            if skip_flag > 0:
                query_batch_offset = batch_id * LQ * HQ * D
                query_head_offset = query_head_id * D
                key_batch_offset = batch_id * LK * HK * D
                key_head_offset = key_head_id * D

                pad_offset_kv = tl.load(pad_offset + batch_id)
                key_length = LK - pad_offset_kv

                query_indices = tl.load(
                    route_indices + batch_id * HQ * LQ + query_head_id * LQ + query_id * BLOCK_M + tl.arange(0, BLOCK_M),
                    mask=query_id * BLOCK_M + tl.arange(0, BLOCK_M) < LQ,
                    other=-1,
                )
                query_range_mask = query_indices >= 0
                query_indices = tl.where(query_range_mask, query_indices, 0)

                query_data = tl.load(
                    q + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]),
                )

                max_pos_q = tl.max(query_indices)
                num_kv_iter = tl.cdiv(max_pos_q - pad_offset_kv + 1, BLOCK_N)
                for tile_kv in tl.range(0, num_kv_iter):
                    key_range_mask = ((tile_kv * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
                    key_data = tl.load(
                        k + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                        mask=key_range_mask[:, None],
                        other=0.0,
                    )
                    value_data = tl.load(
                        v + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                        mask=key_range_mask[:, None],
                        other=0.0,
                    )

                    qk = tl.dot(query_data, key_data.T) * qk_scale

                    # causal mask for each query pos
                    causal_mask = query_indices[:, None] >= (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[None, :]
                    qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

                    score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new[:, None])
                    score_sum = score_sum * score_scale + tl.sum(qk, 1)
                    acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                    score_max = score_max_new

                acc /= score_sum[:, None]
                tl.store(
                    out + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]),
                    acc.to(out.dtype.element_ty),
                    mask=query_range_mask[:, None],
                )
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        m_sort, m_sort_indices = torch.sort(route_mask.transpose(1, 2).contiguous(), descending=True, stable=False) # [B, HQ, LQ]
        # offline calculate skipping
        m_sort_pad = torch.nn.functional.pad(m_sort, (0, BLOCK_M - LQ % BLOCK_M), value=0).reshape(B, HQ, -1, BLOCK_M)
        m_sort_pad = m_sort_pad.any(dim=-1)
        m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)

        out = torch.zeros_like(q)
        grid = lambda META: (triton.cdiv(LQ, BLOCK_M), HQ, B)
        attention[grid](
            q, k, v,
            m_sort_pad, m_sort_indices, pad_offset, out,
            LQ, LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for head sparse attention (prefill)\\
        **dense & sort**\\
        Return [batch_size, query_length, num_query_heads, head_dim]
        Args:
            q, k, v: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
            route_mask: Optional[torch.Tensor] with shape [batch_size, seqlen, num_heads], with 1 for active heads
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        
        q = kwargs.get('q')
        k = kwargs.get('k')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        route_mask = kwargs.pop('route_mask', None)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]
        num_stages = 2
        num_warps = 4

        if impl == 'auto': impl = 'sort'

        estimated_sparsity = kwargs.pop('estimated_sparsity', 0)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, LQ, HQ), dtype=torch.bool, device=device)

        BLOCK_M = triton.next_power_of_2(int(LQ * estimated_sparsity))
        if BLOCK_M > int(LQ * estimated_sparsity): BLOCK_M = BLOCK_M >> 1
        BLOCK_M = min(128, max(16, BLOCK_M))
        BLOCK_N = min(32, max(16, triton.next_power_of_2(LQ)))
        
        BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)

        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            pad_offset=pad_offset,
            **kwargs
        )