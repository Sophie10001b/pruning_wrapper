import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, List, Any

class QuerySparseDecode:

    support_kernel = [
        'split',
        'fuse',
    ]

    @classmethod
    def _fuse_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=16,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor,
            pad_offset: tl.tensor,
            out: tl.tensor,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr, # next pow of 2 for query heads in each group (>= 16)
            BLOCK_N: tl.constexpr,
        ):
            batch_id, key_head_id = tl.program_id(1), tl.program_id(0)
            qk_scale *= 1.44269504 # 1/log(2)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            is_active = tl.load(route_mask + batch_id)

            if is_active == 1:
                start_q, end_q = batch_id, batch_id + 1
                seq_offset_q = start_q

                pad_offset_k = tl.load(pad_offset + batch_id)
                start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
                key_length = end_k - start_k

                query_range_mask = tl.arange(0, BLOCK_M) < G
                query_data = tl.load(
                    q + ((seq_offset_q * HQ * D) + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=query_range_mask[:, None],
                    other=0.0
                )

                split_n_range = tl.cdiv(key_length, BLOCK_N)
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

                    # qk size [BLOCK_M, BLOCK_N]
                    qk = tl.dot(query_data, key_data.T) * qk_scale
                    qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))

                    # scalar
                    score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new[:, None])
                    score_sum = score_sum * score_scale + tl.sum(qk, 1)
                    acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                    score_max = score_max_new
                
                acc /= score_sum[:, None]
                tl.store(
                    out + ((seq_offset_q * HQ * D) + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :],
                    acc.to(out.dtype.element_ty),
                    mask=query_range_mask[:, None],
                )
        
        B, LQ, HQ, D = q.shape
        assert k.dim() == 4 and q.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.zeros_like(q)
        grid = (HK, B)
        attention[grid](
            q, k, v,
            route_mask, pad_offset, out,
            LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
        
    @classmethod
    def _split_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=16,
        BLOCK_N: Optional[int]=32,
        split_k_size: Optional[int]=-1,
        num_split: Optional[int]=-1,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'LBK', 'NBK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def split(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor,
            pad_offset: tl.tensor,
            out: tl.tensor, # [B, 1, HQ, splitk, D]
            metadata: tl.tensor, # [B, 1, HQ, 2, splitk]
            LK: tl.int64,
            LBK: tl.int64, # size of each split k block
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            NBK: tl.int64, # num of split k for each seq
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            split_id, key_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            qk_scale *= 1.44269504 # 1/log(2)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            is_active = tl.load(route_mask + batch_id)
            if is_active == 1:
                start_q, end_q = batch_id, batch_id + 1
                seq_offset_q = start_q

                pad_offset_k = tl.load(pad_offset + batch_id)
                start_k = batch_id * LK + split_id * LBK + pad_offset_k
                end_k = tl.minimum(start_k + LBK, (batch_id + 1) * LK)
                key_length = end_k - start_k

                query_range_mask = tl.arange(0, BLOCK_M) < G
                query_data = tl.load(
                    q + ((seq_offset_q * HQ * D) + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=query_range_mask[:, None],
                    other=0.0
                )

                split_n_range = tl.cdiv(key_length, BLOCK_N)
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

                    # qk size [BLOCK_M, BLOCK_N]
                    qk = tl.dot(query_data, key_data.T) * qk_scale
                    qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))

                    # scalar
                    score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new[:, None])
                    score_sum = score_sum * score_scale + tl.sum(qk, 1)
                    acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                    score_max = score_max_new
            
                tl.store(
                    out + (seq_offset_q * HQ * NBK * D + (key_head_id * G + tl.arange(0, BLOCK_M))[:, None] * NBK * D + split_id * D) + tl.arange(0, D)[None, :],
                    acc.to(out.dtype.element_ty),
                    mask=query_range_mask[:, None],
                )
                tl.store(
                    metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + split_id,
                    score_max.to(metadata.dtype.element_ty),
                    mask=query_range_mask,
                )
                tl.store(
                    metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + NBK + split_id,
                    score_sum.to(metadata.dtype.element_ty),
                    mask=query_range_mask,
                )
        
        @triton.jit(do_not_specialize=['NBK',])
        def merge(
            out_tmp: tl.tensor, # [B, 1, HQ, splitk, D]
            metadata: tl.tensor, # [B, 1, HQ, 2, splitk] for local max and local sum
            route_mask: tl.tensor,
            out: tl.tensor, 
            HQ: tl.constexpr,
            D: tl.constexpr,
            NBK: tl.int64, # num of split k for each seq
            NBK2: tl.constexpr, # next_pow_of_2(NBK)
        ):
            query_head_id, batch_id = tl.program_id(0), tl.program_id(1)

            NBK_mask = tl.arange(0, NBK2) < NBK
            is_active = tl.load(route_mask + batch_id)
            if is_active == 1:
                acc = tl.load(
                    out_tmp + (batch_id * HQ + query_head_id) * NBK * D + (tl.arange(0, NBK2) * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=NBK_mask[:, None],
                    other=0.0
                )
                score_max = tl.load(
                    metadata + (batch_id * HQ + query_head_id) * 2 * NBK + tl.arange(0, NBK2),
                    mask=NBK_mask,
                    other=-float('inf')
                )
                score_sum = tl.load(
                    metadata + (batch_id * HQ + query_head_id) * 2 * NBK + NBK + tl.arange(0, NBK2),
                    mask=NBK_mask,
                    other=0.0
                )

                global_max = tl.max(score_max)
                global_scale = tl.exp2(score_max - global_max)
                score_sum *= global_scale
                global_sum = tl.sum(score_sum)

                acc = tl.sum(acc * global_scale[:, None], axis=0)
                acc /= global_sum

                tl.store(
                    out + (batch_id * HQ + query_head_id) * D + tl.arange(0, D),
                    acc.to(out.dtype.element_ty)
                )
        
        assert q.shape[1] == 1 and q.shape[:2] == route_mask.shape

        B, LQ, HQ, D = q.shape
        assert k.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.zeros_like(q)
        out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
        metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)
        grid = lambda META: (num_split, HK, B)

        split[grid](
            q, k, v,
            route_mask, pad_offset, out_local, metadata,
            LK, split_k_size, HQ, HK, D, G, num_split, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        grid = lambda META: (HQ, B)
        merge[grid](
            out_local, metadata,
            route_mask, out,
            HQ, D, num_split, triton.next_power_of_2(num_split),
            num_stages=2,
            num_warps=4,
        )
        return out

    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for query sparse attention (decode)\\
        **fuse:**\\
        Use one-pass decoding kernel with flatGEMM based on GQA/MQA\\
        **split:**\\
        Use flash-decoding style two-stage decoding kernel with flatGEMM based on GQA/MQA
        Args:
            q: torch.Tensor with shape [batch_size, 1, num_heads, head_dim]
            k, v: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
            route_mask: Optional[torch.Tensor] with shape [batch_size, 1], with 1 for active tokens
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]

        if impl == 'auto': impl = 'split'

        q = kwargs.get('q')
        k = kwargs.get('k')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK
        route_mask = kwargs.pop('route_mask', None)
        if route_mask is None: route_mask = torch.ones((B, LQ), dtype=torch.int32, device=device)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)

        BLOCK_M = max(16, triton.next_power_of_2(G))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LK)))
        num_stages = 3
        num_warps = 4

        num_split = 1
        while num_split * B * HK < num_sm * 2 and int(LK / num_split) > max(BLOCK_N, 64):
            num_split += 1
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        split_k_size = triton.cdiv(LK, num_split)
        if num_split > 1:
            return getattr(cls, f"_{impl}_kernel")(
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                split_k_size=split_k_size,
                num_split=num_split,
                num_stages=num_stages,
                num_warps=num_warps,
                route_mask=route_mask,
                pad_offset=pad_offset,
                **kwargs
            )
        else:
            return cls._fuse_kernel(
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_stages=num_stages,
                num_warps=num_warps,
                route_mask=route_mask,
                pad_offset=pad_offset,
                **kwargs
            )

class GroupSparseDecode:

    support_kernel = [
        'split',
        'fuse',
    ]

    @classmethod
    def _fuse_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=16,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor,
            pad_offset: tl.tensor,
            out: tl.tensor,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr, # next pow of 2 for query heads in each group (>= 16)
            BLOCK_N: tl.constexpr,
        ):
            batch_id, key_head_id = tl.program_id(1), tl.program_id(0)
            qk_scale *= 1.44269504 # 1/log(2)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            is_active = tl.load(route_mask + batch_id * HK + key_head_id)

            if is_active == 1:
                start_q, end_q = batch_id, batch_id + 1
                seq_offset_q = start_q

                pad_offset_k = tl.load(pad_offset + batch_id)
                start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
                key_length = end_k - start_k

                query_range_mask = tl.arange(0, BLOCK_M) < G
                query_data = tl.load(
                    q + ((seq_offset_q * HQ * D) + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=query_range_mask[:, None],
                    other=0.0
                )

                split_n_range = tl.cdiv(key_length, BLOCK_N)
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

                    # qk size [BLOCK_M, BLOCK_N]
                    qk = tl.dot(query_data, key_data.T) * qk_scale
                    qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))

                    # scalar
                    score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new[:, None])
                    score_sum = score_sum * score_scale + tl.sum(qk, 1)
                    acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                    score_max = score_max_new
                
                acc /= score_sum[:, None]
                tl.store(
                    out + ((seq_offset_q * HQ * D) + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :],
                    acc.to(out.dtype.element_ty),
                    mask=query_range_mask[:, None],
                )
        
        B, LQ, HQ, D = q.shape
        assert k.dim() == 4 and q.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.zeros_like(q)
        grid = (HK, B)
        attention[grid](
            q, k, v,
            route_mask, pad_offset, out,
            LK, HQ, HK, D, G, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
        
    @classmethod
    def _split_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=16,
        BLOCK_N: Optional[int]=32,
        split_k_size: Optional[int]=-1,
        num_split: Optional[int]=-1,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'LBK', 'NBK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def split(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor,
            pad_offset: tl.tensor,
            out: tl.tensor, # [B, 1, HQ, splitk, D]
            metadata: tl.tensor, # [B, 1, HQ, 2, splitk]
            LK: tl.int64,
            LBK: tl.int64, # size of each split k block
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            NBK: tl.int64, # num of split k for each seq
            qk_scale: tl.float32,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            split_id, key_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            qk_scale *= 1.44269504 # 1/log(2)

            score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
            score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

            is_active = tl.load(route_mask + batch_id * HK + key_head_id)
            if is_active == 1:
                start_q, end_q = batch_id, batch_id + 1
                seq_offset_q = start_q

                pad_offset_k = tl.load(pad_offset + batch_id)
                start_k = batch_id * LK + split_id * LBK + pad_offset_k
                end_k = tl.minimum(start_k + LBK, (batch_id + 1) * LK)
                key_length = end_k - start_k

                query_range_mask = tl.arange(0, BLOCK_M) < G
                query_data = tl.load(
                    q + ((seq_offset_q * HQ * D) + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=query_range_mask[:, None],
                    other=0.0
                )

                split_n_range = tl.cdiv(key_length, BLOCK_N)
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

                    # qk size [BLOCK_M, BLOCK_N]
                    qk = tl.dot(query_data, key_data.T) * qk_scale
                    qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))

                    # scalar
                    score_max_new = tl.maximum(score_max, tl.max(qk, 1))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new[:, None])
                    score_sum = score_sum * score_scale + tl.sum(qk, 1)
                    acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
                    score_max = score_max_new
            
                tl.store(
                    out + (seq_offset_q * HQ * NBK * D + (key_head_id * G + tl.arange(0, BLOCK_M))[:, None] * NBK * D + split_id * D) + tl.arange(0, D)[None, :],
                    acc.to(out.dtype.element_ty),
                    mask=query_range_mask[:, None],
                )
                tl.store(
                    metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + split_id,
                    score_max.to(metadata.dtype.element_ty),
                    mask=query_range_mask,
                )
                tl.store(
                    metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + NBK + split_id,
                    score_sum.to(metadata.dtype.element_ty),
                    mask=query_range_mask,
                )
        
        @triton.jit(do_not_specialize=['NBK',])
        def merge(
            out_tmp: tl.tensor, # [B, 1, HQ, splitk, D]
            metadata: tl.tensor, # [B, 1, HQ, 2, splitk] for local max and local sum
            route_mask: tl.tensor,
            out: tl.tensor,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            NBK: tl.int64, # num of split k for each seq
            NBK2: tl.constexpr, # next_pow_of_2(NBK)
        ):
            query_head_id, batch_id = tl.program_id(0), tl.program_id(1)
            key_head_id = query_head_id // G

            NBK_mask = tl.arange(0, NBK2) < NBK
            is_active = tl.load(route_mask + batch_id * HK + key_head_id)
            if is_active == 1:
                acc = tl.load(
                    out_tmp + (batch_id * HQ + query_head_id) * NBK * D + (tl.arange(0, NBK2) * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=NBK_mask[:, None],
                    other=0.0
                )
                score_max = tl.load(
                    metadata + (batch_id * HQ + query_head_id) * 2 * NBK + tl.arange(0, NBK2),
                    mask=NBK_mask,
                    other=-float('inf')
                )
                score_sum = tl.load(
                    metadata + (batch_id * HQ + query_head_id) * 2 * NBK + NBK + tl.arange(0, NBK2),
                    mask=NBK_mask,
                    other=0.0
                )

                global_max = tl.max(score_max)
                global_scale = tl.exp2(score_max - global_max)
                score_sum *= global_scale
                global_sum = tl.sum(score_sum)

                acc = tl.sum(acc * global_scale[:, None], axis=0)
                acc /= global_sum

                tl.store(
                    out + (batch_id * HQ + query_head_id) * D + tl.arange(0, D),
                    acc.to(out.dtype.element_ty)
                )
        
        assert q.shape[1] == 1 and q.shape[:2] == route_mask.shape[:2]

        B, LQ, HQ, D = q.shape
        assert k.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.zeros_like(q)
        out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
        metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)
        grid = lambda META: (num_split, HK, B)

        split[grid](
            q, k, v,
            route_mask, pad_offset, out_local, metadata,
            LK, split_k_size, HQ, HK, D, G, num_split, D**-0.5,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        grid = lambda META: (HQ, B)
        merge[grid](
            out_local, metadata,
            route_mask, out,
            HQ, HK, D, G, num_split, triton.next_power_of_2(num_split),
            num_stages=2,
            num_warps=4,
        )
        return out

    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for group sparse attention (decode)\\
        **fuse:**\\
        Use one-pass decoding kernel with flatGEMM based on GQA/MQA\\
        **split:**\\
        Use flash-decoding style two-stage decoding kernel with flatGEMM based on GQA/MQA
        Args:
            q: torch.Tensor with shape [batch_size, 1, num_query_heads, head_dim]
            k: torch.Tensor with shape [batch_size, key_length, num_key_heads, head_dim]
            v: torch.Tensor with shape [batch_size, key_length, num_key_heads, head_dim]
            route_mask: Optional[torch.Tensor] with shape [batch_size, 1, num_key_heads], with 1 for active tokens
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """

        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]

        if impl == 'auto': impl = 'split'

        q = kwargs.get('q')
        k = kwargs.get('k')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK
        route_mask = kwargs.pop('route_mask', None)
        if route_mask is None: route_mask = torch.ones((B, LQ, HK), dtype=torch.int32, device=device)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)

        BLOCK_M = max(16, triton.next_power_of_2(G))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LK)))
        num_stages = 3
        num_warps = 4

        num_split = 1
        while num_split * B * HK < num_sm * 2 and int(LK / num_split) > max(BLOCK_N, 64):
            num_split += 1
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        split_k_size = triton.cdiv(LK, num_split)
        if num_split > 1:
            return getattr(cls, f"_{impl}_kernel")(
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                split_k_size=split_k_size,
                num_split=num_split,
                num_stages=num_stages,
                num_warps=num_warps,
                route_mask=route_mask,
                pad_offset=pad_offset,
                **kwargs
            )
        else:
            return cls._fuse_kernel(
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_stages=num_stages,
                num_warps=num_warps,
                route_mask=route_mask,
                pad_offset=pad_offset,
                **kwargs
            )


class HeadSparseDecode:

    support_kernel = [
        'split',
        'fuse',
    ]

    @classmethod
    def _fuse_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def attention(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor,
            pad_offset: tl.tensor,
            out: tl.tensor,
            LK: tl.int64,
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            qk_scale: tl.float32,
            BLOCK_N: tl.constexpr,
        ):
            batch_id, query_head_id = tl.program_id(1), tl.program_id(0)
            qk_scale *= 1.44269504 # 1/log(2)

            score_max = -float('inf')
            score_sum = float(0)
            acc = tl.zeros([D], dtype=tl.float32)
            key_head_id = query_head_id // G

            is_active = tl.load(route_mask + batch_id * HQ + query_head_id)
            if is_active == 1:
                start_q, end_q = batch_id, batch_id + 1
                seq_offset_q = start_q

                pad_offset_k = tl.load(pad_offset + batch_id)
                start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
                key_length = end_k - start_k

                query_data = tl.load(q + (seq_offset_q * HQ * D + query_head_id * D) + tl.arange(0, D))[None, :]

                split_n_range = tl.cdiv(key_length, BLOCK_N)
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

                    # qk size [BLOCK_N]
                    qk = tl.sum(query_data * key_data, axis=-1, dtype=tl.float32) * qk_scale
                    qk = tl.where(key_range_mask, qk, -float('inf'))

                    # scalar
                    score_max_new = tl.maximum(score_max, tl.max(qk))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new)
                    score_sum = score_sum * score_scale + tl.sum(qk)
                    acc = acc * score_scale + tl.sum(qk.to(q.dtype.element_ty) * value_data.T, axis=-1, dtype=tl.float32)
                    score_max = score_max_new
                
                acc /= score_sum
                tl.store(
                    out + (seq_offset_q * HQ * D + query_head_id * D) + tl.arange(0, D),
                    acc.to(out.dtype.element_ty)
                )
        
        B, LQ, HQ, D = q.shape
        assert k.dim() == 4 and q.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.zeros_like(q)
        grid = (HQ, B)
        attention[grid](
            q, k, v,
            route_mask, pad_offset, out,
            LK, HQ, HK, D, G, D**-0.5,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
        
    @classmethod
    def _split_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        BLOCK_N: Optional[int]=32,
        split_k_size: Optional[int]=-1,
        num_split: Optional[int]=-1,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'LBK', 'NBK', 'qk_scale'],
        **kwargs
    ):
        @triton.jit(do_not_specialize=do_not_specialize)
        def split(
            q: tl.tensor,
            k: tl.tensor,
            v: tl.tensor,
            route_mask: tl.tensor,
            pad_offset: tl.tensor,
            out: tl.tensor, # [B, 1, HQ, splitk, D]
            metadata: tl.tensor, # [B, 1, HQ, 2, splitk]
            LK: tl.int64,
            LBK: tl.int64, # size of each split k block
            HQ: tl.constexpr,
            HK: tl.constexpr,
            D: tl.constexpr,
            G: tl.constexpr,
            NBK: tl.int64, # num of split k for each seq
            qk_scale: tl.float32,
            BLOCK_N: tl.constexpr,
        ):
            split_id, query_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
            qk_scale *= 1.44269504 # 1/log(2)

            score_max = -float('inf')
            score_sum = float(0)
            acc = tl.zeros([D], dtype=tl.float32)
            key_head_id = query_head_id // G

            is_active = tl.load(route_mask + batch_id * HQ + query_head_id)
            if is_active == 1:
                start_q, end_q = batch_id, batch_id + 1
                seq_offset_q = start_q

                pad_offset_k = tl.load(pad_offset + batch_id)
                start_k = batch_id * LK + split_id * LBK + pad_offset_k
                end_k = tl.minimum(start_k + LBK, (batch_id + 1) * LK)
                key_length = end_k - start_k

                query_data = tl.load(q + (seq_offset_q * HQ * D + query_head_id * D) + tl.arange(0, D))[None, :]

                split_n_range = tl.cdiv(key_length, BLOCK_N)
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

                    # qk size [BLOCK_N]
                    qk = tl.sum(query_data * key_data, axis=-1, dtype=tl.float32) * qk_scale
                    qk = tl.where(key_range_mask, qk, -float('inf'))

                    # scalar
                    score_max_new = tl.maximum(score_max, tl.max(qk))
                    score_scale = tl.exp2(score_max - score_max_new)
                    qk = tl.exp2(qk - score_max_new)
                    score_sum = score_sum * score_scale + tl.sum(qk)
                    acc = acc * score_scale + tl.sum(qk.to(q.dtype.element_ty) * value_data.T, axis=-1, dtype=tl.float32)
                    score_max = score_max_new
            
                tl.store(
                    out + (seq_offset_q * HQ * NBK * D + query_head_id * NBK * D + split_id * D) + tl.arange(0, D),
                    acc.to(out.dtype.element_ty),
                )
                tl.store(
                    metadata + seq_offset_q * HQ * 2 * NBK + query_head_id * 2 * NBK + split_id,
                    score_max.to(metadata.dtype.element_ty),
                )
                tl.store(
                    metadata + seq_offset_q * HQ * 2 * NBK + query_head_id * 2 * NBK + NBK + split_id,
                    score_sum.to(metadata.dtype.element_ty),
                )
        
        @triton.jit(do_not_specialize=['NBK',])
        def merge(
            out_tmp: tl.tensor, # [B, 1, HQ, splitk, D]
            metadata: tl.tensor, # [B, 1, HQ, 2, splitk] for local max and local sum
            route_mask: tl.tensor,
            out: tl.tensor, 
            HQ: tl.constexpr,
            D: tl.constexpr,
            NBK: tl.int64, # num of split k for each seq
            NBK2: tl.constexpr, # next_pow_of_2(NBK)
        ):
            query_head_id, batch_id = tl.program_id(0), tl.program_id(1)

            NBK_mask = tl.arange(0, NBK2) < NBK
            is_active = tl.load(route_mask + batch_id * HQ + query_head_id)
            if is_active == 1:
                acc = tl.load(
                    out_tmp + (batch_id * HQ + query_head_id) * NBK * D + (tl.arange(0, NBK2) * D)[:, None] + tl.arange(0, D)[None, :],
                    mask=NBK_mask[:, None],
                    other=0.0
                )
                score_max = tl.load(
                    metadata + (batch_id * HQ + query_head_id) * 2 * NBK + tl.arange(0, NBK2),
                    mask=NBK_mask,
                    other=-float('inf')
                )
                score_sum = tl.load(
                    metadata + (batch_id * HQ + query_head_id) * 2 * NBK + NBK + tl.arange(0, NBK2),
                    mask=NBK_mask,
                    other=0.0
                )

                global_max = tl.max(score_max)
                global_scale = tl.exp2(score_max - global_max)
                score_sum *= global_scale
                global_sum = tl.sum(score_sum)

                acc = tl.sum(acc * global_scale[:, None], axis=0)
                acc /= global_sum

                tl.store(
                    out + (batch_id * HQ + query_head_id) * D + tl.arange(0, D),
                    acc.to(out.dtype.element_ty)
                )
        
        assert q.shape[1] == 1 and q.shape[:3] == route_mask.shape

        B, LQ, HQ, D = q.shape
        assert k.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.zeros_like(q)
        out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
        metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)
        grid = lambda META: (num_split, HQ, B)

        split[grid](
            q, k, v,
            route_mask, pad_offset, out_local, metadata,
            LK, split_k_size, HQ, HK, D, G, num_split, D**-0.5,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        grid = lambda META: (HQ, B)
        merge[grid](
            out_local, metadata,
            route_mask, out,
            HQ, D, num_split, triton.next_power_of_2(num_split),
            num_stages=2,
            num_warps=4,
        )
        return out

    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for head sparse attention (decode)\\
        **fuse:**\\
        Use one-pass decoding kernel with GEMV for each head\\
        **split:**\\
        Use flash-decoding style two-stage decoding kernel with GEMV for each head
        Args:
            q: torch.Tensor with shape [batch_size, 1, num_heads, head_dim]
            k, v: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
            route_mask: Optional[torch.Tensor] with shape [batch_size, 1], with 1 for active tokens
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]

        if impl == 'auto': impl = 'split'

        q = kwargs.get('q')
        k = kwargs.get('k')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK
        route_mask = kwargs.pop('route_mask', None)
        if route_mask is None: route_mask = torch.ones((B, LQ), dtype=torch.int32, device=device)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)

        BLOCK_N = min(128, max(16, triton.next_power_of_2(LK)))
        num_stages = 3
        num_warps = 4

        num_split = 1
        while num_split * B * HQ < num_sm * 2 and int(LK / num_split) >= max(BLOCK_N, 64):
            num_split += 1
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        split_k_size = triton.cdiv(LK, num_split)
        if num_split > 1:
            return getattr(cls, f"_{impl}_kernel")(
                BLOCK_N=BLOCK_N,
                split_k_size=split_k_size,
                num_split=num_split,
                num_stages=num_stages,
                num_warps=num_warps,
                route_mask=route_mask,
                pad_offset=pad_offset,
                **kwargs
            )
        else:
            return cls._fuse_kernel(
                BLOCK_N=BLOCK_N,
                num_stages=num_stages,
                num_warps=num_warps,
                route_mask=route_mask,
                pad_offset=pad_offset,
                **kwargs
            )