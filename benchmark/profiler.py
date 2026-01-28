import os
import torch
import torch.nn as nn
import torch.distributed as dist
import tqdm

from functools import partial, cache
from typing import Optional, Tuple, Dict, List, Any, Callable, Sequence
from argparse import Namespace
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from torch.profiler import record_function

# using Triton profiling func
from triton import runtime
from triton.testing import _summarize_statistics

@cache
def is_distributed() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1

class ModelProfiler:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        args: Namespace,
        **kwargs,
    ):
        self.device = args.device
        self.args = args

        self.model = model.to(dtype=kwargs.get('dtype', torch.bfloat16), device=self.device)
        self.tokenizer = tokenizer

        self.profiler = None
        if self.args.torch_profiler:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs', use_gzip=True),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
    
    def _get_dummy_inputs(
        self,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        dummy_inputs = torch.randint(low=0, high=self.tokenizer.vocab_size, size=(batch_size, seq_len), device=self.device)
        return dummy_inputs
    
    def _shuffle_mask(
        self,
        pruning_kwargs: Dict[str, Any],
        estimated_sparsity: float=0.0,
    ):
        for k, v in pruning_kwargs.items():
            if isinstance(v, Dict): self._shuffle_mask(v, estimated_sparsity)
            elif isinstance(v, torch.Tensor):
                pruning_kwargs[k].copy_(torch.rand_like(v.float()) > estimated_sparsity)
    
    def _run_torch_profiler(
        self,
        func: Callable,
    ):
        with self.profiler: func()
    
    def _inplace_copy(
        self,
        old: Sequence[torch.Tensor],
        new: Sequence[torch.Tensor],
    ):
        if isinstance(old, torch.Tensor):
            new.copy_(old)
        elif isinstance(old, Dict):
            for k in old.keys(): self._inplace_copy(old[k], new[k])
        elif isinstance(old, Sequence):
            for o, n in zip(old, new): self._inplace_copy(o, n)
        
    @torch.no_grad()
    def profile_ttft(
        self,
        batch_size: int=1,
        seq_len: int=2048,
        cuda_graph: bool=False,
        **kwargs,
    ):
        """
        Profile the time-to-first-token (TTFT) of the pruned model.
        
        - We first generate dummy inputs [batch_size, seq_len] as the prompts
        - Then run the forward pass repeatedly to measure the TTFT with Triton profiler
        - Each run will clean the L2 cache and past KV cache first
        """
        print(f"[INFO] Profiling TTFT with batch_size={batch_size}, seq_len={seq_len}")
        dummy_inputs = self._get_dummy_inputs(batch_size, seq_len)
        model_inputs_kwargs = dict(
            input_ids=dummy_inputs,
            attention_mask=torch.ones_like(dummy_inputs, dtype=torch.bool),
            use_cache=True,
            past_key_values=None,
            estimated_sparsity=kwargs.get('sparsity', 0.0),
            use_cuda_graph=cuda_graph,
        )
        
        device_interface = runtime.driver.active.get_device_interface()
        device_cache = runtime.driver.active.get_empty_cache_for_benchmark()

        # 1. warmup
        print("[INFO] Warmup the model...")
        for _ in range(kwargs.get('warmup', 1)):
            dummy_pruning_kwargs = self.model.generate_pruning_kwargs(**model_inputs_kwargs)
            model_inputs_kwargs['pruning_kwargs'] = dummy_pruning_kwargs
            self.model(**model_inputs_kwargs)
        device_interface.synchronize()
    
        # 2. run profiler
        n_repeat = kwargs.get('repeat', 20)

        ########## without cuda graph ##########
        if not cuda_graph:
            start_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]
            end_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]

            if self.profiler is not None: self.profiler.start()
            device_interface.synchronize()
            for i in tqdm.trange(n_repeat, desc="Profiling TTFT..."):
                self._shuffle_mask(model_inputs_kwargs['pruning_kwargs'], kwargs.get('sparsity', 0.0))

                runtime.driver.active.clear_cache(device_cache)
                device_interface.synchronize()
                start_events[i].record()
                with record_function("**Model Prefill**"):
                    self.model(**model_inputs_kwargs)
                if self.profiler is not None: self.profiler.step()
                end_events[i].record()
                device_interface.synchronize()
            
            device_interface.synchronize()
            if self.profiler is not None: self.profiler.stop()

            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            return _summarize_statistics(
                times=times,
                quantiles=kwargs.get('quantiles', [0.5, 0.2, 0.8]),
                return_mode=kwargs.get('return_mode', 'mean'),
            )
        else:
            with torch.cuda.stream(torch.cuda.Stream()):
                start_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]
                end_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]

                # capture cuda graph
                dummy_pruning_kwargs = self.model.generate_pruning_kwargs(**model_inputs_kwargs)
                model_inputs_kwargs['pruning_kwargs'] = dummy_pruning_kwargs
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.model(**model_inputs_kwargs)
                
                if self.profiler is not None: self.profiler.start()
                device_interface.synchronize()
                runtime.driver.active.clear_cache(device_cache)

                for i in tqdm.trange(n_repeat, desc="Profiling TTFT with CUDA Graph..."):
                    self._shuffle_mask(model_inputs_kwargs['pruning_kwargs'], kwargs.get('sparsity', 0.0))

                    new_dummy_inputs = self._get_dummy_inputs(batch_size, seq_len)
                    self._inplace_copy(model_inputs_kwargs['input_ids'], new_dummy_inputs)

                    device_interface.synchronize()
                    start_events[i].record()
                    with record_function("**Model Prefill**"):
                        g.replay()
                    if self.profiler is not None: self.profiler.step()
                    end_events[i].record()
                    device_interface.synchronize()
                
                device_interface.synchronize()
                if self.profiler is not None: self.profiler.stop()

                times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
                return _summarize_statistics(
                    times=times,
                    quantiles=kwargs.get('quantiles', [0.5, 0.2, 0.8]),
                    return_mode=kwargs.get('return_mode', 'mean'),
                )

    @torch.no_grad()
    def profile_tpot(
        self,
        batch_size: int=1,
        seq_len: int=2048,
        cuda_graph: bool=True,
        **kwargs,
    ):
        """
        Profile the time-per-output-token (TPOT) of the pruned model.
        
        - We first generate dummy inputs [batch_size, seq_len] as the KV cache
        - Then run the forward pass repeatedly with dummy inputs [batch_size, 1] to measure the TPOT with/without CUDA Graph
        - Each run will clean the L2 cache and reset KV cache first
        """

        print(f"[INFO] Profiling TPOT with batch_size={batch_size}, kv cache seq_len={seq_len}")
        dummy_inputs = self._get_dummy_inputs(batch_size, 1)
        dummy_kv_cache = self._get_dummy_inputs(batch_size, seq_len)
        model_inputs_kwargs = dict(
            input_ids=dummy_kv_cache,
            attention_mask=torch.ones_like(dummy_kv_cache, dtype=torch.bool),
            use_cache=True,
            estimated_sparsity=kwargs.get('sparsity', 0.0),
        )
        device_interface = runtime.driver.active.get_device_interface()
        device_cache = runtime.driver.active.get_empty_cache_for_benchmark()

        # get dummy kv cache
        model_outputs: CausalLMOutputWithPast = self.model(**model_inputs_kwargs)
        model_inputs_kwargs['past_key_values'] = model_outputs.past_key_values
        model_inputs_kwargs['input_ids'] = dummy_inputs

        # 1. warmup
        print("[INFO] Warmup the model...")
        for _ in range(kwargs.get('warmup', 1)):
            dummy_pruning_kwargs = self.model.generate_pruning_kwargs(**model_inputs_kwargs)
            model_inputs_kwargs['pruning_kwargs'] = dummy_pruning_kwargs
            self.model(**model_inputs_kwargs)
            model_inputs_kwargs['past_key_values'].fallback_cache(seq_len)

        device_interface.synchronize()

        # 2. run profiler
        n_repeat = kwargs.get('repeat', 20)

        ########## without cuda graph ##########
        if not cuda_graph:
            start_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]
            end_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]

            if self.profiler is not None: self.profiler.start()
            device_interface.synchronize()
            for i in tqdm.trange(n_repeat, desc="Profiling TPOT..."):
                self._shuffle_mask(model_inputs_kwargs['pruning_kwargs'], kwargs.get('sparsity', 0.0))

                runtime.driver.active.clear_cache(device_cache)
                device_interface.synchronize()
                start_events[i].record()
                with record_function("**Model Decode**"):
                    self.model(**model_inputs_kwargs)
                if self.profiler is not None: self.profiler.step()
                end_events[i].record()
                device_interface.synchronize()
                model_inputs_kwargs['past_key_values'].fallback_cache(seq_len)
            
            device_interface.synchronize()
            if self.profiler is not None: self.profiler.stop()

            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            return _summarize_statistics(
                times=times,
                quantiles=kwargs.get('quantiles', [0.5, 0.2, 0.8]),
                return_mode=kwargs.get('return_mode', 'mean'),
            )
        
        ########## with cuda graph ##########
        else:
            with torch.cuda.stream(torch.cuda.Stream()):
                start_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]
                end_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]

                # capture cuda graph
                dummy_pruning_kwargs = self.model.generate_pruning_kwargs(**model_inputs_kwargs)
                model_inputs_kwargs['pruning_kwargs'] = dummy_pruning_kwargs
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.model(**model_inputs_kwargs)
                
                if self.profiler is not None: self.profiler.start()
                device_interface.synchronize()
                runtime.driver.active.clear_cache(device_cache)

                for i in tqdm.trange(n_repeat, desc="Profiling TPOT with CUDA Graph..."):
                    self._shuffle_mask(model_inputs_kwargs['pruning_kwargs'], kwargs.get('sparsity', 0.0))

                    new_dummy_inputs = self._get_dummy_inputs(batch_size, 1)
                    self._inplace_copy(model_inputs_kwargs['input_ids'], new_dummy_inputs)

                    device_interface.synchronize()
                    start_events[i].record()
                    with record_function("**Model Decode**"):
                        g.replay()
                    if self.profiler is not None: self.profiler.step()
                    end_events[i].record()
                    device_interface.synchronize()
                    model_inputs_kwargs['past_key_values'].fallback_cache(seq_len)
                
                device_interface.synchronize()
                if self.profiler is not None: self.profiler.stop()

                times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
                return _summarize_statistics(
                    times=times,
                    quantiles=kwargs.get('quantiles', [0.5, 0.2, 0.8]),
                    return_mode=kwargs.get('return_mode', 'mean'),
                )