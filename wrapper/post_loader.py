import os
import regex as re
import torch
import torch.nn as nn

from typing import Dict, Tuple, List, Optional, Any
from transformers import PreTrainedModel
from torch.sparse.semi_structured import SparseSemiStructuredTensorCUTLASS
from einops import rearrange

from config import PruningConfig, RouterConfig, LoRAConfig, ComponentConfig
from wrapper.backbone import PrunedDecoderLayer, PrunedModel, PrunedModelForCausalLM
from ops.router import Router

###############
# load router, lora, mask after backbone loading and initialization
###############

def load_unstructured_mask(
    model: nn.Linear,
    mask: torch.Tensor,
    **kwargs
):
    assert isinstance(model, nn.Linear)
    assert mask.shape == model.weight.shape

    pattern = kwargs.get('pattern', None)
    model.weight *= mask.to(model.weight.dtype)
    
    if ':' in pattern:
        if kwargs.get('checking', True):
            zero_size, group_size = list(map(int, pattern.split(':')))
            sparsity = rearrange(model.weight, 'n (h d) -> (n h) d', d=group_size)
            sparsity = (sparsity == 0).float().mean(dim=-1)
            assert (sparsity == (zero_size / group_size)).all(), f'semi-structured pattern {pattern} mismatch'

        model.weight = nn.Parameter(SparseSemiStructuredTensorCUTLASS.from_dense(model.weight))

def post_load(
    model: PrunedModelForCausalLM,
    **kwargs,
):
    cached_state_dicts = {}
    available_components = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention', 'glu', 'down_proj']

    # 1. check lora (from peft-like ckpt)
    lora_config = model.pruning_config.lora
    if lora_config.path is not None:
        if not os.path.exists(lora_config.path): raise FileNotFoundError(f'lora path {lora_config.path} not found')
        cached_state_dicts[lora_config.path] = torch.load(lora_config.path, map_location='cpu', weights_only=True)

        tmp_state_dict = {}
        for k, v in cached_state_dicts[lora_config.path].items():
            name = re.findall(r'[\w\W]*?(model.layers.[\d]+.[\w\W]+)', k)[-1]
            name = name.replace('.block', '')
            component = re.findall(r'[\w\W]*?(lora_A|lora_B)', name)[-1]
            prefix = name.split('.lora')[0]
            if prefix not in tmp_state_dict:
                tmp_state_dict[prefix] = {}
            tmp_state_dict[prefix][component] = v

        param_dict = {}
        for k, v in model.named_parameters(): param_dict[k] = v
        for k, v in tmp_state_dict.items():
            if k + '.weight' in param_dict:
                    lora_weight = ((v['lora_B'].float() @ v['lora_A'].float()) * (lora_config.alpha / lora_config.rank)).to(v['lora_A'].dtype)
                    param_dict[k + '.weight'].data.add_(lora_weight)

    for layer_idx in range(len(model.model.layers)):
        layer: PrunedDecoderLayer = model.model.layers[layer_idx]
        prefix = f'model.layers.{layer_idx}'
        for layer_config in layer.pruning_configs:
            for component in available_components:
                if component not in set(layer_config): continue

                # 2. check router
                component_config: ComponentConfig = getattr(layer_config, f'{component}')
                router_config: RouterConfig = component_config.router
                router_path = router_config.path

                if router_config.type is not None:
                    if not os.path.exists(router_path): raise FileNotFoundError(f'router path {router_path} not found')
                    if router_path not in cached_state_dicts:
                        cached_state_dicts[router_path] = torch.load(router_path, map_location='cpu', weights_only=True)
                    
                    tgt_router: Router = getattr(model.model.layers[layer_idx], f'router_{component}')
                    router_prefix = router_config.prefix
                    for i in range(len(router_prefix)):
                        router_prefix[i] = f'{prefix}.{router_prefix[i]}'
                    
                    tgt_router.load_router_dict(cached_state_dicts[router_path], router_prefix)
                
                # 3. check mask
                mask_path = component_config.mask
                if mask_path is not None:
                    if not os.path.exists(mask_path): raise FileNotFoundError(f'mask path {mask_path} not found')
                    if mask_path not in cached_state_dicts:
                        cached_state_dicts[mask_path] = torch.load(mask_path, map_location='cpu', weights_only=True)
                    
                    mask_prefix = component_config.mask_prefix
                    if mask_prefix is None: mask_prefix = 'mask'
                    mask_prefix = f'{prefix}.{component}.{mask_prefix}'

                    mask = cached_state_dicts[mask_path][mask_prefix]
                    load_unstructured_mask(
                        model=getattr(model.model.layers[layer_idx], component),
                        mask=mask,
                        pattern=component_config.router.pattern,
                    )


