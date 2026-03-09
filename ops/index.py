import random
import torch
import torch.nn as nn

from einops import rearrange
from typing import Optional, Dict, Sequence

###########################
#   Index for Structured
###########################

def get_module_recursive(start_module: nn.Module, target: str) -> nn.Module:
    module = start_module
    for part in target.split('.'):
        if len(part) == 0: continue
        module = getattr(module, part)
    return module

class BaseIndex:
    __propagate_targets__ = [
        'attention.q_proj',
        'attention.o_proj',
        'ffn.up_proj',
        'ffn.down_proj'
    ]

    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        dim: Optional[int]=0,
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        pass
    
    @classmethod
    def monkey_patch_layer(
        cls,
        root_module: nn.Linear,
        sparsity: Optional[float]=0.5,
        **kwargs,
    ):
        pass

    @classmethod
    def monkey_patch_sublayer(
        cls,
        root_module: nn.Linear,
        target: str,
        sparsity: Optional[float]=0.5,
        ignore_idx: Optional[Sequence[int]]=None,
        **kwargs,
    ):
        pass

    @classmethod
    def check_if_pruneable(
        cls,
        root_module: nn.Module,
        module_name: Optional[str]='',
        **kwargs,
    ):
        tgt_module = get_module_recursive(root_module, module_name)
        pruneable = (not hasattr(tgt_module, '_applied_pruning')) or all([len(applied_pruning) == 0 for applied_pruning in tgt_module._applied_pruning])
        if hasattr(tgt_module, '_leaf_modules'):
            for leaf_module in tgt_module._leaf_modules:
                pruneable &= cls.check_if_pruneable(tgt_module, leaf_module, **kwargs)
        else:
            return True
    
    @classmethod
    def mark_pruned(
        cls,
        root_module: nn.Module,
        pruning_type: str,
        module_name: Optional[str]='',
        **kwargs,
    ):
        tgt_module = get_module_recursive(root_module, module_name)
        if not hasattr(tgt_module, '_applied_pruning'): tgt_module._applied_pruning = []
        tgt_module._applied_pruning.append(pruning_type)

        if hasattr(tgt_module, '_leaf_modules'):
            for leaf_module in tgt_module._leaf_modules:
                cls.mark_pruned(tgt_module, pruning_type, leaf_module, **kwargs)

class StructuredIndex(BaseIndex):
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        dim: Optional[int]=0,
        k: Optional[int]=-1,
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        k = int(shape[dim] * sparsity) if (k < 1 or k > shape[dim]) else k
        indices = torch.randint(0, shape[dim], size=(k,), device=device)
        return indices
    
    @classmethod
    def monkey_patch_layer(
        cls,
        root_module: nn.Module,
        sparsity: Optional[float]=0.5,
        indices: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        module = root_module.model.layers
        device = module[0].self_attn.q_proj.weight.device

        if indices is None:
            candidate_indices = [i for i in range(len(module)) if cls.check_if_pruneable(module[i])]
            indices = random.sample(candidate_indices, k=int(len(module) * sparsity))
        
        for layer_id in indices:
            module[layer_id].forward = lambda x: x[0]
            cls.mark_pruned(module[layer_id], 'M')
        
        if not isinstance(indices, torch.Tensor): indices = torch.tensor(indices, device=device)
        return indices
    
    @classmethod
    def monkey_patch_sublayer(
        cls,
        root_module: nn.Module,
        target: str,
        sparsity: Optional[float]=0.5,
        indices: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        module = root_module.model.layers
        device = module[0].self_attn.q_proj.weight.device

        if indices is None:
            candidate_indices = [i for i in range(len(module)) if cls.check_if_pruneable(module[i], target)]
            indices = random.sample(candidate_indices, k=int(len(module) * sparsity))
        
        for layer_id in indices:
            setattr(module[layer_id], target, lambda x: x[0])
            cls.mark_pruned(module[layer_id], 'M', target)
        
        if not isinstance(indices, torch.Tensor): indices = torch.tensor(indices, device=device)
        return indices
    
    @classmethod
    def monkey_patch_nk(
        cls,
        root_module: nn.Module,
        src_targets: Sequence[str],
        dst_targets: Sequence[str],
        sparsity: Optional[float]=0.5,
        indices: Optional[torch.Tensor]=None,
        num_query_heads: Optional[int]=0,
        num_key_value_heads: Optional[int]=0,
        **kwargs,
    ):
        device = root_module.model.layers[0].self_attn.q_proj.weight.device

        if len(src_targets) > 0:
            start_module = get_module_recursive(root_module, src_targets[0])
            prune_dim = 1 if start_module == 'model.embed_tokens' else 0
        else:
            start_module = get_module_recursive(root_module, dst_targets[0])
            prune_dim = 1

        start_module = get_module_recursive(root_module, src_targets[0] if len(src_targets) > 0 else dst_targets[0])
        is_src_prunable = all([cls.check_if_pruneable(root_module, _) for _ in src_targets]) if len(src_targets) > 0 else True
        is_tgt_prunable = all([cls.check_if_pruneable(root_module, _) for _ in dst_targets]) if len(dst_targets) > 0 else True

        if not (is_src_prunable and is_tgt_prunable): return None
        
        is_head_pruning = len(src_targets) * len(dst_targets) != 0 and src_targets[0].split('.')[-1] == 'q_proj' and dst_targets[0].split('.')[-1] == 'o_proj'
        if is_head_pruning:
            group_size = num_query_heads // num_key_value_heads
            if indices is None:
                indices = cls.random_sample(
                    shape=(num_key_value_heads,),
                    dim=0,
                    sparsity=sparsity,
                    device=device,
                )
                indices = (torch.arange(0, group_size, device=device)[None, :] * indices[:, None]).flatten()
            
            for module_name in src_targets:
                module = get_module_recursive(root_module, module_name)
                module.weight = module.weight[indices if 'q_proj' in module_name else indices.reshape(-1, group_size)[:, 0] // group_size, :]
                module.out_features = module.weight.shape[0]
                cls.mark_pruned(module, 'N')
            
            for module_name in dst_targets:
                module = get_module_recursive(root_module, module_name)
                module.weight = module.weight[:, indices]
                module.in_features = module.weight.shape[1]
                cls.mark_pruned(module, 'K')
        
        else:
            if indices is None:
                indices = cls.random_sample(
                    shape=start_module.weight.shape,
                    dim=prune_dim,
                    sparsity=sparsity,
                    device=device,
                )
            
            for module_name in src_targets:
                module = get_module_recursive(root_module, module_name)
                if isinstance(module, nn.Linear):
                    module.weight = module.weight[indices, :]
                    module.out_features = module.weight.shape[0]
                elif isinstance(module, nn.Embedding):
                    module.weight = module.weight[:, indices]
                    module.embedding_dim = module.weight.shape[1]
                
                cls.mark_pruned(module, 'N')
        
            for module_name in dst_targets:
                module = get_module_recursive(root_module, module_name)
                if isinstance(module, nn.Linear):
                    module.weight = module.weight[:, indices]
                    module.in_features = module.weight.shape[1]
                
                cls.mark_pruned(module, 'K')
    
    @classmethod
    def monkey_patch(
        cls,
        root_module: nn.Module,
        target: str,
        layer_id: Optional[int]=0,
        dim: Optional[int]=1,
        indices: Optional[torch.Tensor]=None,
        sparsity: Optional[float]=0.5,
        **kwargs,
    ):
        """
        Gather target row / col for target parameters in layer_id, with its downstream dependency

        We only handle the NK-dim pruning propagation case for each linear layer (N[src linear] --> K[tgt linear], with extra handle for residual and norm), since the 'layer', 'attention', 'ffn' skipping are handled with M-dim pruning at the very begin
        """
        assert dim == 1 or (dim == 0 and target in cls.__propagate_targets__), f"target {target} under BN dim not in {cls.__propagate_targets__}"
        module = root_module.model.layers
        device = module[0].self_attn.q_proj.weight.device
        
        if target in ['layer', 'attention', 'ffn']:
            if indices is None: indices = cls.random_sample(
                shape=(len(module),),
                dim=0,
                sparsity=sparsity,
                device=device,
            )
            
            true_target = target if target != 'attention' else 'self_attn'
            for i in indices:
                if target == 'layer': module[i].forward = lambda x: x[0]
                else: getattr(module[i], true_target).forward = lambda x: x[0]

        else:
            start_module = module[layer_id]
            
            head_dim = start_module.self_attn.head_dim
            num_key_value_heads = start_module.self_attn.config.num_key_value_heads
            num_key_value_groups = start_module.self_attn.num_key_value_groups

            if dim == 0: # N-dim propagation
                if indices is None: indices = cls.random_sample(
                    shape=(num_key_value_heads,) if target == 'attention.q_proj' else get_module_recursive(start_module, target).weight.shape,
                    dim=0,
                    sparsity=sparsity,
                    device=device,
                )
                
                # [q k v] --> downstream [o]
                if target == 'attention.q_proj':
                    # N-dim
                    q_proj = get_module_recursive(start_module.self_attn, 'q_proj')
                    k_proj = get_module_recursive(start_module.self_attn, 'k_proj')
                    v_proj = get_module_recursive(start_module.self_attn, 'v_proj')

                    q_proj.weight = rearrange(q_proj.weight, '(h g d) k -> h g d k', d=head_dim, h=num_key_value_heads)[indices].flatten(0, -2)
                    k_proj.weight = rearrange(k_proj.weight, '(h d) k -> h d k', d=head_dim, h=num_key_value_heads)[indices].flatten(0, -2)
                    v_proj.weight = rearrange(v_proj.weight, '(h d) k -> h d k', d=head_dim, h=num_key_value_heads)[indices].flatten(0, -2)

                    # K-dim
                    o_proj = get_module_recursive(start_module.self_attn, 'o_proj')
                    o_proj.weight = rearrange(o_proj.weight, 'n (h g d) -> n h g d', d=head_dim, h=num_key_value_heads)[:, indices].flatten(1, -1)

                    q_proj.out_features = q_proj.weight.shape[0]
                    k_proj.out_features = k_proj.weight.shape[0]
                    v_proj.out_features = v_proj.weight.shape[0]
                    o_proj.in_features = o_proj.weight.shape[1]
                
                # [o] -> downstream [up gate]
                elif target == 'attention.o_proj':
                    # N-dim
                    o_proj = get_module_recursive(start_module.self_attn, 'o_proj')
                    o_proj.weight = o_proj.weight[indices]

                    # K-dim
                    norm = get_module_recursive(start_module, 'post_attention_layernorm')
                    up_proj = get_module_recursive(start_module.ffn, 'up_proj')
                    gate_proj = get_module_recursive(start_module.ffn, 'gate_proj')

                    norm.weight = norm.weight[indices]
                    up_proj.weight = up_proj.weight[:, indices]
                    gate_proj.weight = gate_proj.weight[:, indices]

                    o_proj.out_features = o_proj.weight.shape[0]
                    up_proj.in_features = up_proj.weight.shape[1]
                    gate_proj.in_features = gate_proj.weight.shape[1]

                    # check the dim of residual
                    if start_module.self_attn.q_proj.in_features != o_proj.out_features:
                        start_module.attn_res_proj = nn.Linear(start_module.self_attn.q_proj.in_features, o_proj.out_features, bias=False)
                
                # [up gate] -> downstream [down]
                elif target == 'ffn.up_proj':
                    # N-dim
                    up_proj = get_module_recursive(start_module.ffn, 'up_proj')
                    gate_proj = get_module_recursive(start_module.ffn, 'gate_proj')
                    up_proj.weight = up_proj.weight[indices]
                    gate_proj.weight = gate_proj.weight[indices]

                    down_proj = get_module_recursive(start_module.ffn, 'down_proj')
                    down_proj.weight = down_proj.weight[:, indices]

                    up_proj.out_features = up_proj.weight.shape[0]
                    gate_proj.out_features = gate_proj.weight.shape[0]
                    down_proj.in_features = down_proj.weight.shape[1]
                
                # [down] -> downstream next layer[q k v]
                elif target == 'ffn.down_proj':
                    # N-dim
                    down_proj = get_module_recursive(start_module.ffn, 'down_proj')
                    down_proj.weight = down_proj.weight[indices]

                    # K-dim
                    if layer_id < len(module) - 1:
                        end_module = module[layer_id + 1]
                        norm = get_module_recursive(end_module, 'input_layernorm')
                        q_proj = get_module_recursive(end_module.self_attn, 'q_proj')
                        k_proj = get_module_recursive(end_module.self_attn, 'k_proj')
                        v_proj = get_module_recursive(end_module.self_attn, 'v_proj')

                        norm.weight = norm.weight[indices]
                        q_proj.weight = q_proj.weight[:, indices]
                        k_proj.weight = k_proj.weight[:, indices]
                        v_proj.weight = v_proj.weight[:, indices]

                        down_proj.out_features = down_proj.weight.shape[0]
                        q_proj.in_features = q_proj.weight.shape[1]
                        k_proj.in_features = k_proj.weight.shape[1]
                        v_proj.in_features = v_proj.weight.shape[1]
                    else:
                        # cut the final norm & lm head to fit the dim
                        norm = get_module_recursive(root_module.model, 'norm')
                        lm_head = get_module_recursive(root_module, 'lm_head')

                        norm.weight = norm.weight[indices]
                        lm_head.weight = lm_head.weight[:, indices]
                        lm_head.in_features = lm_head.weight.shape[1]
                    
                    # check the dim of residual
                    if start_module.ffn.up_proj.in_features != down_proj.out_features:
                        start_module.ffn_res_proj = nn.Linear(start_module.ffn.up_proj.in_features, down_proj.out_features, bias=False)

            elif dim == 1: # K-dim
                start_module = get_module_recursive(start_module, target)
                if indices is None: indices = cls.random_sample(
                    shape=start_module.weight.shape,
                    dim=1,
                    sparsity=sparsity,
                    device=device,
                )
                start_module.weight = start_module.weight[:, indices]
                start_module.in_features = start_module.weight.shape[1]

                if target == 'attention.q_proj' and layer_id == 0:
                    # cut the vocabulary to fit the dim
                    vocab = get_module_recursive(root_module.model, 'embed_tokens')
                    vocab.weight = vocab.weight[:, indices]
                    vocab.embedding_dim = vocab.weight.shape[1]
        
        return indices
