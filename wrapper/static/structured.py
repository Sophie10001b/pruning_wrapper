import torch
import torch.nn as nn

from typing import Optional, Tuple, Dict, List, Union, Any, Sequence
from einops import rearrange

from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from torch.profiler import record_function

from ops import __ATTENTION__, __MLP__, __ROUTER__, __KV_CACHE__, __APPROXIMATOR__, __INDEX__
from ops.utils import triton_rmsnorm, triton_rope_qk_align
from wrapper.base import PrunedModelForCausalLM
from wrapper.static.dense import DenseAttention, DenseMLP, DenseDecoderLayer, DenseModel, DenseForCausalLM

#################### Layer ####################
class StructuredDecoderLayer(DenseDecoderLayer):
    """
    HF style decoder layer
    """
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: nn.Module,
        layer_idx: int,
        **kwargs,
    ):
        super().__init__(config, pruning_config, block, layer_idx, **kwargs)
        self._propagate_group = tuple(
            ("attention.q_proj", "attention.k_proj", "attention.v_proj"),
            ("attention.o_proj"),
            ("ffn.up_proj", "ffn.gate_proj"),
            ("ffn.down_proj"),
        )

class StructuredPretrainedModel(PreTrainedModel):
    config: PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["StructuredDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = False
    _can_record_outputs = {
        "hidden_states": StructuredDecoderLayer,
        "attentions": DenseAttention,
    }

class StructuredModel(StructuredPretrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config)
        self.pruning_config = pruning_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = block.embed_tokens
        self.layers = nn.ModuleList([StructuredDecoderLayer(config, pruning_config, block.layers[i], i) for i in range(config.num_hidden_layers)])
        self.norm = block.norm
        self.rotary_emb = block.rotary_emb
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_values: Optional[Cache]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        cache_position: Optional[torch.LongTensor]=None,
        use_cache: Optional[bool]=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = __KV_CACHE__[self.pruning_config.get('cache_type', 'base')](self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        pad_offset = attention_mask.shape[1] - attention_mask.sum(-1)

        for i, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                pad_offset=pad_offset,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        

class UnstructuredForCausalLM(PrunedModelForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config, pruning_config, block, **kwargs)

        # initialize pruning chain
        self.pruning_chain = []
        if pruning_config.get('prune_embedding', False): self.pruning_chain.append('model.embed_tokens')
        for i, layer in enumerate(self.model.layers):
            layer_chain = layer._propagate_group
            self.pruning_chain.append([])
            for group in layer_chain:
                self.pruning_chain[-1].append(tuple([f'model.layers.{i}.{component}' for component in group]))
        
        if self.pruning_config.get('prune_lm_head', False): self.pruning_chain.append('lm_head')

        self.model = StructuredModel(config, pruning_config, block.model, **kwargs)
    
    # Generate random route mask for benchmark
    def generate_pruning_kwargs(self, **kwargs) -> Dict[str, torch.Tensor]:
        pruning_kwargs = {}
        attention_components = set(['attention.q_proj', 'attention.k_proj', 'attention.v_proj', 'attention.o_proj'])
        ffn_components = set(['ffn.up_proj', 'ffn.gate_proj', 'ffn.down_proj'])

        patch_type = self.pruning_config.get('patch_type', 'structured')

        # layer & sublayer first
        layer_pruning_config = self.pruning_config.get('layer', {})
        attention_pruning_config = self.pruning_config.get('attention', {})
        ffn_pruning_config = self.pruning_config.get('ffn', {})

        layer_skip_ids = []
        if layer_pruning_config.get('pruning_type', None) is not None:
            layer_skip_ids = set(__INDEX__[patch_type].monkey_patch_layer(
                root_module=self,
                sparsity=layer_pruning_config.get('estimated_sparsity', 0)
            ))
        
        attention_skip_ids = []
        if attention_pruning_config.get('pruning_type', None) is not None:
            attention_skip_ids = set(__INDEX__[patch_type].monkey_patch_sublayer(
                root_module=self,
                target='attention',
                sparsity=attention_pruning_config.get('estimated_sparsity', 0)
            ))
        
        ffn_skip_ids = []
        if ffn_pruning_config.get('pruning_type', None) is not None:
            ffn_skip_ids = set(__INDEX__[patch_type].monkey_patch_sublayer(
                root_module=self,
                target='ffn',
                sparsity=ffn_pruning_config.get('estimated_sparsity', 0)
            ))
        
        # then for each layer's row & col
        i = 0
        while i < len(self.pruning_chain):
            group = self.pruning_chain[i]
            if not isinstance(group, Sequence): continue
            layer_id = int(group[0].split('.')[2])

            component = group[0]
            if component in attention_components and layer_id in attention_skip_ids: continue
            if component in ffn_components and layer_id in ffn_skip_ids: continue

            prefix, suffix = component.split('.')
            component_config = self.pruning_config.get(prefix, {}).get(suffix, {})
            pruning_type = component_config.get('pruning_type', None)
            src_targets = dst_targets = ()
            if pruning_type == 'bn':
                src_targets = group
                dst_targets = self.pruning_chain[i+1] if i+1 < len(self.pruning_chain) else ()
            elif pruning_type == 'bk':
                src_targets = self.pruning_chain[0] if i == 1 and self.pruning_chain[0] == 'model.embed_tokens' else ()
                dst_targets = group
            else: continue
            
            is_multi_block = False
            if len(src_targets) * len(dst_targets) > 0 or src_targets[0].split('.')[0] != dst_targets[0].split('.')[0]:
                is_multi_block = True

            component_skip_ids = set(__INDEX__[patch_type].monkey_patch_nk(
                root_module=self,
                src_targets=src_targets,
                dst_targets=dst_targets,
                sparsity=component_config.get('estimated_sparsity', 0),
                is_multi_block=is_multi_block,
                num_query_heads=self.config.num_attention_heads,
                num_key_heads=self.config.num_key_value_heads,
            ))

            i += 2 if pruning_type == 'bn' else 1
        
        return {}
    
    def post_load(
        self,
        router_ckpt: Optional[Dict[str, torch.Tensor]]=None,
        lora_ckpt: Optional[Dict[str, torch.Tensor]]=None,
        full_ckpt: Optional[Dict[str, torch.Tensor]]=None,
        lora_rank: Optional[int]=16,
        lora_alpha: Optional[float]=32,
        **kwargs,
    ):
        """
        SkipGPT-style post load:
        - First, we load all the router from ckpt
        - Then, we load all the lora adapter from ckpt
        """
        pass