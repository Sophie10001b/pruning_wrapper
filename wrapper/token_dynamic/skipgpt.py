import torch
import torch.nn as nn

from typing import Optional, Tuple, Dict, List, Union, Any
from einops import rearrange

from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from torch.profiler import record_function

from ops import __ATTENTION__, __MLP__, __ROUTER__, __KV_CACHE__, __APPROXIMATOR__
from ops.utils import triton_rmsnorm, triton_rope_qk_align
from wrapper.base import PrunedModelForCausalLM

def dispatch(key, candidates: Dict[str, Any], default: str, **kwargs):
    if default is None: default = 'null'
    cls = candidates.get(key, candidates.get(default, None))
    if cls is None: return None
    else: return cls(**kwargs)

def get_router_group_num(config: PretrainedConfig, prefix: str, pruning_type: str, block_size: Optional[int]=1):
    if prefix == 'attention':
        if pruning_type in ['query']: return 1
        elif pruning_type in ['group']: return config.num_key_value_heads
        elif pruning_type in ['head']: return config.num_attention_heads
    elif prefix == 'ffn':
        if pruning_type in ['bm']: return 1
        elif pruning_type in ['bn']: return config.intermediate_size // block_size
    elif prefix in ['up_proj', 'gate_proj']:
        if pruning_type in ['bm']: return 1
        elif pruning_type in ['bn']: return config.intermediate_size // block_size
        elif pruning_type in ['bk']: return config.hidden_size // block_size
    elif prefix == 'down_proj':
        if pruning_type in ['bm']: return 1
        elif pruning_type in ['bn']: return config.hidden_size // block_size
        elif pruning_type in ['bk']: return config.intermediate_size // block_size
    else:
        if pruning_type in ['bm']: return 1
        elif pruning_type in ['bn', 'bk']: return config.hidden_size // block_size


#################### FFN ####################
class SkipGPTMLP(nn.Module):
    """
    SkipGPT style MLP pruning impl

    - **M-axis Pruning:** Generate route mask [B, L], with up,gate,down in `M-axis` pruning
    - **N-axis Pruning:** Generate route mask [B, L, NG], with up,gate in `N-axis` pruning and down in `K-axis` pruning
    - **K-axis Pruning:** Generate route mask [B, L, NG], with up,gate,down in `K-axis` pruning
    """
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: nn.Module,
        **kwargs,
    ):
        super().__init__()
        self._support_pruning_components = ['ffn.', 'ffn.up_proj', 'ffn.gate_proj', 'ffn.down_proj']
        self.config = config
        self.pruning_config = pruning_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.activation = config.hidden_act

        self.up_proj = block.up_proj
        self.gate_proj = block.gate_proj
        self.down_proj = block.down_proj

        # ffn pruning impl
        for key in self._support_pruning_components:
            prefix, suffix = key.split('.')
            if suffix == '': setattr(self, f'{prefix}_kwargs', self.pruning_config.get(prefix, {}))
            else: setattr(self, f'{suffix}_kwargs', self.pruning_config[prefix].get(suffix, {}))
        
        self.ffn_impl = None
        self.up_proj_impl = None
        self.gate_proj_impl = None
        self.down_proj_impl = None

        self._init_impl()
    
    def _init_impl(self):
        for key in self._support_pruning_components:
            prefix, suffix = key.split('.')
            if suffix == '': suffix = prefix
            pruning_type = getattr(self, f'{suffix}_kwargs', {}).get('pruning_type', None)
            if pruning_type is None: continue
            impl_name = f'{suffix}_impl'
            setattr(
                self,
                impl_name,
                dispatch(
                    pruning_type,
                    __MLP__,
                    'base',
                ),
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        pruning_kwargs: Optional[Dict[str, Any]]=None,
        **kwargs,
    ):
        route_mask = pruning_kwargs.get('ffn', None)
        # ffn fuse
        if self.ffn_impl is not None:
            ffn_output = self.ffn_impl(
                hidden_states,
                w_up=self.up_proj.weight,
                w_gate=self.gate_proj.weight,
                w_down=self.down_proj.weight,
                b_up=self.up_proj.bias,
                b_gate=self.gate_proj.bias,
                route_mask=route_mask,
                activation=self.activation,
                **getattr(self, f'ffn_kwargs', {}), # pruning_type, backend, estimated_sparsity, prefill_impl, decode_impl, etc.
            )
        else:
            up_output = self.up_proj_impl(
                hidden_states,
                w_up=self.up_proj.weight,
                b_up=self.up_proj.bias,
                route_mask=pruning_kwargs.get('up_proj', None),
                **getattr(self, f'up_proj_kwargs', {}),
            )
            gate_output = self.gate_proj_impl(
                hidden_states,
                w_up=self.gate_proj.weight,
                b_up=self.gate_proj.bias,
                route_mask=pruning_kwargs.get('gate_proj', None),
                activation=self.activation,
                **getattr(self, f'gate_proj_kwargs', {}),
            )
            ffn_output = up_output * gate_output
            ffn_output = self.down_proj_impl(
                ffn_output,
                w_up=self.down_proj.weight,
                b_up=self.down_proj.bias,
                route_mask=pruning_kwargs.get('down_proj', None),
                **getattr(self, f'down_proj_kwargs', {}),
            )
        
        return ffn_output

#################### Attention ####################
class SkipGPTAttention(nn.Module):
    """
    SkipGPT style attention pruning impl

    - **Query Pruning:** Generate route mask [B, L], with q,o in `M-axis` pruning
    - **Query Head Group Pruning:** Generate route mask [B, L, Hk], with q in `N-axis` pruning and o in `K-axis` pruning
    - **Query Head Pruning:** Generate route mask [B, L, Hq] with q in `N-axis` pruning and o in `K-axis` pruning
    """
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: nn.Module,
        **kwargs,
    ):
        super().__init__()
        self._support_pruning_components = ['attention.', 'attention.q_proj', 'attention.k_proj', 'attention.v_proj', 'attention.o_proj']
        self.config = config
        self.pruning_config = pruning_config
        self.layer_idx = block.layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = block.q_proj
        self.k_proj = block.k_proj
        self.v_proj = block.v_proj
        self.o_proj = block.o_proj

        self.q_norm = getattr(block, "q_norm", None)
        self.k_norm = getattr(block, "k_norm", None)

        # attention pruning impl
        for key in self._support_pruning_components:
            prefix, suffix = key.split('.')
            if suffix == '': setattr(self, f'{prefix}_kwargs', self.pruning_config.get(prefix, {}))
            else: setattr(self, f'{suffix}_kwargs', self.pruning_config[prefix].get(suffix, {}))
        
        if self.attention_kwargs['pruning_type'] == 'query':
            for key in ['q_proj', 'o_proj']:
                attr = getattr(self, f'{key}_kwargs', {})
                attr['pruning_type'] = 'bm'
        elif self.attention_kwargs['pruning_type'] in ['group', 'head']:
            self.q_proj_kwargs['pruning_type'] = 'bn'
            self.o_proj_kwargs['pruning_type'] = 'bk'

        # attention pruning impl
        self.q_proj_impl = None
        self.k_proj_impl = None
        self.v_proj_impl = None
        self.o_proj_impl = None
        self.attention_impl = None
        self._init_impl()
    
    def _init_impl(self):
        for key in self._support_pruning_components:
            prefix, suffix = key.split('.')
            if suffix == '': suffix = prefix
            pruning_type = getattr(self, f'{suffix}_kwargs', {}).get('pruning_type', None)
            impl_name = f'{suffix}_impl'
            setattr(
                self,
                impl_name,
                dispatch(
                    pruning_type,
                    __MLP__ if suffix != 'attention' else __ATTENTION__,
                    'base',
                ),
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        past_key_values: Optional[Cache]=None,
        cache_position: Optional[torch.LongTensor]=None,
        pad_offset: Optional[torch.Tensor]=None,
        pruning_kwargs: Optional[Dict[str, Any]]=None,
        **kwargs,
    ) -> torch.Tensor:
        route_mask = pruning_kwargs.get('attention', None)
        q = self.q_proj_impl(
            hidden_states,
            w_up=self.q_proj.weight,
            b_up=self.q_proj.bias,
            route_mask=pruning_kwargs.get('q_proj', route_mask),
            **getattr(self, f'q_proj_kwargs', {}),
        )
        k = self.k_proj_impl(
            hidden_states,
            w_up=self.k_proj.weight,
            b_up=self.k_proj.bias,
            route_mask=pruning_kwargs.get('k_proj', None),
            **getattr(self, f'k_proj_kwargs', {}),
        )
        v = self.v_proj_impl(
            hidden_states,
            w_up=self.v_proj.weight,
            b_up=self.v_proj.bias,
            route_mask=pruning_kwargs.get('v_proj', None),
            **getattr(self, f'v_proj_kwargs', {}),
        )

        q, k, v = list(map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), [q, k, v]))
        if self.q_norm: q = self.q_norm(q)
        if self.k_norm: k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = triton_rope_qk_align(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        
        attn_output = self.attention_impl(
            q, k, v,
            attention_mask=attention_mask,
            pad_offset=pad_offset,
            route_mask=route_mask,
            **getattr(self, f'attention_kwargs', {}),
        )
        attn_output = rearrange(attn_output, '... h d -> ... (h d)')
        attn_output = self.o_proj_impl(
            attn_output,
            w_up=self.o_proj.weight,
            b_up=self.o_proj.bias,
            route_mask=pruning_kwargs.get('o_proj', route_mask),
            **getattr(self, f'o_proj_kwargs', {}),
        )

        return attn_output

#################### Layer ####################
class SkipGPTDecoderLayer(nn.Module):
    """
    SkipGPT style layer pruning impl
    """
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: nn.Module,
        layer_idx: int,
        **kwargs,
    ):
        super().__init__()
        self._support_pruning_components = ['layer.', 'attention.', 'ffn.', 'ffn.up_proj', 'ffn.gate_proj', 'ffn.down_proj']
        self.config = config
        self.pruning_config = pruning_config
        self.layer_idx = layer_idx
        self.input_layernorm = block.input_layernorm
        self.post_attention_layernorm = block.post_attention_layernorm

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_attention_heads

        # router init
        self.layer_router = None
        self.attention_router = None
        self.ffn_router = None
        self.up_proj_router = None
        self.gate_proj_router = None
        self.down_proj_router = None

        # 'diff' for ffn router generate mask after attn block finished
        self.router_order = self.pruning_config['ffn'].get('router', {}).get('router_order', 'same')
        self.layer_pruning = self.pruning_config['layer'].get('pruning_type', None) is not None
        if self.layer_pruning:
            # equal to query & bm pruning
            self.pruning_config['attention']['pruning_type'] = 'query'
            self.pruning_config['attention']['router']['router_type'] = None
            self.pruning_config['ffn']['pruning_type'] = 'bm'
            self.pruning_config['ffn']['router']['router_type'] = None
        
        if self.pruning_config['ffn']['pruning_type'] != None:
            self.pruning_config['ffn']['up_proj']['pruning_type'] = None
            self.pruning_config['ffn']['gate_proj']['pruning_type'] = None
            self.pruning_config['ffn']['down_proj']['pruning_type'] = None
        
        self._init_router()
        
        # approximator init
        self.layer_approximator = None
        self.attn_approximator = None
        self.ffn_approximator = None
        
        self.self_attn = SkipGPTAttention(config, pruning_config, block.self_attn, **kwargs)
        self.mlp = SkipGPTMLP(config, pruning_config, block.mlp, **kwargs)
    
    def _init_router(self):
        for key in self._support_pruning_components:
            prefix, suffix = key.split('.')
            router_type = self.pruning_config[prefix]['router'].get('router_type', None)
            rank_size = self.pruning_config[prefix]['router'].get('router_rank_size', 16)

            if suffix == '':
                pruning_config = self.pruning_config[prefix]
                suffix = prefix
            else:
                pruning_config = self.pruning_config[prefix][suffix]

            pruning_type = pruning_config.get('pruning_type', None)
            if pruning_type is None or router_type is None: continue

            block_size = pruning_config.get('block_size', 1)
            num_groups = get_router_group_num(self.config, suffix, pruning_type, block_size)
            if router_type == None: continue
            router_name = f'{suffix}_router'
            setattr(
                self,
                router_name,
                dispatch(
                    router_type,
                    __ROUTER__,
                    None,
                    hidden_size=self.hidden_size,
                    rank_size=rank_size,
                    num_groups=num_groups,
                ),
            )
    
    def generate_pruning_kwargs(
        self,
        hidden_states: torch.Tensor,
        pruning_targets: List[str]=['layer', 'attention', 'ffn', 'up_proj', 'gate_proj', 'down_proj'],
        estimated_sparsity: Optional[float]=-1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        with record_function("**Router**"):
            pruning_kwargs = {}
            for name in pruning_targets:
                router_name = f'{name}_router'
                if getattr(self, router_name, None) is not None:
                    route = getattr(self, router_name)(hidden_states)[0]
                    if name == 'layer':
                        pruning_kwargs['attention'] = pruning_kwargs['ffn'] = route
                    else:
                        pruning_kwargs[name] = route
                    
            if estimated_sparsity >= 0: # generated benchmarking mask
                for key in pruning_kwargs.keys():
                    pruning_kwargs[key] = torch.rand_like(pruning_kwargs[key].float()) > estimated_sparsity

        return pruning_kwargs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        pad_offset: Optional[torch.Tensor]=None,
        pruning_kwargs: Optional[Dict[str, Any]]=None,
        **kwargs,
    ):
        # 1. pre attn pruning
        pruning_targets = ['layer', 'attention']
        if self.router_order == 'same': 
            pruning_targets += ['ffn', 'up_proj', 'gate_proj', 'down_proj']
        local_pruning_kwargs = self.generate_pruning_kwargs(hidden_states, pruning_targets)
        
        # 1.1 check if pruning_kwargs includes route mask
        use_predefined_route = pruning_kwargs is not None

        # 2. attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        with record_function("**Attention**"):
            hidden_states = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                pad_offset=pad_offset,
                pruning_kwargs=pruning_kwargs if use_predefined_route else local_pruning_kwargs,
                **kwargs,
            )

        hidden_states = hidden_states + residual

        # 3. FFN
        if self.router_order == 'diff':
            pruning_targets = ['ffn', 'up_proj', 'gate_proj', 'down_proj']
            local_pruning_kwargs = self.generate_pruning_kwargs(hidden_states, pruning_targets)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        with record_function("**FFN**"):
            hidden_states = self.mlp(
                hidden_states,
                pruning_kwargs=pruning_kwargs if use_predefined_route else local_pruning_kwargs,
                **kwargs,
            )
        hidden_states = hidden_states + residual
        return hidden_states

class SkipGPTPretrainedModel(PreTrainedModel):
    config: PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["SkipGPTDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = False
    _can_record_outputs = {
        "hidden_states": SkipGPTDecoderLayer,
        "attentions": SkipGPTAttention,
    }

class SkipGPTModel(SkipGPTPretrainedModel):
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
        self.layers = nn.ModuleList([SkipGPTDecoderLayer(config, pruning_config, block.layers[i], i) for i in range(config.num_hidden_layers)])
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
        pruning_kwargs: Optional[Dict[str, Any]]=None,
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
                pruning_kwargs=pruning_kwargs[f'layer_{i}'] if pruning_kwargs is not None else None,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class SkipGPTForCausalLM(PrunedModelForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config, block, **kwargs)
        self.model = SkipGPTModel(config, pruning_config, block.model, **kwargs)
    
    # Generate random route mask for benchmark
    def generate_pruning_kwargs(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        pruning_kwargs = {}
        hidden_states = self.model.embed_tokens(input_ids)
        for i in range(self.config.num_hidden_layers):
            pruning_kwargs[f'layer_{i}'] = self.model.layers[i].generate_pruning_kwargs(hidden_states=hidden_states, **kwargs)
        
        return pruning_kwargs
    
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
