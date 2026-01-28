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

#################### FFN ####################
class DenseAttention(nn.Module):
    """
    HF style attention with FA2
    """
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: nn.Module,
        **kwargs,
    ):
        super().__init__()
        self._support_pruning_components = []
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

        self.attention_impl = __ATTENTION__['base']()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        past_key_values: Optional[Cache]=None,
        cache_position: Optional[torch.LongTensor]=None,
        pad_offset: Optional[torch.Tensor]=None,
        **kwargs,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

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
        )
        attn_output = rearrange(attn_output, '... h d -> ... (h d)')
        attn_output = self.o_proj(attn_output)

        return attn_output


#################### Layer ####################
class DenseDecoderLayer(nn.Module):
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
        super().__init__()
        self._support_pruning_components = []
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
        
        self.self_attn = DenseAttention(config, pruning_config, block.self_attn, **kwargs)
        self.mlp = block.mlp
    
    def generate_pruning_kwargs(
        self,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return dict()

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
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        with record_function("**Attention**"):
            hidden_states = self.self_attn(
                hidden_states,
                attention_mask=attention_mask if not kwargs.get('use_cuda_graph', False) else None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                pad_offset=pad_offset,
                **kwargs,
            )

        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        with record_function("**FFN**"):
            hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states

class DensePretrainedModel(PreTrainedModel):
    config: PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DenseDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = False
    _can_record_outputs = {
        "hidden_states": DenseDecoderLayer,
        "attentions": DenseAttention,
    }

class DenseModel(DensePretrainedModel):
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
        self.layers = nn.ModuleList([DenseDecoderLayer(config, pruning_config, block.layers[i], i) for i in range(config.num_hidden_layers)])
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

class DenseForCausalLM(PrunedModelForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config, block, **kwargs)
        self.model = DenseModel(config, pruning_config, block.model, **kwargs)
    
    # Generate random route mask for benchmark
    def generate_pruning_kwargs(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return dict()
    
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