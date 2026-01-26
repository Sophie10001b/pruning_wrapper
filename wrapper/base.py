import torch
import torch.nn as nn

from typing import Optional, Tuple, Dict, List, Union, Any

from transformers import PretrainedConfig
from transformers.models import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput

class PrunedModelForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: PretrainedConfig,
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.lm_head = block.lm_head
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_values: Optional[Cache]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        labels: Optional[torch.LongTensor]=None,
        use_cache: Optional[bool]=None,
        cache_position: Optional[torch.LongTensor]=None,
        logits_to_keep: Union[int, torch.Tensor]=0,
        estimated_sparsity: Optional[float]=0.0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            estimated_sparsity=estimated_sparsity,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate_pruning_kwargs(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        - Generate random pruning settings for benchmark
        - Or update pruning settings for batch-dynamic pruning
        """
        raise NotImplementedError()
    
    def post_load(self, **kwargs):
        """
        - Post loading router, mask, and lora ... etc.
        """
        raise NotImplementedError()