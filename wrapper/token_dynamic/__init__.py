from .skipgpt import SkipGPTForCausalLM
from .sparse_attention import SparseAttentionForCausalLM

from config import register_wrapper

# Register SkipGPT wrapper
register_wrapper("skipgpt", "token_dynamic")(SkipGPTForCausalLM)
register_wrapper("sparse_attention", "token_dynamic")(SparseAttentionForCausalLM)

__all__ = ["SkipGPTForCausalLM", "SparseAttentionForCausalLM"]