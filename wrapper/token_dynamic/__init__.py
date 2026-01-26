from .skipgpt import SkipGPTForCausalLM
from config import register_wrapper

# Register SkipGPT wrapper
register_wrapper("skipgpt", "token_dynamic")(SkipGPTForCausalLM)

__all__ = ["SkipGPTForCausalLM"]