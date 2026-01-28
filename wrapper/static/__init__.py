from .dense import DenseForCausalLM
from config import register_wrapper

register_wrapper("dense", "static")(DenseForCausalLM)

__all__ = ["DenseForCausalLM"]
