from .dense import DenseForCausalLM
from .unstructured import UnstructuredForCausalLM
from config import register_wrapper

register_wrapper("dense", "static")(DenseForCausalLM)
register_wrapper("unstructured", "static")(UnstructuredForCausalLM)

__all__ = [
    "DenseForCausalLM",
    "UnstructuredForCausalLM",
]
