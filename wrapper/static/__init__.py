from .dense import DenseForCausalLM
from .unstructured import UnstructuredForCausalLM
from .structured import StructuredForCausalLM
from config import register_wrapper

register_wrapper("dense", "static")(DenseForCausalLM)
register_wrapper("unstructured", "static")(UnstructuredForCausalLM)
register_wrapper("structured", "static")(StructuredForCausalLM)

__all__ = [
    "DenseForCausalLM",
    "UnstructuredForCausalLM",
    "StructuredForCausalLM",
]
