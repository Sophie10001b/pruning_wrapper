import os
import torch
import torch.nn as nn

from typing import Optional, Dict, List
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from config import load_config_and_class

def init_model(args: Namespace) -> Dict:
    """
    Initialize model wrapper and tokenizer based on arguments
    
    Args:
        args: Command line arguments containing 'model_path' key
        
    Returns:
        Dict containing 'model' and 'tokenizer'
    """
    model_path = getattr(args, "model_path", None)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # apply wrapper
    wrapper_config, wrapper_cls = load_config_and_class(args)
    model = wrapper_cls(
        config=config,
        pruning_config=wrapper_config,
        block=model,
    )

    model.post_load()
    
    return model, tokenizer