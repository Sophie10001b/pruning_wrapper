import torch
import tilelang
import tilelang.language as T

from typing import Optional, Tuple, Dict, List, Union, Any
from einops import rearrange
from ops.utils import get_autotune_config, get_autotune_cache, check_shared_memory_attn

##############################################################
#                     Seer Attention
##############################################################