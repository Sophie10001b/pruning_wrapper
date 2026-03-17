import torch
import torch.nn as nn

from einops import rearrange
from typing import Optional, Tuple, Dict, List, Any

class BaseThreshold(nn.Module):
    """
    Base attention threshold class
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def generate_mask(self):
        raise NotImplementedError
    
    def get_threshold_kwargs(self):
        return {}


class BlasstThreshold(BaseThreshold):
    def __init__(self, threshold: Optional[float]=-1, **kwargs):
        super().__init__()
        self.threshold = threshold
    
    def generate_mask(
        self,
        kv_cache_len: int,
        sparsity: Optional[float]=0,
        device: Optional[torch.device]='cuda:0',
    ):
        block_num = (kv_cache_len + 16 - 1) // 16
        return torch.rand((block_num,), device=device) > sparsity
    
    def get_threshold_kwargs(self):
        return {'threshold': self.threshold}
