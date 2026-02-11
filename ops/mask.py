import torch
import torch.nn as nn

from einops import rearrange
from typing import Optional, Dict, Sequence
from torch.sparse.semi_structured import SparseSemiStructuredTensorCUTLASS

###########################
#   Mask for Unstructured
###########################

class BaseMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        pass
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        pass


class UnstructuredMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        return torch.rand(shape, device=device) < sparsity
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        if mask is None:
            mask = cls.random_sample(module.weight.shape, device=module.weight.device, **kwargs)
        
        module.weight.data *= mask

class SemiStructuredMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        pattern: Optional[str]='2:4',
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        left, right = map(int, pattern.split(':'))
        assert left / right == sparsity, f'pattern {pattern} is not compatible with sparsity {sparsity}'

        mask = torch.ones(shape, device=device).flatten()
        mask = rearrange(mask, '(a b) -> a b', b=right)
        indices = torch.multinomial(mask, left, replacement=False)

        rows = torch.arange(mask.shape[0], device=device).view(-1, 1).expand(-1, left)
        mask[rows, indices] = False
        return mask
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        if mask is None:
            mask = cls.random_sample(module.weight.shape, device=module.weight.device, **kwargs)
        
        module.weight.data *= mask
        sparse_weight = SparseSemiStructuredTensorCUTLASS.from_dense(module.weight.data)
        module.weight.data = sparse_weight
