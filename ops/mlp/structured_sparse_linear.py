import os
import itertools
import torch
import torch.nn as nn

from functools import partial
from typing import Optional, Tuple, Dict, Any
from einops import rearrange
from triton.testing import do_bench, do_bench_cudagraph

from .tilelang_kernel.spmm import structured_sparse_gemm

os.environ['CUDA_LAUNCH_BLOCKING']='0'

STYLES = [('blue', '-'), ('red', '-'), ('green', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-'), ('pink', '-'), ('gray', '-'), ('olive', '-'), ('cyan', '-')]

