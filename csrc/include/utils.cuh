#pragma once

#include <mma.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "flashinfer/permuted_smem.cuh"
#include "flashinfer/cp_async.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "flashinfer/mma.cuh"

#include "torch/extension.h"
#include "torch/csrc/utils/pybind.h"

#ifndef NUM_THREADS_PER_WARP
#define NUM_THREADS_PER_WARP 32
#endif

#ifndef CDIV
#define CDIV(x, y) ((x + y - 1) / y)
#endif

namespace cg = cooperative_groups;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

#define DISPATCH_TORCH_TO_NV_WITH_UINT32(pytorch_float_type, nv_float_type, nv_int_type, ...) [&]() -> bool { \
  using nv_int_type = uint32_t;                                                                     \
  switch (pytorch_float_type) {                                                                     \
    case at::ScalarType::Half: {                                                                    \
      using nv_float_type = nv_half;                                                                \
      return __VA_ARGS__();                                                                         \
    }                                                                                               \
    case at::ScalarType::BFloat16: {                                                                \
      using nv_float_type = nv_bfloat16;                                                            \
      return __VA_ARGS__();                                                                         \
    }                                                                                               \
  }                                                                                                 \
} ()