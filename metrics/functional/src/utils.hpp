#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

#define CHECK_IS_FLOAT(x)                                                      \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,                        \
              #x " must be a float tensor")

#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_IS_FLOAT(x)

#endif
