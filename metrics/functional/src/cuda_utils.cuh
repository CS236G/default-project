#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERRORS()                                                    \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",           \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__,          \
              __FILE__);                                                       \
      exit(-1);                                                                \
    }                                                                          \
  }

#endif
