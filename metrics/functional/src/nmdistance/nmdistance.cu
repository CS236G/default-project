#include <ATen/ATen.h>

#include "../cuda_utils.cuh"

__global__ void nmdistance_kernel(int b, int n, const float *xyz, int m,
                                  const float *xyz2, float *result,
                                  int *result_i) {
  const int batch = 512;
  __shared__ float buf[batch * 3];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += batch) {
      int end_k = min(m, k2 + batch) - k2;
      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 3 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
           j += blockDim.x * gridDim.y) {
        float x1 = xyz[(i * n + j) * 3 + 0];
        float y1 = xyz[(i * n + j) * 3 + 1];
        float z1 = xyz[(i * n + j) * 3 + 2];
        int best_i = 0;
        float best = 0;
        int end_ka = end_k - (end_k & 3);
        if (end_ka == batch) {
          for (int k = 0; k < batch; k += 4) {
            {
              float x2 = buf[k * 3 + 0] - x1;
              float y2 = buf[k * 3 + 1] - y1;
              float z2 = buf[k * 3 + 2] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              float x2 = buf[k * 3 + 3] - x1;
              float y2 = buf[k * 3 + 4] - y1;
              float z2 = buf[k * 3 + 5] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              float x2 = buf[k * 3 + 6] - x1;
              float y2 = buf[k * 3 + 7] - y1;
              float z2 = buf[k * 3 + 8] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              float x2 = buf[k * 3 + 9] - x1;
              float y2 = buf[k * 3 + 10] - y1;
              float z2 = buf[k * 3 + 11] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              float x2 = buf[k * 3 + 0] - x1;
              float y2 = buf[k * 3 + 1] - y1;
              float z2 = buf[k * 3 + 2] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              float x2 = buf[k * 3 + 3] - x1;
              float y2 = buf[k * 3 + 4] - y1;
              float z2 = buf[k * 3 + 5] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              float x2 = buf[k * 3 + 6] - x1;
              float y2 = buf[k * 3 + 7] - y1;
              float z2 = buf[k * 3 + 8] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              float x2 = buf[k * 3 + 9] - x1;
              float y2 = buf[k * 3 + 10] - y1;
              float z2 = buf[k * 3 + 11] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          float x2 = buf[k * 3 + 0] - x1;
          float y2 = buf[k * 3 + 1] - y1;
          float z2 = buf[k * 3 + 2] - z1;
          float d = x2 * x2 + y2 * y2 + z2 * z2;
          if (k == 0 || d < best) {
            best = d;
            best_i = k + k2;
          }
        }
        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

void nmdistance(int b, int n, int m, at::Tensor xyz1, at::Tensor xyz2,
                at::Tensor dist1, at::Tensor dist2, at::Tensor idx1,
                at::Tensor idx2) {
  nmdistance_kernel<<<dim3(32, 16, 1), 512>>>(
      b, n, xyz1.data<float>(), m, xyz2.data<float>(), dist1.data<float>(),
      idx1.data<int>());
  nmdistance_kernel<<<dim3(32, 16, 1), 512>>>(
      b, m, xyz2.data<float>(), n, xyz1.data<float>(), dist2.data<float>(),
      idx2.data<int>());
  CUDA_CHECK_ERRORS();
}

__global__ void nmdistance_grad_kernel(int b, int n, const float *xyz1, int m,
                                       const float *xyz2,
                                       const float *grad_dist1, const int *idx1,
                                       float *grad_xyz1, float *grad_xyz2) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
         j += blockDim.x * gridDim.y) {
      float x1 = xyz1[(i * n + j) * 3 + 0];
      float y1 = xyz1[(i * n + j) * 3 + 1];
      float z1 = xyz1[(i * n + j) * 3 + 2];
      int j2 = idx1[i * n + j];
      float x2 = xyz2[(i * m + j2) * 3 + 0];
      float y2 = xyz2[(i * m + j2) * 3 + 1];
      float z2 = xyz2[(i * m + j2) * 3 + 2];
      float g = grad_dist1[i * n + j] * 2;
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 2]), g * (z1 - z2));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 1]), -(g * (y1 - y2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 2]), -(g * (z1 - z2)));
    }
  }
}

void nmdistance_grad(int b, int n, int m, at::Tensor xyz1, at::Tensor xyz2,
                     at::Tensor gradxyz1, at::Tensor gradxyz2,
                     at::Tensor graddist1, at::Tensor graddist2,
                     at::Tensor idx1, at::Tensor idx2) {
  nmdistance_grad_kernel<<<dim3(1, 16, 1), 256>>>(
      b, n, xyz1.data<float>(), m, xyz2.data<float>(), graddist1.data<float>(),
      idx1.data<int>(), gradxyz1.data<float>(), gradxyz2.data<float>());
  nmdistance_grad_kernel<<<dim3(1, 16, 1), 256>>>(
      b, m, xyz2.data<float>(), n, xyz1.data<float>(), graddist2.data<float>(),
      idx2.data<int>(), gradxyz2.data<float>(), gradxyz1.data<float>());
  CUDA_CHECK_ERRORS();
}
