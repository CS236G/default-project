#include "nmdistance.hpp"
#include "../utils.hpp"
#include "nmdistance.cuh"

std::vector<at::Tensor> nmdistance_forward(at::Tensor xyz1, at::Tensor xyz2) {
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  const int b = xyz1.size(0);
  const int n = xyz1.size(1);
  const int m = xyz2.size(1);
  at::Tensor dist1 = at::empty({b, n}, xyz1.options());
  at::Tensor dist2 = at::empty({b, m}, xyz1.options());
  at::Tensor idx1 = at::empty(
      {b, n},
      torch::TensorOptions().dtype(torch::kInt32).device(xyz1.device()));
  at::Tensor idx2 = at::empty(
      {b, m},
      torch::TensorOptions().dtype(torch::kInt32).device(xyz1.device()));
  nmdistance(b, n, m, xyz1, xyz2, dist1, dist2, idx1, idx2);
  return std::vector<at::Tensor>({dist1, dist2, idx1, idx2});
}

std::vector<at::Tensor> nmdistance_backward(at::Tensor xyz1, at::Tensor xyz2,
                                            at::Tensor graddist1,
                                            at::Tensor graddist2,
                                            at::Tensor idx1, at::Tensor idx2) {
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  CHECK_INPUT(graddist1);
  CHECK_INPUT(graddist2);
  const int b = xyz1.size(0);
  const int n = xyz1.size(1);
  const int m = xyz2.size(1);
  at::Tensor gradxyz1 = at::zeros({b, n, 3}, xyz1.options());
  at::Tensor gradxyz2 = at::zeros({b, m, 3}, xyz2.options());
  nmdistance_grad(b, n, m, xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2,
                  idx1, idx2);
  return std::vector<at::Tensor>({gradxyz1, gradxyz2});
}
