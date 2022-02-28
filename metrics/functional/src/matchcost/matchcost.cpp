#include "matchcost.hpp"
#include "../utils.hpp"
#include "matchcost.cuh"

at::Tensor approxmatch_forward(const at::Tensor xyz1, const at::Tensor xyz2) {
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  const int b = xyz1.size(0);
  const int n = xyz1.size(1);
  const int m = xyz2.size(1);
  at::Tensor match = at::empty({b, m, n}, xyz1.options());
  at::Tensor temp = at::empty({b, (n + m) * 2}, xyz1.options());
  approxmatch(b, n, m, xyz1, xyz2, match, temp);
  return match;
}

at::Tensor matchcost_forward(const at::Tensor xyz1, const at::Tensor xyz2,
                             const at::Tensor match) {
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  const int b = xyz1.size(0);
  const int n = xyz1.size(1);
  const int m = xyz2.size(1);
  at::Tensor cost = at::empty({b}, xyz1.options());
  matchcost(b, n, m, xyz1, xyz2, match, cost);
  return cost;
}

std::vector<at::Tensor> matchcost_backward(const at::Tensor grad_cost,
                                           const at::Tensor xyz1,
                                           const at::Tensor xyz2,
                                           const at::Tensor match) {
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  const int b = xyz1.size(0);
  const int n = xyz1.size(1);
  const int m = xyz2.size(1);
  at::Tensor grad1 = at::empty({b, n, 3}, xyz1.options());
  at::Tensor grad2 = at::empty({b, m, 3}, xyz1.options());
  matchcost_grad(b, n, m, grad_cost, xyz1, xyz2, match, grad1, grad2);
  return std::vector<at::Tensor>({grad1, grad2});
}
