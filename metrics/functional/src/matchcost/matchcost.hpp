#ifndef _MATCHCOST_HPP
#define _MATCHCOST_HPP

#include <torch/torch.h>
#include <vector>

at::Tensor approxmatch_forward(const at::Tensor xyz1, const at::Tensor xyz2);

at::Tensor matchcost_forward(const at::Tensor xyz1, const at::Tensor xyz2,
                             const at::Tensor match);

std::vector<at::Tensor> matchcost_backward(const at::Tensor grad_cost,
                                           const at::Tensor xyz1,
                                           const at::Tensor xyz2,
                                           const at::Tensor match);

#endif
