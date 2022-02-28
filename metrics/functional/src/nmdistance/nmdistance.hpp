#ifndef _NMDISTANCE_HPP
#define _NMDISTANCE_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> nmdistance_forward(at::Tensor xyz1, at::Tensor xyz2);

std::vector<at::Tensor> nmdistance_backward(at::Tensor xyz1, at::Tensor xyz2,
                                            at::Tensor graddist1,
                                            at::Tensor graddist2,
                                            at::Tensor idx1, at::Tensor idx2);

#endif
