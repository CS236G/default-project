#ifndef _MATCHCOST_CUH
#define _MATCHCOST_CUH

void approxmatch(int b, int n, int m, const at::Tensor xyz1,
                 const at::Tensor xyz2, const at::Tensor match,
                 const at::Tensor temp);

void matchcost(int b, int n, int m, const at::Tensor xyz1,
               const at::Tensor xyz2, const at::Tensor match,
               const at::Tensor cost);

void matchcost_grad(int b, int n, int m, const at::Tensor grad_cost,
                    const at::Tensor xyz1, const at::Tensor xyz2,
                    const at::Tensor match, const at::Tensor grad1,
                    const at::Tensor grad2);

#endif
