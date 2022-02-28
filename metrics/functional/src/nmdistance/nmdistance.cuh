#ifndef _NMDISTANCE_CUH
#define _NMDISTANCE_CUH

void nmdistance(int b, int n, int m, at::Tensor xyz1, at::Tensor xyz2,
                at::Tensor dist1, at::Tensor dist2, at::Tensor idx1,
                at::Tensor idx2);

void nmdistance_grad(int b, int n, int m, at::Tensor xyz1, at::Tensor xyz2,
                     at::Tensor gradxyz1, at::Tensor gradxyz2,
                     at::Tensor graddist1, at::Tensor graddist2,
                     at::Tensor idx1, at::Tensor idx2);

#endif
