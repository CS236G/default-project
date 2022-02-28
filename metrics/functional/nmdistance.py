import torch

from metrics.functional.backend import _backend


class NMDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = _backend.nmdistance_forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        gradxyz1, gradxyz2 = _backend.nmdistance_backward(
            xyz1, xyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


nmdistance = NMDistance.apply
