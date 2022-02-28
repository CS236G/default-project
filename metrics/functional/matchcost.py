import torch

from metrics.functional.backend import _backend


class MatchCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        match = _backend.approxmatch_forward(xyz1, xyz2)
        cost = _backend.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = _backend.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


matchcost = MatchCost.apply
