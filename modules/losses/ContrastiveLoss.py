import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Implementation of Contrastive Loss from the paper "Dimensionality Reduction by Learning an Invariant Mapping".
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, m: float = 1.0, distance: nn.Module = nn.PairwiseDistance()):
        super(ContrastiveLoss, self).__init__()
        self.m = m
        self.distance = distance

    def forward(self, x_1, x_2, y):
        dist = self.distance(x_1, x_2)
        return 0.5 * torch.mean(
            (1 - y) * torch.pow(dist, 2) +
            y * torch.pow(torch.maximum(torch.zeros(dist.size()), self.m - dist), 2)
        )
