import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Implementation of Contrastive Loss from the paper "Dimensionality Reduction by Learning an Invariant Mapping".
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, m: float = 1.0, distance: nn.Module = nn.PairwiseDistance(), device=torch.device("cpu")):
        super(ContrastiveLoss, self).__init__()
        self.m = m
        self.distance = distance
        self.device = device

    def forward(self, x_1, x_2, y):
        x_1 = F.normalize(x_1, p=2, dim=1)
        x_2 = F.normalize(x_2, p=2, dim=1)
        dist = self.distance(x_1, x_2).to(self.device)
        return 0.5 * torch.mean(
            (1 - y) * torch.pow(dist, 2) +
            y * torch.pow(torch.maximum(torch.zeros(dist.size()).to(self.device), self.m - dist), 2)
        )
