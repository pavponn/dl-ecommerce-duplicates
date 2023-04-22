import torch
import torch.nn as nn


class ManhattanDistance(nn.Module):

    def __init__(self):
        super(ManhattanDistance, self).__init__()

    def forward(self, x_1, x_2):
        return torch.norm(x_1 - x_2)
