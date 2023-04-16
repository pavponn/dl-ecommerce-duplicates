import torch.nn as nn
import torch.nn.functional as F


class CosineDistance(nn.Module):

    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward(self, x_1, x_2):
        return 1 - F.cosine_similarity(x_1, x_2)
