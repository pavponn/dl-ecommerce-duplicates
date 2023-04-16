import torch.nn as nn
import torch


class SiameseNet(nn.Module):
    """
    Given a model uses it to create siamese model/
    """

    def __init__(self, model: nn.Module):
        super(SiameseNet, self).__init__()
        self.model = model

    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, x_1, x_2):
        out_1 = self.forward_once(x_1)
        out_2 = self.forward_once(x_2)
        return out_1, out_2
