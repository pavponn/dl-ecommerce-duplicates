from torch import nn
import torchvision.models as models
import torch.nn.functional as F


class DenseNetEmbeddingsShopeeNet(nn.Module):

    def __init__(self, freeze_layers=0):

        super(DenseNetEmbeddingsShopeeNet, self).__init__()
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.model = model
        ct = 0
        for child in self.model.children():
            ct += 1
            if ct <= freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, img):
        out = self.model(img)
        return out
