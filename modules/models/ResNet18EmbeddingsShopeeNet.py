from torch import nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet18EmbeddingsShopeeNet(nn.Module):

    def __init__(self, batch_norm=False, freeze_layers=0):

        super(ResNet18EmbeddingsShopeeNet, self).__init__()
        self.batch_norm = batch_norm
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
        self.model = model
        ct = 0
        for child in self.model.children():
            ct += 1
            if ct <= freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False
        self.bn = nn.BatchNorm2d(512)

    def forward(self, img):
        out = self.model(img)
        if self.batch_norm:
            out = self.bn(out)
        return out
