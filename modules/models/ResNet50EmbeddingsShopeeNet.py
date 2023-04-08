from torch import nn
import torchvision.models as models


class ResNet50EmbeddingsShopeeNet(nn.Module):

    def __init__(self):
        super(ResNet50EmbeddingsShopeeNet, self).__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out
