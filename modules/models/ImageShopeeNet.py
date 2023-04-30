from torch import nn
import torchvision.models as models
import torch.nn.functional as F


class ImageShopeeNet(nn.Module):

    def __init__(self, freeze_layers=0, dropout=0.1, fc_dim=512, model_name='resnet18'):

        super(ImageShopeeNet, self).__init__()
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # 61 layers
            final_out_features = model.fc.out_features
        else:
            # 363 layers
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            #
            final_out_features = model.classifier.out_features
        self.model = model
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(final_out_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self.relu = nn.ReLU()
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, img):
        out = self.model(img)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.bn(out)
        return out
