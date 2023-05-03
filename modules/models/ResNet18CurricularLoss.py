import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from modules.losses.CurricularFace import CurricularFace

class ResNet18CurricularLoss(nn.Module):
	def __init__(self, channel_size, out_feature, dropout=0.5, pretrained=True, s=10.0, m=0.5, easy_margin=False):
		super(ResNet18CurricularLoss, self).__init__()
		self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
		self.channel_size = channel_size
		self.out_feature = out_feature
		self.in_features = self.backbone.fc.in_features
		self.margin = CurricularFace(in_features=self.channel_size, out_features=self.out_feature, s=s, m=m)
		self.dropout = nn.Dropout2d(dropout, inplace=True)
		self.fc1 = nn.Linear(self.backbone.fc.out_features, self.channel_size)
		self.bn2 = nn.BatchNorm1d(self.channel_size)

	def forward(self, x, labels=None):
		features = self.backbone(x)
		features = self.dropout(features)
		features = features.view(features.size(0), -1)
		features = self.fc1(features)
		features = self.bn2(features)
		features = F.normalize(features)
		if labels is not None:
			return self.margin(features, labels)
		return features