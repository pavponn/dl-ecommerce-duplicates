import torch.nn as nn
import torch.nn.functional as F
import timm
from modules.losses.CurricularFace import CurricularFace

class DenseNetCurricularLoss(nn.Module):
	def __init__(self, channel_size, out_feature, dropout=0.5, backbone='densenet121', pretrained=True, s=10.0, m=0.5, easy_margin=False):
		super(DenseNetCurricularLoss, self).__init__()
		self.backbone = timm.create_model(backbone, pretrained=pretrained)
		self.channel_size = channel_size
		self.out_feature = out_feature
		self.in_features = self.backbone.classifier.in_features
		self.margin = CurricularFace(in_features=self.channel_size, out_features=self.out_feature, s=s, m=m)
		self.bn1 = nn.BatchNorm2d(self.in_features)
		self.dropout = nn.Dropout2d(dropout, inplace=True)
		self.fc1 = nn.Linear(self.in_features * 16 * 16 , self.channel_size)
		self.bn2 = nn.BatchNorm1d(self.channel_size)

	def forward(self, x, labels=None):
		features = self.backbone.features(x)
		features = self.bn1(features)
		features = self.dropout(features)
		features = features.view(features.size(0), -1)
		features = self.fc1(features)
		features = self.bn2(features)
		features = F.normalize(features)
		if labels is not None:
			return self.margin(features, labels)
		return features