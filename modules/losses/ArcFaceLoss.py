import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# https://arxiv.org/pdf/1801.07698.pdf

class ArcFaceLoss(nn.Module):
	def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
		super(ArcFaceLoss, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_normal_(self.weight)
		self.easy_margin = easy_margin
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = torch.tensor(math.cos(math.pi - m))
		self.mm = torch.tensor(math.sin(math.pi - m) * m)

	def forward(self, embbedings, labels):
		cos_th = F.linear(embbedings, F.normalize(self.weight))
		cos_th = cos_th.clamp(-1, 1)
		sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
		cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
		if self.easy_margin:
			cos_th_m = torch.where(cos_th > 0, cos_th_m, cos_th)
		else:
			cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)
		cond = (cos_th - self.th) <= 0
		cos_th_m[cond] = (cos_th - self.mm)[cond]
		if labels.dim() == 1:
			labels = labels.unsqueeze(-1)
		onehot = torch.zeros(cos_th.size()).cuda()
		labels = labels.type(torch.LongTensor).cuda()
		onehot.scatter_(1, labels, 1.0)
		outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
		outputs = outputs * self.s
		return outputs
