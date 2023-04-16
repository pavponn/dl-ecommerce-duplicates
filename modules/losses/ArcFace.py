import torch
import torch.nn as nn
import math

# https://arxiv.org/pdf/1801.07698.pdf

class ArcFace(nn.Module):
	"""Implementation of Additive Angular Margin Loss:
	Args:
		in_features: size of each input sample
		out_features: size of each output sample
		s: norm of input feature
		m: margin
		easy_margin: if True, uses theta = cos(theta+m) when theta > pi
	"""
	def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
		super(ArcFace, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
		# nn.init.xavier_uniform_(self.weight)
		nn.init.normal_(self.weight, std=0.01)
		self.easy_margin = easy_margin
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - m) * m

	def forward(self, embbedings, label):
		embbedings = l2_norm(embbedings, axis=1)
		weight_norm = l2_norm(self.weight, axis=0)
		cos_theta = torch.mm(embbedings, weight_norm)
		cos_theta = cos_theta.clamp(-1, 1)
		target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)
		sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
		cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
		if self.easy_margin:
			final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
		else:
			final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)
		cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
		output = cos_theta * self.s
		return output

def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output