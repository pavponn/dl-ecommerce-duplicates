import torch
import torch.nn as nn
import math

# https://arxiv.org/pdf/2004.00288.pdf

class CurricularFace(nn.Module):
	"""Implementation of Adaptive Curriculum Learning Loss:
	Args:
		in_features: size of each input sample
		out_features: size of each output sample
		s: norm of input feature
		m: margin
	"""
	def __init__(self, in_features, out_features, m=0.5, s=64.):
		super(CurricularFace, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.m = m
		self.s = s
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.threshold = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - m) * m
		self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
		self.register_buffer('t', torch.zeros(1))
		nn.init.normal_(self.weight, std=0.01)

	def forward(self, embbedings, label):
		embbedings = l2_norm(embbedings, axis = 1)
		weight_norm = l2_norm(self.weight, axis = 0)
		cos_theta = torch.mm(embbedings, weight_norm)
		cos_theta = cos_theta.clamp(-1, 1)
		target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

		sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
		cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
		mask = cos_theta > cos_theta_m
		final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

		hard_example = cos_theta[mask]
		with torch.no_grad():
			self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
		cos_theta[mask] = hard_example * (self.t + hard_example)
		cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
		output = cos_theta * self.s
		return output

def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output