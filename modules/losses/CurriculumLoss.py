import torch
import torch.nn as nn
from .ContrastiveLoss import ContrastiveLoss


def LinearSchedule(alpha, increment):
    temp = alpha + increment
    
    if temp >= 1:
        return 1
    else:
        return temp 


class CurriculumLoss(nn.Module):
    """
    Implementation of Curriculum loss that uses cross-entropy and negative log likelihood of the predicted probabilities for the
    target classes. The alpha parameter controls the balance between the two losses and can be adjusted over time.
    
    Based on paper "Curriculum Learning" Bengio 2009
    
    increment: the change in alpha. depends on the alpha schedule which could be linear, quad, exp, etc.
    """

    def __init__(self, alpha: float = 0.5, increment: float = 0.1):
        super(CurriculumLoss, self).__init__()
        self.alpha = alpha
        self.base_loss = ContrastiveLoss
        self.alpha_schedule = LinearSchedule
        self.increment = increment

    def forward(self, x_1, x_2, targets, epoch):
        base_loss = self.base_loss(x_1, x_2, targets)
        prob = torch.softmax(outputs, dim=1)
        prob = torch.gather(prob, 1, targets.view(-1, 1)).squeeze()
        prob_loss = - torch.log(prob)
        loss = ( 1- self.alpha) * base_loss + self.alpha * prob_loss.mean()
        self.alpha = self.alpha_schedule(self.alpha, increment)
        return loss
