# From: https://github.com/HarshSulakhe/siamesenetworks-pytorch/blob/master/loss.py
import torch
from torch.nn import Module
import torch.nn.functional as F


class ContrastiveLoss(Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    pass

    def forward(self, input1, input2, label):
        euclidean_distance = F.pairwise_distance(input1, input2)
        pos = (1 - label) * torch.pow(euclidean_distance, 2)
        neg = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.mean(pos + neg)
        return loss

    pass


pass
