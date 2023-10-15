import torch
from .losses import *


class CombinedLoss(nn.Module):
    def __init__(self, loss_classes, loss_weights):
        super(CombinedLoss, self).__init__()
        self.loss_classes = loss_classes
        self.loss_weights = loss_weights

    def forward(self, prediction, target):
        loss = 0
        for lc, lw in zip(self.loss_classes, self.loss_weights):
            loss += lw * lc(prediction, target)
        return loss

