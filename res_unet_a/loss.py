import torch 
from torch import nn
import torch.functional as F


class TanimotoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        y_pred_softmax = F.softmax(y_pred, 1)

        