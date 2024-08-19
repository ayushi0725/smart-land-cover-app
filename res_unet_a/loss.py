import torch 
from torch import nn
import torch.functional as F


class TanimotoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        """
        Input shapes
        y_pred: (b, c, h, w)
        y: (b, h, w)
        """
        y_pred_softmax = F.softmax(y_pred, 1)

        y_pred_squared = torch.square(y_pred_softmax)
        y_squared = torch.square(y)
        squared_sum = torch.sum(y_pred_squared, y_squared)
        
        product = torch.prod(y_pred_softmax, y)
        product_sum = torch.sum
