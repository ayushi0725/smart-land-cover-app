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
        y: (b, c, h, w)
        """
        p, l = F.softmax(y_pred, 1), y
        epsilon = 1e-7

        V = torch.sum(l, dim=[1, 2])
        w = V ** -2

        p2 = p ** 2
        l2 = l ** 2
        
        sum_prod_p_l = torch.sum(p * l, dim=[1, 2])
        numerator = torch.sum(w * sum_prod_p_l)

        sum_p2_l2 = torch.sum(p2 + l2, dim=[1, 2])
        sum_p2_l2_minus_p_l = sum_p2_l2 - sum_prod_p_l
        denominator = torch.sum(w * sum_p2_l2_minus_p_l)

        loss = numerator / (denominator + epsilon)

        return loss
    

class ComplementedTanimotoLoss(nn.Module):
    def __init__(self):
        self.loss_func = TanimotoLoss()

    def forward(self, y_pred, y):
        loss = self.loss_func(y_pred, y)
        loss_complement = self.loss_func(1 - y_pred, 1 - y)

        return (loss + loss_complement) / 2


if __name__ == '__main__':
    label = torch.randn([4, 5, 5])
    V = torch.sum(label, dim=[1, 2])
    V_mean = torch.mean(V, dim=0)
    print(V.shape)
    print(V_mean.shape)