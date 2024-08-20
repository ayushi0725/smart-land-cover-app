import torch 
from torch import nn
import torch.nn.functional as F


class TanimotoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        """
        Input shapes (one-hot encoded)
        y_pred: (b, c, h, w)
        y: (b, c, h, w)
        """
        p, l = y_pred, y
        epsilon = 1e-7

        V = torch.mean(torch.sum(l, dim=[2, 3], dtype=torch.float), dim=0)
        w = V ** -2

        inf = torch.tensor(float('inf'))
        new_weights = torch.where(w == inf, torch.zeros_like(w), w)
        w = torch.where(w == inf, torch.ones_like(w) * torch.max(new_weights), w)

        p2 = p ** 2
        l2 = l ** 2
        
        sum_prod_p_l = torch.sum(p * l, dim=[2, 3])
        numerator = torch.sum(w * sum_prod_p_l)

        sum_p2_l2 = torch.sum(p2 + l2, dim=[2, 3])
        sum_p2_l2_minus_p_l = sum_p2_l2 - sum_prod_p_l
        denominator = torch.sum(w * sum_p2_l2_minus_p_l)

        loss = numerator / (denominator + epsilon)

        return 1 - loss
    

class ComplementedTanimotoLoss(nn.Module):
    def __init__(self):
        self.loss_func = TanimotoLoss()

    def forward(self, y_pred, y):
        y_pred = F.softmax(y_pred, 1)
        loss = self.loss_func(y_pred, y)
        loss_complement = self.loss_func(1 - y_pred, 1 - y)

        return (loss + loss_complement) / 2


if __name__ == '__main__':
    label = torch.randn([4, 5, 5])
    V = torch.sum(label, dim=[1, 2])
    V_mean = torch.mean(V, dim=0)
    print(V.shape)
    print(V_mean.shape)