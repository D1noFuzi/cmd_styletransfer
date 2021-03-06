import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self, eps=0):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()

    def __call__(self, prediction, target):
        #         l = prediction - target
        #         l = l**2
        #         return torch.sqrt(torch.mean(l))
        return F.mse_loss(prediction, target)


class StyleLoss(nn.Module):

    def __init__(self, target, axis=(0, 2, 3), k=5, weights=(1, 1, 1, 1, 1)):
        super(StyleLoss, self).__init__()
        assert k == len(weights)
        self.target = target
        self.k = k
        self.weights = weights
        self.axis = axis
        self.rmse = RMSELoss()
        self.c1_y, self.m_ys = self._init_targets()

    def _init_targets(self):
        c1_y = self.target.mean(dim=self.axis).view(1, -1, 1, 1)
        m_ys = list()
        for i in range(2, self.k + 1):
            # watch out: zeroth element is pow 2, first is pow 3...
            m_ys.append((self.target - c1_y).pow(i).mean(dim=self.axis))
        return c1_y, m_ys

    def __call__(self, x):
        c1_x = x.mean(dim=self.axis).view(1, -1, 1, 1)
        loss = self.weights[0] * self.rmse(c1_x, self.c1_y)
        for i in range(2, self.k + 1):
            m_x = (x - c1_x).pow(i).mean(dim=self.axis)
            loss = loss + self.weights[i - 1] * self.rmse(m_x, self.m_ys[i - 2])
        return loss
