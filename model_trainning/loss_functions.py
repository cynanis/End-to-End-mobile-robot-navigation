import torch
import torch.nn as nn


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, output, target):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(output, target) + eps)
        loss_linear = torch.sqrt(criterion(output[:, 0], target[:, 0]) + eps)
        loss_angular = torch.sqrt(criterion(output[:, 1], target[:, 1]) + eps)

        return loss, loss_linear, loss_angular


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        criterion = nn.MSELoss()
        loss = criterion(output, target)

        loss_linear = criterion(output[:, 0], target[:, 0])
        loss_angular = criterion(output[:, 1], target[:, 1])

        return loss, loss_linear, loss_angular


class MAELoss(torch.nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output, target):
        criterion = nn.L1Loss()
        loss = criterion(output, target)

        loss_linear = criterion(output[:, 0], target[:, 0])
        loss_angular = criterion(output[:, 1], target[:, 1])

        return loss, loss_linear, loss_angular
