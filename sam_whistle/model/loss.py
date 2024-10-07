from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice
    

class Charbonnier_loss(nn.Module):
    """L1 Charbonnier Loss."""
    def __init__(self, epsilon=1e-3):
        super(Charbonnier_loss, self).__init__()
        self.eps = epsilon ** 2

    def forward(self, X, Y):
        diff = X - Y
        square_err = diff ** 2
        # Summing over the channel and spatial dimensions (assuming input is N x C x H x W)
        square_err_sum = torch.sum(square_err, dim=(1, 2, 3))
        loss = torch.sqrt(square_err_sum + self.eps)
        return torch.mean(loss)

