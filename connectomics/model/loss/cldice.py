import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import SoftSkeletonize

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, pred, target, weight_mask=None):  # Add weight arg for compatibility
        if self.exclude_background:
            target = target[:, 1:, :, :]
            pred = pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(pred)
        skel_true = self.soft_skeletonize(target)
        tprec = (torch.sum(torch.multiply(skel_pred, target)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1., exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, pred, target, weight_mask=None):  # 🔧 fixed signature
        if self.exclude_background:
            target = target[:, 1:, :, :]
            pred = pred[:, 1:, :, :]

        # soft dice
        intersection = torch.sum(target * pred)
        dice = 1. - (2. * intersection + self.smooth) / (torch.sum(target) + torch.sum(pred) + self.smooth)

        # clDice
        skel_pred = self.soft_skeletonize(pred)
        skel_true = self.soft_skeletonize(target)
        tprec = (torch.sum(skel_pred * target) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)

        # weighted sum
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice
