import torch.nn.functional as F


def l1_loss(noise_pred, noise):
    return F.l1_loss(noise_pred, noise)
