import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
  def __init__(self, gamma=1.0):
    super().__init__()
    self.gamma = gamma

  def forward(self, pixel_probabilities, ground_truth):
    weights = pixel_probabilities.clone().detach()
    indices = ground_truth == 1.0
    weights[indices] = (1 - weights)[indices]
    weights = weights ** self.gamma
    return F.binary_cross_entropy(pixel_probabilities, ground_truth, weight=weights)