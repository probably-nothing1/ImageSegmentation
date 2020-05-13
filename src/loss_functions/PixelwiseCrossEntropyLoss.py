import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelwiseCrossEntropyLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, pixel_probabilities, ground_truth):
    return F.binary_cross_entropy(pixel_probabilities, ground_truth)