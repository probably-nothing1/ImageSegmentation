import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
  """
         2 |A intersect B|
  Dice = -----------------
             |A| + |B|
  """
  def __init__(self):
    super().__init__()

  def forward(self, ground_truth, prediction):
    count_A = ground_truth.sum(dim=0)
    count_B = prediction.sum(dim=0)
    intersection = (ground_truth * prediction).sum(dim=0)
    dice = 2 * intersection / (count_A + count_B)
    return 1 - dice