import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
  """
         2 |A intersect B|
  Dice = -----------------
             |A| + |B|
  """
  def __init__(self):
    super().__init__()

  def forward(self, ground_truth, pixel_probabilities):
    intersection = (ground_truth * pixel_probabilities).sum(dim=0)
    count_A = ground_truth.sum(dim=0)
    count_B = pixel_probabilities.sum(dim=0)
    iou = intersection / (count_A + count_B - intersection)
    iou = iou.mean()
    return 1 - iou
