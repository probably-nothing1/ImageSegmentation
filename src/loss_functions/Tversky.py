import torch
import torch.nn as nn
import torch.nn.functional as F

class Tversky(nn.Module):
  """
         2 |A intersect B|
  Dice = -----------------
             |A| + |B|
  """
  def __init__(self, beta=0.5):
    super().__init__()
    self.beta = beta

  def forward(self, pixel_probabilities, ground_truth):
    intersection = (ground_truth * pixel_probabilities).sum(dim=0)
    false_positives = self.beta * ((1 - ground_truth) * pixel_probabilities).sum(dim=0)
    false_negatives = (1 - self.beta) * (ground_truth * (1 - pixel_probabilities)).sum(dim=0)
    tversky_iou = intersection / (intersection + false_positives + false_negatives)
    tversky_iou = tversky_iou.mean()
    return 1 - tversky_iou
