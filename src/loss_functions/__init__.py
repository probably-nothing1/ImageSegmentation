from DiceLoss import DiceLoss
from PixelwiseCrossEntropyLoss import PixelwiseCrossEntropyLoss

# https://neptune.ai/blog/image-segmentation-in-2020
# https://github.com/JunMa11/SegLoss

def dispatch_loss_function(loss_function):
  if loss_function == 'dice':
    return DiceLoss()
  elif loss_function == 'pixelwise_cross_entropy':
    return PixelwiseCrossEntropyLoss()
  elif loss_function == 'focal':
    raise NotImplementedError('Focal loss not implemented yet')
  elif loss_function == 'IoU':
    raise NotImplementedError('IoU loss not implemented yet')
  elif loss_function == 'Lovasz-Softmax':
    raise NotImplementedError('Lovasz-Softmax loss not implemented yet')
