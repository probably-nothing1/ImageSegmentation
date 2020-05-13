from .DiceLoss import DiceLoss
from .IoULoss import IoULoss
from .PixelwiseCrossEntropyLoss import PixelwiseCrossEntropyLoss

# https://neptune.ai/blog/image-segmentation-in-2020
# https://github.com/JunMa11/SegLoss

def dispatch_loss_function(loss_function):
  if loss_function == 'Dice':
    return DiceLoss()
  elif loss_function == 'pixelwise_cross_entropy':
    return PixelwiseCrossEntropyLoss()
  elif loss_function == 'focal':
    raise NotImplementedError('Focal loss not implemented yet')
  elif loss_function == 'IoU':
    return IoULoss()
  elif loss_function == 'Lovasz-Softmax':
    raise NotImplementedError('Lovasz-Softmax loss not implemented yet')
  else:
    raise ValueError(f'Unrecognized loss function: "{loss_function}"')

def compute_loss(model, dataloader, loss_function, device, num=20):
    total_loss = 0
    model.eval()

    for i, (image, true_mask) in enumerate(dataloader):
        if i > num:
            break
        image, true_mask = image.to(device), true_mask.to(device)
        predicted_mask = model(image)
        loss = loss_function(true_mask, predicted_mask)
        total_loss += loss.item()

    model.train()
    return total_loss / num