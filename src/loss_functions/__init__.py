from .DiceLoss import DiceLoss
from .IoULoss import IoULoss
from .Tversky import Tversky
from .FocalLoss import FocalLoss
from .PixelwiseCrossEntropyLoss import PixelwiseCrossEntropyLoss

# https://neptune.ai/blog/image-segmentation-in-2020
# https://github.com/JunMa11/SegLoss

def dispatch_loss_function(args):
  if args.loss_function == 'Dice':
    return DiceLoss()
  elif args.loss_function == 'PixelwiseCrossEntropy':
    return PixelwiseCrossEntropyLoss()
  elif args.loss_function == 'Focal':
    return FocalLoss(args.focal_gamma)
  elif args.loss_function == 'IoU':
    return IoULoss()
  elif args.loss_function == 'Lovasz-Softmax':
    raise NotImplementedError('Lovasz-Softmax loss not implemented yet')
    return IoULoss()
  elif args.loss_function == 'Tversky':
    return Tversky(args.tversky_beta)
  else:
    raise ValueError(f'Unrecognized loss function: "{args.loss_function}"')

def compute_loss(model, dataloader, loss_function, device, num=20):
    total_loss = 0
    model.eval()

    for i, (image, true_mask) in enumerate(dataloader):
        if i > num:
            break
        image, true_mask = image.to(device), true_mask.to(device)
        predicted_mask = model(image)
        loss = loss_function(predicted_mask, true_mask)
        total_loss += loss.item()

    model.train()
    return total_loss / num