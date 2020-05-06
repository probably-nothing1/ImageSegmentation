import torch

def compute_IoU(model, dataloader, device):
    total_iou = 0
    num_images = len(dataloader.dataset)
    model.eval()
    for i, (image, true_mask) in enumerate(dataloader):
        image, true_mask = image.to(device), true_mask.to(device)
        predicted_mask = model(image)
        predicted_mask[predicted_mask > 0.5] = 1.0
        iou = compute_IoU_batch(true_mask, predicted_mask)
        total_iou += iou.item()

    model.train()
    return total_iou / num_images

def compute_IoU_batch(true_mask, predicted_mask):
    union = true_mask + predicted_mask
    union[union > 1.0] = 1.0
    union = union.sum(dim=(1,2,3))

    intersection = true_mask * predicted_mask
    intersection = intersection.sum(dim=(1,2,3))

    iou = intersection / union
    return iou.sum()
