import torch

def compute_IoU(model, dataloader, device):
    model.eval()
    total_iou = 0
    num_images = len(dataloader.dataset)
    for i, (image, true_mask) in enumerate(dataloader):
        image, true_mask = image.to(device), true_mask.to(device)
        pixel_probabilities = model(image)
        iou = compute_IoU_batch(true_mask, pixel_probabilities)
        total_iou += iou.item()

    model.train()
    return total_iou / num_images

def compute_IoU_batch(true_mask, pixel_probabilities):
    pixel_probabilities[pixel_probabilities > 0.5] = 1.0
    pixel_probabilities[pixel_probabilities <= 0.5] = 0.0

    union = true_mask + pixel_probabilities
    union = torch.clamp(union, max=1.0)
    union = union.sum(dim=(1,2,3))

    intersection = true_mask * pixel_probabilities
    intersection = intersection.sum(dim=(1,2,3))

    iou = intersection / union
    return iou.sum()
