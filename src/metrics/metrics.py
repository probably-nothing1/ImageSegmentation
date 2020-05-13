import torch

def compute_metrics(model, dataloader, device):
    model.eval()
    total_iou = 0
    total_accuracy = 0
    samples_seen = 0
    for image, true_mask in dataloader:
        image, true_mask = image.to(device), true_mask.to(device)
        pixel_probabilities = model(image)

        iou_batch = compute_IoU_batch(pixel_probabilities, true_mask)
        total_iou += iou_batch.item()

        accuracy_batch = compute_pixel_accuracy_batch(pixel_probabilities, true_mask)
        total_accuracy += accuracy_batch.item()

        samples_seen += len(image)

    iou = total_iou / samples_seen
    accuracy = total_accuracy / samples_seen
    model.train()
    return accuracy, iou

def compute_IoU_batch(pixel_probabilities, true_mask):
    pixel_probabilities[pixel_probabilities > 0.5] = 1.0
    pixel_probabilities[pixel_probabilities <= 0.5] = 0.0

    union = true_mask + pixel_probabilities
    union = torch.clamp(union, max=1.0)
    union = union.sum(dim=(1,2,3))

    intersection = true_mask * pixel_probabilities
    intersection = intersection.sum(dim=(1,2,3))

    iou = intersection / union
    return iou.sum()

def compute_pixel_accuracy_batch(pixel_probabilities, true_mask):
    h, w = true_mask.shape[-2:]
    pixel_probabilities[pixel_probabilities > 0.5] = 1.0
    pixel_probabilities[pixel_probabilities <= 0.5] = 0.0

    true_positive = (true_mask * pixel_probabilities).sum(dim=(1,2,3))
    true_negative = ((1 - true_mask) * (1 - pixel_probabilities)).sum(dim=(1,2,3))


    accuracy = (true_positive + true_negative) / (h * w)
    return accuracy.sum()