import torch
from dataset.augmentations import apply_inverse_transforms

def compute_metrics(model, dataloader, transforms, device, dataset_version='flip'):
    if dataset_version == 'single':
        assert dataloader.batch_size == 1, 'Dataloader for evaluation should operate on batch size equal to 1'
    elif dataset_version == 'flip':
        assert dataloader.batch_size == 2, 'Dataloader for evaluation should operate on batch size equal to 2'
    elif dataset_version == 'all':
        assert dataloader.batch_size == 6, 'Dataloader for evaluation should operate on batch size equal to 6'
    else:
        assert False

    print('Start evaluating')
    model.eval()
    total_iou = 0
    total_accuracy = 0
    for true_sample_count, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        pixel_probabilities = model(images)
        pixel_probabilities = apply_inverse_transforms(pixel_probabilities, dataset_version=dataset_version)
        pixel_probability = pixel_probabilities.mean(dim=0, keepdim=True)
        iou_batch = compute_IoU_batch(pixel_probability, masks[:1])
        total_iou += iou_batch.item()

        accuracy_batch = compute_pixel_accuracy_batch(pixel_probability, masks[:1])
        total_accuracy += accuracy_batch.item()

    iou = total_iou / true_sample_count
    accuracy = total_accuracy / true_sample_count
    model.train()
    print('Done evaluating')
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