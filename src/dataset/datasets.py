import os
import numpy as np
import random
from PIL import Image

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.functional as tvtf

from .constants import IMAGE_MEAN, IMAGE_STD


class TensorDatasetWithTransformations(TensorDataset):
    def __init__(self, *tensors, use_transforms=True):
        super().__init__(*tensors)
        self.use_transforms = use_transforms

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        image = tvtf.to_pil_image(image)
        if self.use_transforms:
            image = tvtf.adjust_brightness(image, 0.65 + torch.rand(1) / 1.3)
            image = tvtf.adjust_contrast(image, 0.75 + torch.rand(1) / 2.0)
            image = tvtf.adjust_hue(image, (torch.rand(1) -0.5) / 25.0)
            image = tvtf.adjust_saturation(image, 0.75 + torch.rand(1) / 2.0)
        image = tvtf.to_tensor(image)
        image = tvtf.normalize(image, IMAGE_MEAN, IMAGE_STD)
        return image, mask

def get_train_dataset(folderpath, dataset_version='flip', use_transforms=True):
    if dataset_version == 'single':
        train_images = np.load(os.path.join(folderpath, 'train_images.npy'))
        train_masks = np.load(os.path.join(folderpath, 'train_masks.npy'))
    elif dataset_version == 'flip':
        train_images = np.load(os.path.join(folderpath, 'train_images_with_flip.npy'))
        train_masks = np.load(os.path.join(folderpath, 'train_masks_with_flip.npy'))
    elif dataset_version == 'all':
        train_images = np.load(os.path.join(folderpath, 'train_images_with_flip_and_rots.npy'))
        train_masks = np.load(os.path.join(folderpath, 'train_masks_with_flip_and_rots.npy'))
    else:
        assert False

    train_images_tensor = torch.from_numpy(train_images)
    train_masks_tensor = torch.from_numpy(train_masks)

    train_images_tensor = torch.einsum('bhwc->bchw', train_images_tensor)
    train_masks_tensor = torch.einsum('bhwc->bchw', train_masks_tensor)

    return TensorDatasetWithTransformations(train_images_tensor, train_masks_tensor, use_transforms=use_transforms)

def get_test_dataset(folderpath, dataset_version='flip', use_transforms=True):
    if dataset_version == 'single':
        test_images = np.load(os.path.join(folderpath, 'test_images.npy'))
        test_masks = np.load(os.path.join(folderpath, 'test_masks.npy'))
    elif dataset_version == 'flip':
        test_images = np.load(os.path.join(folderpath, 'test_images_with_flip.npy'))
        test_masks = np.load(os.path.join(folderpath, 'test_masks_with_flip.npy'))
    elif dataset_version == 'all':
        test_images = np.load(os.path.join(folderpath, 'test_images_with_flip_and_rots.npy'))
        test_masks = np.load(os.path.join(folderpath, 'test_masks_with_flip_and_rots.npy'))
    else:
        assert False

    test_images_tensor = torch.from_numpy(test_images)
    test_masks_tensor = torch.from_numpy(test_masks)

    test_images_tensor = torch.einsum('bhwc->bchw', test_images_tensor)
    test_masks_tensor = torch.einsum('bhwc->bchw', test_masks_tensor)

    return TensorDatasetWithTransformations(test_images_tensor, test_masks_tensor, use_transforms=use_transforms)

def get_train_dataloader(folderpath, batch_size, dataset_version, use_transforms=True, shuffle=False):
    dataset = get_train_dataset(folderpath, dataset_version, use_transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=11)

def get_test_dataloader(folderpath, batch_size, dataset_version, use_transforms=True, shuffle=False):
    dataset = get_test_dataset(folderpath, dataset_version, use_transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=11)
