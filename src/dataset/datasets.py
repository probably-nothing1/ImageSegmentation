import os
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from .constants import IMAGE_MEAN, IMAGE_STD
from .augmentations import Identity

class TensorDatasetWithTransformations(TensorDataset):
    def __init__(self, *tensors, transforms=[]):
        super().__init__(*tensors)
        self.transforms = [Identity()] + transforms

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return self.transform(*item)

    def transform(self, *tensors):
        transformation = random.choice(self.transforms)
        return transformation.transform(*tensors)


def normalize(tensor, means, stds):
    '''
    Normalizes tensor to 0 mean unit variance
    Assuming [B, C, H, W] dimensions
    '''
    means = torch.FloatTensor(means)
    stds = torch.FloatTensor(stds)
    return (tensor - means[None, :, None, None]) / stds[None, :, None, None]

def get_train_dataset(folderpath, transforms=[]):
    train_images = np.load(os.path.join(folderpath, 'train_img_preprocessed.npy'))
    train_masks = np.load(os.path.join(folderpath, 'train_mask_preprocessed.npy'))

    train_images_tensor = torch.from_numpy(train_images)
    train_masks_tensor = torch.from_numpy(train_masks)

    train_images_tensor = normalize(train_images_tensor, IMAGE_MEAN, IMAGE_STD)

    return TensorDatasetWithTransformations(train_images_tensor, train_masks_tensor, transforms=transforms)

def get_test_dataset(folderpath, transforms=[]):
    test_images = np.load(os.path.join(folderpath, 'test_img_preprocessed.npy'))
    test_masks = np.load(os.path.join(folderpath, 'test_mask_preprocessed.npy'))

    test_images_tensor = torch.from_numpy(test_images)
    test_masks_tensor = torch.from_numpy(test_masks)

    test_images_tensor = normalize(test_images_tensor, IMAGE_MEAN, IMAGE_STD)

    return TensorDatasetWithTransformations(test_images_tensor, test_masks_tensor, transforms=transforms)

def get_train_dataloader(folderpath, batch_size, transforms=[]):
    dataset = get_train_dataset(folderpath, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=11)

def get_test_dataloader(folderpath, batch_size, transforms=[]):
    dataset = get_test_dataset(folderpath, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=11)
