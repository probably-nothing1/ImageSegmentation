import torch

def identity(tensor):
    return tensor

def flip(tensor):
    return torch.flip(tensor, dims=(2,))

def rot90(tensor):
    return torch.rot90(tensor, k=1, dims=(1, 2))

def rot270(tensor):
    return torch.rot90(tensor, k=3, dims=(1, 2))

def rot90_flip(tensor):
    return flip(rot90(tensor))

def rot270_flip(tensor):
    return flip(rot270(tensor))


def apply_inverse_transforms(augmentations, dataset_version='flip'):
    """
    takes input of shape [6,C,H,W] where 6 is nuber of augmentations
    outputs tensor of shape [6,C,H,W] where #A is nuber of augmentations
    """
    assert len(augmentations.shape) == 4, 'apply_transform() should take one tensor of shape [#A, C, H, W] where 6 number of augmentations'
    if dataset_version == 'single':
        assert augmentations.shape[0] == 1
        transforms = [identity]
        inverse_transformed_tensors = [tf(tensor) for tensor, tf in zip(augmentations, transforms)]
    elif dataset_version == 'flip':
        assert augmentations.shape[0] == 2
        transforms = [identity, flip]
        inverse_transformed_tensors = [tf(tensor) for tensor, tf in zip(augmentations, transforms)]
    elif dataset_version == 'all':
        assert augmentations.shape[0] == 6
        transforms = [identity, rot270, rot90, flip, rot270_flip, rot90_flip]
        inverse_transformed_tensors = [tf(tensor) for tensor, tf in zip(augmentations[:4], transforms)]

    return torch.stack(inverse_transformed_tensors)