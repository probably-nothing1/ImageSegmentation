import torch


class Transformation():
    def __init__(self):
        pass

    def transform(self, imgs, masks):
        raise NotImplementedError

    def inverse_transform(self, imgs, masks):
        raise NotImplementedError

class Identity(Transformation):
    def __init__(self):
        super().__init__()

    def transform(self, *tensors):
        return tensors

    def inverse_transform(self, *tensors):
        return tensors

class HorizontalFlip(Transformation):
    def __init__(self):
        super().__init__()

    def transform(self, *tensors):
        return tuple(torch.flip(tensor, dims=(2,)) for tensor in tensors)

    def inverse_transform(self, *tensors):
        return self.transform(*tensors)


class Rotate(Transformation):
    def __init__(self, degree=90):
        super().__init__()
        assert degree in [90, 270], 'Can only rotate 90 and 270 degrees'
        self.rotate_num_times = int(degree / 90)

    def rotation(self, *tensors, k):
        return tuple(torch.rot90(tensor, k=k, dims=(1,2)) for tensor in tensors)

    def transform(self, *tensors):
        k = self.rotate_num_times
        return self.rotation(*tensors, k=k)

    def inverse_transform(self, *tensors):
        k = 4 - self.rotate_num_times
        return self.rotation(*tensors, k=k)

def create_transforms(transforms):
    return [create_transform(transform) for transform in transforms]

def create_transform(transform):
    if transform == 'flip':
        return HorizontalFlip()
    elif transform == 'rot90':
        return Rotate(90)
    elif transform == 'rot270':
        return Rotate(270)
    else:
        raise ValueError(f'{transform} not recognized')
