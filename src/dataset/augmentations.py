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
        return tuple(torch.flip(tensor, dims=(3,)) for tensor in tensors)

    def inverse_transform(self, *tensors):
        return self.transform(tensors)


class Rotate(Transformation):
    def __init__(self, degree=90):
        super().__init__()
        assert degree in [90, 270], 'Can only rotate 90 and 270 degrees'
        self.rotate_num_times = int(degree / 90)

    def rotation(self, *tensors, k):
        return tuple(torch.rot90(tensor, k=k, dims=(2,3)) for tensor in tensors)

    def transform(self, *tensors, k):
        k = self.rotate_num_times
        return self.rotation(tensors, k)

    def inverse_transform(self, *tensors):
        k = 4 - self.rotate_num_times
        return self.rotation(tensors, k)
