import torch
from torchvision.transforms import Resize
import numpy as np


def generate_hann_mask(height, width, a0=0.5):
    x = torch.linspace(-1, 1, width).repeat(height, 1)
    y = torch.linspace(-1, 1, height).repeat(width, 1).transpose(1, 0)
    dist_map = torch.sqrt(x ** 2 + y ** 2)
    dist_map[dist_map > 1] = 1
    mask = a0 - (1 - a0) * torch.cos(torch.pi * (1 - dist_map))
    return mask


def rescale_images(images, factor):
    if factor == 1:
        return images
    height = images.shape[2]
    width = images.shape[3]
    new_height = np.floor(height * factor).astype(np.int32)
    new_width = np.floor(width * factor).astype(np.int32)
    transform = Resize((new_height, new_width), antialias=True)
    return transform(images)


# function for generating levels
def scales(n):
    if n == 1:
        return [1]
    return [2 ** (-n + 1)] + scales(n - 1)
