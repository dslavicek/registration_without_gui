import os
import skimage.io
import numpy as np
import torch
import matplotlib.pyplot as plt


def from_path_to_path(path1, path2, output_path):
    im1 = skimage.io.imread(path1, as_gray=True)
    im2 = skimage.io.imread(path2, as_gray=True)
    im_composed = np.stack((im1, im2, im1), axis=2)
    skimage.io.imsave(output_path, im_composed)

# requires grayscale tensors
def from_tensor_to_tensor(purple_tens, green_tens, n=0):
    result = torch.cat((purple_tens[n,0,:,:], green_tens[n,0,:,:]), dim=1)
    tensor_image = result[0].permute(1, 2, 0)
    plt.imshow(tensor_image)
