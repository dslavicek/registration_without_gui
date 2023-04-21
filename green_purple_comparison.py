import os
import skimage.io
import numpy as np


def from_path_to_path(path1, path2, output_path):
    im1 = skimage.io.imread(path1, as_gray=True)
    im2 = skimage.io.imread(path2, as_gray=True)
    im_composed = np.stack((im1, im2, im1), axis=2)
    skimage.io.imsave(output_path, im_composed)
