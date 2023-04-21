import os
import skimage.io
import torch
import matplotlib.pyplot as plt

def save_images_from_tensor(tensor, output_folder, filenames):
    if tensor.shape[0] != len(filenames):
        print("Error: tensor size does not correspond to filename count.")
        return 1
    for i in range(tensor.shape[0]):
        skimage.io.imsave(os.path.join(output_folder, filenames[i]), tensor[i, 0, :, :])
