import os
import skimage.io
import torch
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd


def from_folder_to_tensor(path, datatype=torch.float32, n=-1):
    # get nuber of file, dimensions and prepare empty output tensor
    logging.debug("executing function from_folder_to_tensor")
    all_files = os.listdir(path)
    if n > 0:
        all_files = all_files[0:n]
    batch_size = len(all_files)
    im = skimage.io.imread(os.path.join(path, all_files[0]))
    height = im.shape[0]
    width = im.shape[1]
    grayscale = len(im.shape) == 2
    if grayscale:
        logging.debug("images in folder are grayscale")
        output_tensor = torch.empty((batch_size, 1, height, width), dtype=datatype)
    else:
        logging.debug("images in folder are rgb")
        output_tensor = torch.empty((batch_size, im.shape[2], height, width), dtype=datatype)

    orig_shape = im.shape
    # transform images to tensor one by one and add them to prepared tensor
    for i, f in enumerate(all_files):
        im = skimage.io.imread(os.path.join(path, f))
        # check that image has same dimensions as first one
        if im.shape != orig_shape:
            logging.error(f"ERROR: image dimensions do not match!\nFirst image:{orig_shape}\nsecond image:{im.shape}")
            return 1
        if grayscale:
            output_tensor[i, 0, :, :] = torch.tensor(im, dtype=datatype) / 255
        else:
            im_tensor = torch.tensor(im, dtype=datatype) / 255
            # output_tensor[i, :, :, :] = im_tensor.view(im_tensor.shape[2], im_tensor.shape[0], im_tensor.shape[1])
            output_tensor[i, :, :, :] = im_tensor.permute(2, 0, 1)
    return output_tensor


def from_image_to_tensor(path, datatype=torch.float32):
    im = skimage.io.imread(path)
    height = im.shape[0]
    width = im.shape[1]
    grayscale = len(im.shape) == 2
    im_tensor = torch.tensor(im, dtype=datatype) / 255

    if grayscale:
        im_tensor = im_tensor.reshape((1, 1, height, width))
    else:
        im_tensor = im_tensor.permute(2, 0, 1)
        im_tensor = im_tensor.reshape((1, im.shape[2], height, width))

    return im_tensor


def save_images_from_tensor(tensor, output_folder, filenames=None):
    num_images = tensor.shape[0]
    if filenames is None:
        filenames = ["im" + str(x) + ".jpg" for x in range(num_images)]
    if tensor.shape[0] != len(filenames):
        print("Warning: tensor size does not correspond to filename count. Using default filenames.")
        filenames = ["im" + str(x) + ".jpg" for x in range(num_images)]
    for i in range(tensor.shape[0]):
        skimage.io.imsave(os.path.join(output_folder, filenames[i]), tensor[i, :, :, :])


def display_nth_image_from_tensor(tensor, n=0):
    if tensor.shape[0] < n:
        print("Error: tensor does not have " + str(n) + " images")
        return 1
    # tensor_image = tensor[n].view(tensor.shape[2], tensor.shape[3], tensor.shape[1]) # CHYBA???
    tensor_image = tensor[n].permute(1, 2, 0)
    plt.imshow(tensor_image)
    plt.show()
    return 0


def make_csv_from_reg_dict(registration_dict, output_path):
    x = registration_dict["x_shifts"]
    y = registration_dict["y_shifts"]
    angle = registration_dict["angles_rad"] * 180 / np.pi
    data = np.stack((x, y, angle), axis=1)
    data = np.transpose(data)
    result = pd.DataFrame(data)
    result.to_csv(output_path, header=["x shift", "y shift", "angle deg"], index=False)