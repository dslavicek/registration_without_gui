import torch
from torchvision.transforms import Resize
import numpy as np
import logging
import skimage.io
import os
import pandas as pd


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


# target registration error
def tre(t_mat1, t_mat2, points=None):
    # t_mat1, t_mat2 are tensors of transformation matrices with size B x 3 x 3 or B x 2 x 3, where B is batch size
    if t_mat1.shape[0] != t_mat2.shape[0]:
        logging.error("Input dimensions do not match")

    batch_size = t_mat1.shape[0]

    if points is None:
        points = torch.tensor([[[0.5, 0.5, 1], [0.5, -0.5, 1], [-0.5, 0.5, 1], [-0.5, -0.5, 1]]])
        points = points.repeat([batch_size, 1, 1])

    # if matrices are in shape B x 2 x 3, convert them to B x 3 x 3
    if t_mat1.shape[1] == 2:
        padding = torch.zeros((batch_size, 1, 3))
        padding[:, 0, 2] = 1
        t_mat1 = torch.cat([t_mat1, padding], 1)
    if t_mat2.shape[1] == 2:
        padding = torch.zeros((batch_size, 1, 3))
        padding[:, 0, 2] = 1
        t_mat2 = torch.cat([t_mat2, padding], 1)

    t_points1 = torch.matmul(points, torch.transpose(t_mat1, 1, 2))
    t_points2 = torch.matmul(points, torch.transpose(t_mat2, 1, 2))

    t_points1 = t_points1[:, :, 0:2]
    t_points2 = t_points2[:, :, 0:2]

    diff = t_points2 - t_points1
    diff_sq = diff ** 2

    euc_dist = torch.sqrt(diff_sq.sum(dim=2))
    euc_dist_avg = euc_dist.mean(dim=1)

    return euc_dist_avg


# loads all images from folder to tensor of size B x C x H x W
# where B=batch size, C=number of color channels, H=height, W=width
# when given optional parameter n, only first n images are read
def load_images(path, datatype=torch.float32, n=-1):
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


def make_csv_from_reg_dict(registration_dict, output_path):
    x = registration_dict["x_shifts"]
    y = registration_dict["y_shifts"]
    angle = registration_dict["angles_rad"] * 180 / np.pi
    data = np.stack((x, y, angle), axis=1)
    data = np.transpose(data)
    result = pd.DataFrame(data)
    result.to_csv(output_path, header=["x shift", "y shift", "angle deg"], index=False)


# saves images from tensor of shape BxCxHxW to given folder
# if filenames are given as a list of strings, then images are saved under those filenames
def save_images(tensor, output_folder, filenames=None):
    num_images = tensor.shape[0]
    if filenames is None:
        filenames = ["im" + str(x) + ".jpg" for x in range(num_images)]
    if tensor.shape[0] != len(filenames):
        print("Warning: tensor size does not correspond to filename count. Using default filenames.")
        filenames = ["im" + str(x) + ".jpg" for x in range(num_images)]
    for i in range(tensor.shape[0]):
        skimage.io.imsave(os.path.join(output_folder, filenames[i]), tensor[i, :, :, :])


def from_list_of_files_to_tensor(filenames, base_path, datatype=torch.float32):
    im = skimage.io.imread(os.path.join(base_path, filenames[0]))
    height = im.shape[0]
    width = im.shape[1]
    batch_size = len(filenames)
    grayscale = len(im.shape) == 2
    if grayscale:
        channels = 1
    else:
        channels = 3
    output_tensor = torch.empty(batch_size, channels, height, width)
    for i, file in enumerate(filenames):
        im = skimage.io.imread(os.path.join(base_path, file))
        im_tensor = torch.tensor(im, dtype=datatype) / 255

        if grayscale:
            im_tensor = im_tensor.reshape((1, 1, height, width))
        else:
            im_tensor = im_tensor.permute(2, 0, 1)
            im_tensor = im_tensor.reshape((1, im.shape[2], height, width))
        output_tensor[i, :, :, :] = im_tensor

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


def mse_with_masks(ims1, ims2, masks1, masks2):
    diff = (ims1 - ims2) ** 2
    diff = diff * masks1 * masks2
    loss = diff.sum() / (masks1 * masks2).sum()
    return loss


def covariance(ims1, ims2, masks1, masks2):
    avg1 = (ims1 * masks1).mean(dim=(2, 3)) / masks1.mean(dim=(2, 3))
    avg2 = (ims2 * masks2).mean(dim=(2, 3)) / masks2.mean(dim=(2, 3))
    cov = ((ims1 - avg1[:, :, None, None])*(ims2 - avg2[:, :, None, None])).mean()
    return cov


def load_masks(mask_dir, ref_mask_name='mask.png', sam_mask_name='feature_mask.png'):
    ref_mask_np = skimage.io.imread(os.path.join(mask_dir, ref_mask_name))
    sam_mask_np = skimage.io.imread(os.path.join(mask_dir, sam_mask_name))
    return ref_mask_np, sam_mask_np


# def covariance(ims1, ims2, masks1, masks2):
#     pcov_sum = 0  # var for acumulating covariances from color channels
#     for c in range(ims1.shape[1]):  # for each channel in RGB
#         avg1 = (ims1[:, c, :, :] * masks1[:, c, :, :]).mean()
#         avg2 = (ims2[:, c, :, :] * masks2[:, c, :, :]).mean()
#         partial_covariance = ((ims1[:, c, :, :] - avg1) * (ims2[:, c, :, :] - avg2)).mean()
#         pcov_sum = pcov_sum + partial_covariance
#     return pcov_sum / ims1.shape[1]