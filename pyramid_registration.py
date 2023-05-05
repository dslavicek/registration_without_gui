import torch
import torch.nn.functional as F
from skimage.transform import rescale
import matplotlib.pyplot as plt
import numpy as np


def pyramid_registration(ref, sample, max_iters=50, mu=0.02, pyramid_level=3, datatype=torch.float32, verbose=False):
    # this function performs registration of two 4D pytorch tensors
    # inputs - reference tensor, tensor of moving images, max. number of iterations, size of registration step,
    # datatype, verbose mode
    # output - dictionary containing tensor of registered images, shifts in x,y, rotations, transformation matrices,
    # loss function
    if verbose:
        print("Begining registration")
    if ref.shape != sample.shape:
        print("ERROR: reference tensor and sample tensor must have same dimensions")
        print("refrerence dimensions: " + str(ref.shape))
        print("sample dimensions: " + str(sample.shape))
        return 1

    batch_size = ref.shape[0]
    # height = ref.shape[2]
    # width = ref.shape[3]
    # registered_tens = None

    x_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
    y_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
    angle_rad = torch.zeros(batch_size, dtype=datatype, requires_grad=True)

    x_scale = torch.ones(batch_size, dtype=datatype, requires_grad=True)
    y_scale = torch.ones(batch_size, dtype=datatype, requires_grad=True)

    shear = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
    optimizer = torch.optim.Adam([x_shift, y_shift, angle_rad], lr=mu)
    scale_factors = [0.25, 0.5, 1]
    for sf in scale_factors:
        ref_rescaled = rescale_images(ref, sf)
        sample_rescaled = rescale_images(sample, sf)
        mask = torch.ones(sample_rescaled.shape, dtype=datatype)

        losses = []

        for i in range(max_iters):
            T = translation_mat(x_shift, y_shift, datatype)
            R = rotation_mat(angle_rad, datatype)
            S = scale_mat(x_scale, y_scale, datatype)
            SH = shear_mat(shear, datatype)
            t_mat = T @ R @ S @ SH
            t_mat = t_mat[:, 0:2, :]
            grid = F.affine_grid(t_mat, sample_rescaled.shape)
            registered_tens = F.grid_sample(sample_rescaled, grid, padding_mode="zeros")

            with torch.no_grad():
                mask = F.grid_sample(mask, grid, padding_mode="zeros")
            # calculate loss function
            diff = (registered_tens - ref_rescaled) ** 2
            diff = diff * mask
            loss = diff.mean()

            losses.append(loss.detach().cpu().numpy())
            # adjust transformation parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        plt.plot(losses)
        plt.show()

    if verbose:
        print("Registration complete.")
        print("batchsize: ", batch_size, "height: ", height, "width: ", width)

    output_dict = {
        "x_shifts": x_shift.detach().cpu().numpy(),
        "y_shifts": y_shift.detach().cpu().numpy(),
        "angles_rad": angle_rad.detach().cpu().numpy(),
        "losses": losses,
        "registered_tens": registered_tens.detach().cpu(),
        "transformation_matrices": t_mat.detach()
    }
    return output_dict


def one_level_registration(reference, sample, x_shift, y_shift, x_scale, y_scale, angle_rad, shear):
    losses = []
    for i in range(max_iters):
        if verbose and (i == 0 or i % 5 == 4 or i == max_iters - 1):
            print("iteration", i + 1, " out of ", max_iters)
        # prepare transformation matrix
        T = translation_mat(x_shift, y_shift, datatype)
        R = rotation_mat(angle_rad, datatype)
        S = scale_mat(x_scale, y_scale, datatype)
        SH = shear_mat(shear, datatype)
        t_mat = T @ R @ S @ SH
        t_mat = t_mat[:, 0:2, :]

        # transform image
        grid = F.affine_grid(t_mat, sample.shape)
        registered_tens = F.grid_sample(sample, grid, padding_mode="zeros")
        # transform mask
        with torch.no_grad():
            mask = F.grid_sample(mask, grid, padding_mode="zeros")
        # calculate loss function
        diff = (registered_tens - ref) ** 2
        diff = diff * mask
        loss = diff.mean()

        losses.append(loss.detach().cpu().numpy())
        # adjust transformation parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def translation_mat(x_shift, y_shift, datatype):
    t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=datatype)
    t_mat = t_mat.repeat((len(x_shift), 1, 1))
    t_mat[:, 0, 2] = x_shift
    t_mat[:, 1, 2] = y_shift
    return t_mat


def scale_mat(scale_x, scale_y, datatype):
    t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=datatype)
    t_mat = t_mat.repeat((len(scale_x), 1, 1))
    t_mat[:, 0, 0] = scale_x
    t_mat[:, 1, 1] = scale_y
    return t_mat


def rotation_mat(angle_rad, datatype):
    t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=datatype)
    t_mat = t_mat.repeat((len(angle_rad), 1, 1))
    t_mat[:, 0, 0] = torch.cos(angle_rad)
    t_mat[:, 0, 1] = -torch.sin(angle_rad)
    t_mat[:, 1, 0] = torch.sin(angle_rad)
    t_mat[:, 1, 1] = torch.cos(angle_rad)
    return t_mat


def shear_mat(shear, datatype):
    t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=datatype)
    t_mat = t_mat.repeat((len(shear), 1, 1))
    t_mat[:, 0, 1] = shear
    return t_mat


def rescale_images(images, factor):
    output = []
    for i in range(images.shape[0]):
        output.append(rescale(np.array(images[i, :, :, :]), factor))
    return torch.tensor(output)


def mutual_information(images1, images2, mask):
    BINS = 32
    batch_size = images1.shape[0]
    hist1 = []
    hist2 = []
    joint_hist = []
    mi = torch.zeros(batch_size)
    for i in range(batch_size):
        hist1.append(torch.histogram(images1[i], BINS, range=(0.0, 1.0), density=True))
        hist2.append(torch.histogram(images2[i], BINS, range=(0.0, 1.0), density=True))
        joint = torch.stack([images1[i].reshape([-1]), images2[i].reshape([-1])], 1)
        joint_hist.append(torch.histogramdd(joint, range=[0., 1., 0., 1.], bins=[BINS, BINS], density=True))
        h_a = -torch.sum(hist1[-1] * torch.log2(hist1[-1]))  # entropy of the first image
        h_b = -torch.sum(hist2[-1] * torch.log2(hist2[-1]))  # entropy of the second image
        h_ab = -torch.sum(joint_hist[-1] * torch.log2(joint_hist[-1]))  # joint entropy
        mi[i] = h_a + h_b - h_ab  # mutual information

    return torch.tensor(mi).mean()

    pass
