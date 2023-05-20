import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

def pyramid_registration_with_scheduler(ref, sample, max_iters=30, mu=0.02, scale_factors=None, datatype=torch.float32,
                                        verbose=False,
                                        enable_translation=True, enable_rotation=True, enable_scaling=True,
                                        enable_shear=True, used_device='cpu'):
    # this function performs registration of two 4D pytorch tensors
    # inputs - reference tensor, tensor of moving images, max. number of iterations, size of registration step,
    # datatype, verbose mode
    # output - dictionary containing tensor of registered images, shifts in x,y, rotations, transformation matrices,
    # loss function

    if scale_factors is None:
        scale_factors = [0.125, 0.25, 0.5, 1]
    if verbose:
        print("Begining registration")
    if ref.shape != sample.shape:
        print("ERROR: reference tensor and sample tensor must have same dimensions")
        print("refrerence dimensions: " + str(ref.shape))
        print("sample dimensions: " + str(sample.shape))
        return 1

    batch_size = ref.shape[0]
    height = ref.shape[2]
    width = ref.shape[3]

    x_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    y_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    angle_rad = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)

    x_scale = torch.ones(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    y_scale = torch.ones(batch_size, dtype=datatype, requires_grad=True, device=used_device)

    shear_x = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    shear_y = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    parameters_to_optimtize = []
    if enable_translation:
        parameters_to_optimtize.append(x_shift)
        parameters_to_optimtize.append(y_shift)
    if enable_rotation:
        parameters_to_optimtize.append(angle_rad)
    if enable_scaling:
        parameters_to_optimtize.append(x_scale)
        parameters_to_optimtize.append(y_scale)
    if enable_shear:
        parameters_to_optimtize.append(shear_x)
        parameters_to_optimtize.append(shear_y)
    optimizer = torch.optim.Adam(parameters_to_optimtize, lr=mu)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     [int(max_iters * 0.33), int(max_iters * 0.67)], gamma=0.2,
                                                     last_epoch=-1)
    # tensor of identity matrices for later use
    I = torch.eye(3, device=used_device).repeat((batch_size, 1, 1))
    ref = ref.to(used_device)
    sample = sample.to(used_device)

    for sf in scale_factors:
        ref_rescaled = rescale_images(ref, sf)
        sample_rescaled = rescale_images(sample, sf)
        mask = torch.ones(sample_rescaled.shape, dtype=datatype, device=used_device)
        print(f"Scale factor: {sf}")
        losses = []
        for i in range(max_iters):
            t_mat = I.clone()
            if enable_translation:
                T = translation_mat(x_shift, y_shift, datatype, used_device)
                t_mat = t_mat @ T
            if enable_rotation:
                R = rotation_mat(angle_rad, datatype, used_device)
                t_mat = t_mat @ R
            if enable_scaling:
                S = scale_mat(x_scale, y_scale, datatype, used_device)
                t_mat = t_mat @ S
            if enable_shear:
                SH = shear_mat(shear_x, shear_y, datatype, used_device)
                t_mat = t_mat @ SH
            t_mat = t_mat[:, 0:2, :]

            grid = F.affine_grid(t_mat, sample_rescaled.shape, align_corners=False)
            registered_tens = F.grid_sample(sample_rescaled, grid,
                                            padding_mode="zeros", align_corners=False)

            with torch.no_grad():
                mask = F.grid_sample(mask, grid, padding_mode="zeros", align_corners=False)

            # calculate loss function
            diff = (registered_tens - ref_rescaled) ** 2
            diff = diff * mask
            loss = diff.mean()

            losses.append(loss.detach().cpu().numpy())

            # adjust transformation parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if verbose:
            plt.plot(losses)
            plt.show()

    if verbose:
        print("Registration complete.")
        print("batchsize: ", batch_size, "height: ", height, "width: ", width)

    output_dict = {
        "x_shifts": x_shift.detach().cpu().numpy(),
        "y_shifts": y_shift.detach().cpu().numpy(),
        "angles_rad": angle_rad.detach().cpu().numpy(),
        "x_scale": x_scale.detach().cpu().numpy(),
        "y_scale": y_scale.detach().cpu().numpy(),
        "x_shear": shear_x.detach().cpu().numpy(),
        "y_shear": shear_y.detach().cpu().numpy(),
        "losses": losses,
        "registered_tens": registered_tens.detach().cpu(),
        "transformation_matrices": t_mat.detach().cpu()
    }
    return output_dict


def translation_mat(x_shift, y_shift, datatype, used_device):
    t_mat = torch.eye(3, dtype=datatype, device=used_device).repeat((len(x_shift), 1, 1))
    t_mat[:, 0, 2] = x_shift
    t_mat[:, 1, 2] = y_shift
    return t_mat


def scale_mat(scale_x, scale_y, datatype, used_device):
    t_mat = torch.eye(3, dtype=datatype, device=used_device).repeat((len(scale_x), 1, 1))
    t_mat[:, 0, 0] = scale_x
    t_mat[:, 1, 1] = scale_y
    return t_mat


def rotation_mat(angle_rad, datatype, used_device):
    t_mat = torch.eye(3, dtype=datatype, device=used_device).repeat((len(angle_rad), 1, 1))
    t_mat[:, 0, 0] = torch.cos(angle_rad)
    t_mat[:, 0, 1] = -torch.sin(angle_rad)
    t_mat[:, 1, 0] = torch.sin(angle_rad)
    t_mat[:, 1, 1] = torch.cos(angle_rad)
    return t_mat


def shear_mat(shear_x, shear_y, datatype, used_device):
    t_mat = torch.eye(3, dtype=datatype, device=used_device).repeat((len(shear_x), 1, 1))
    t_mat[:, 0, 1] = shear_x
    t_mat[:, 1, 0] = shear_y
    return t_mat


def rescale_images(images, factor):
    if factor == 1:
        return images
    height = images.shape[2]
    width = images.shape[3]
    # new_height = torch.floor(height * factor).to(torch.uint32)
    # new_width = torch.floor(width * factor).to(torch.uint32)
    new_height = np.floor(height * factor).astype(np.int32)
    new_width = np.floor(width * factor).astype(np.int32)
    transform = Resize((new_height, new_width), antialias=True)
    return transform(images)