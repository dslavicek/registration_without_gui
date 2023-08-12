import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import numpy as np
from matrix_operations import *
from utils import *



def pyramid_registration(ref, sample, max_iters=30, mu=0.02, scale_factors=None,
                         datatype=torch.float32, verbose=False,
                         mask_ref=None,
                         mask_sample=None,
                         enable_translation=True, enable_rotation=True,
                         enable_scaling=True, enable_shear=True, used_device='cpu', loss_fcn='mse'):
    # this function performs registration of two 4D pytorch tensors
    # inputs - reference tensor, tensor of moving images, max. number of iterations, size of registration step,
    # datatype, verbose mode
    # output - dictionary containing tensor of registered images, shifts in x,y, rotations, transformation matrices,
    # loss function

    batch_size = ref.shape[0]
    channels = ref.shape[1]
    height = ref.shape[2]
    width = ref.shape[3]

    assert loss_fcn in ['cov', 'mse'], f"Error: Unknown loss function '{loss_fcn}'. Use 'mse'(default) or 'cov'."

    if mask_ref is None:
        mask_ref = torch.ones(ref.shape, dtype=datatype, device=used_device)
    elif len(mask_ref.shape) == 2:
        mask_ref = torch.tensor(mask_ref, dtype=datatype).unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1).to(used_device)
    if mask_sample is None:
        mask_sample = torch.ones(sample.shape, dtype=datatype, device=used_device)
    elif len(mask_sample.shape) == 2:
        mask_sample = torch.tensor(mask_sample, dtype=datatype).unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1).to(used_device)
        if verbose:
            print(f"Reshaped sample mask, shape: {mask_sample.shape}")

    if scale_factors is None:
        scale_factors = [0.125, 0.25, 0.5, 1]
    if verbose:
        print("Begining registration")
    if ref.shape != sample.shape:
        print("ERROR: reference tensor and sample tensor must have same dimensions")
        print("refrerence dimensions: " + str(ref.shape))
        print("sample dimensions: " + str(sample.shape))
        return 1

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
    # tensor of identity matrices for later use
    I = torch.eye(3, device=used_device).repeat((batch_size, 1, 1))
    ref = ref.to(used_device)
    sample = sample.to(used_device)

    for sf in scale_factors:
        ref_rescaled = rescale_images(ref, sf)
        sample_rescaled = rescale_images(sample, sf)
        mask_ref_rescaled = rescale_images(mask_ref, sf)
        mask_sample_rescaled = rescale_images(mask_sample, sf)
        if verbose:
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
                mask_sample_transf = F.grid_sample(mask_sample_rescaled, grid, padding_mode="zeros",
                                                   align_corners=False)

            # calculate loss function
            if loss_fcn == 'mse':
                loss = mse_with_masks(ref_rescaled, registered_tens, mask_ref_rescaled, mask_sample_transf)
            if loss_fcn == 'cov':
                loss = -covariance(ref_rescaled, registered_tens, mask_ref_rescaled, mask_sample_transf)

            losses.append(loss.detach().cpu().numpy())
            # termination condition
            if len(losses) > 2:
                if max(losses[-3:-1]) <= losses[-1]:
                    if verbose:
                        print(f"Registration at scale {sf} terminated after {i+1} iterations")
                    break

            # adjust transformation parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            plt.plot(losses)

    if verbose:
        plt.title(loss_fcn)
        plt.xlabel("iterations")
        plt.ylabel(loss_fcn)
        plt.legend(scale_factors)
        plt.show()

    if verbose:
        print("Registration complete.")
        print("batch size: ", batch_size, "height: ", height, "width: ", width)

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

