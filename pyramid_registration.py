import torch
import torch.nn.functional as F
# from skimage.transform import rescale
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import numpy as np


def pyramid_registration(ref, sample, max_iters=30, mu=0.02, scale_factors=None, datatype=torch.float32, verbose=False,
                         enable_translation=True, enable_rotation=True, enable_scaling=True, enable_shear=True,
                         used_device='cpu'):
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


# def pyramid_registration(ref, sample, max_iters=30, mu=0.02, scale_factors=None, datatype=torch.float32, verbose=False,
#                          enable_translation=True, enable_rotation=True, enable_scaling=True, enable_shear=True):
#     # this function performs registration of two 4D pytorch tensors
#     # inputs - reference tensor, tensor of moving images, max. number of iterations, size of registration step,
#     # datatype, verbose mode
#     # output - dictionary containing tensor of registered images, shifts in x,y, rotations, transformation matrices,
#     # loss function
#     if scale_factors is None:
#         scale_factors = [0.125, 0.25, 0.5, 1]
#     if verbose:
#         print("Begining registration")
#     if ref.shape != sample.shape:
#         print("ERROR: reference tensor and sample tensor must have same dimensions")
#         print("refrerence dimensions: " + str(ref.shape))
#         print("sample dimensions: " + str(sample.shape))
#         return 1
#
#     batch_size = ref.shape[0]
#     height = ref.shape[2]
#     width = ref.shape[3]
#     # registered_tens = None
#
#     x_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
#     y_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
#     angle_rad = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
#
#     x_scale = torch.ones(batch_size, dtype=datatype, requires_grad=True)
#     y_scale = torch.ones(batch_size, dtype=datatype, requires_grad=True)
#
#     shear = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
#     optimizer = torch.optim.Adam([x_shift, y_shift, angle_rad], lr=mu)
#
#     for sf in scale_factors:
#         ref_rescaled = rescale_images(ref, sf)
#         sample_rescaled = rescale_images(sample, sf)
#         mask = torch.ones(sample_rescaled.shape, dtype=datatype)
#         print(f"Scale factor: {sf}")
#         losses = []
#         for i in range(max_iters):
#             I = torch.eye(3).repeat((batch_size, 1, 1))
#             T = translation_mat(x_shift, y_shift, datatype)
#             R = rotation_mat(angle_rad, datatype)
#             S = scale_mat(x_scale, y_scale, datatype)
#             SH = shear_mat(shear, datatype)
#             # t_mat = T @ R @ S @ SH
#             t_mat = I
#             if enable_translation:
#                 t_mat = t_mat @ T
#             if enable_rotation:
#                 t_mat = t_mat @ R
#             if enable_scaling:
#                 t_mat = t_mat @ S
#             if enable_shear:
#                 t_mat = t_mat @ SH
#             t_mat = t_mat[:, 0:2, :]
#             grid = F.affine_grid(t_mat, sample_rescaled.shape)
#             registered_tens = F.grid_sample(sample_rescaled, grid, padding_mode="zeros")
#
#             with torch.no_grad():
#                 mask = F.grid_sample(mask, grid, padding_mode="zeros")
#             # calculate loss function
#             diff = (registered_tens - ref_rescaled) ** 2
#             diff = diff * mask
#             loss = diff.mean()
#
#             losses.append(loss.detach().cpu().numpy())
#             # adjust transformation parameters
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         if verbose:
#             plt.plot(losses)
#             plt.show()
#
#     if verbose:
#         print("Registration complete.")
#         print("batchsize: ", batch_size, "height: ", height, "width: ", width)
#
#     output_dict = {
#         "x_shifts": x_shift.detach().cpu().numpy(),
#         "y_shifts": y_shift.detach().cpu().numpy(),
#         "angles_rad": angle_rad.detach().cpu().numpy(),
#         "losses": losses,
#         "registered_tens": registered_tens.detach().cpu(),
#         "transformation_matrices": t_mat.detach()
#     }
#     return output_dict
#
#
# def translation_mat(x_shift, y_shift, datatype):
#     t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=datatype)
#     t_mat = t_mat.repeat((len(x_shift), 1, 1))
#     t_mat[:, 0, 2] = x_shift
#     t_mat[:, 1, 2] = y_shift
#     return t_mat
#
#
# def scale_mat(scale_x, scale_y, datatype):
#     t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=datatype)
#     t_mat = t_mat.repeat((len(scale_x), 1, 1))
#     t_mat[:, 0, 0] = scale_x
#     t_mat[:, 1, 1] = scale_y
#     return t_mat
#
#
# def rotation_mat(angle_rad, datatype):
#     t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=datatype)
#     t_mat = t_mat.repeat((len(angle_rad), 1, 1))
#     t_mat[:, 0, 0] = torch.cos(angle_rad)
#     t_mat[:, 0, 1] = -torch.sin(angle_rad)
#     t_mat[:, 1, 0] = torch.sin(angle_rad)
#     t_mat[:, 1, 1] = torch.cos(angle_rad)
#     return t_mat
#
#
# def shear_mat(shear, datatype):
#     t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=datatype)
#     t_mat = t_mat.repeat((len(shear), 1, 1))
#     t_mat[:, 0, 1] = shear
#     return t_mat
#
#
# def rescale_images(images, factor):
#     output = []
#     for i in range(images.shape[0]):
#         output.append(rescale(np.array(images[i, :, :, :]), factor))
#     return torch.tensor(output)
#
#
#
# # NOT DIFFERENTIABLE
# def mutual_information(images1, images2, mask):
#     BINS = 32
#     batch_size = images1.shape[0]
#     hist1 = []
#     hist2 = []
#     joint_hist = []
#     mi = torch.zeros(batch_size)
#     for i in range(batch_size):
#         hist1 = torch.histogram(images1[i], BINS, range=(0.0, 1.0), density=True)
#         hist2 = torch.histogram(images2[i], BINS, range=(0.0, 1.0), density=True)
#         joint = torch.stack([images1[i].reshape([-1]), images2[i].reshape([-1])], 1)
#         joint_hist = torch.histogramdd(joint, range=[0., 1., 0., 1.], bins=[BINS, BINS], density=True)
#         h_a = -torch.sum(hist1 * torch.log2(hist1))  # entropy of the first image
#         h_b = -torch.sum(hist2 * torch.log2(hist2))  # entropy of the second image
#         h_ab = -torch.sum(joint_hist * torch.log2(joint_hist))  # joint entropy
#         mi[i] = h_a + h_b - h_ab  # mutual information
#
#     return mi.mean()
