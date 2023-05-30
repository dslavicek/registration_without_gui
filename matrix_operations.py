import torch


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


def make_inverse_mats(matrices):
    full_mats = add_bottom_row(matrices)
    inv = torch.inverse(full_mats)
    result = inv[:, 0:2, :]
    return result


def add_bottom_row(matrices):
    bottom_row = torch.tensor([0, 0, 1.0])
    bottom_row = bottom_row.repeat((matrices.shape[0], 1, 1))
    matrices = torch.cat((matrices, bottom_row), dim=1)
    return matrices
