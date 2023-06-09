import torch
import torch.nn.functional as F


def rigid_registration(ref, sample, max_iters=120, mu=0.02, datatype=torch.float32, verbose=False):
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
    height = ref.shape[2]
    width = ref.shape[3]
    registered_tens = None

    mask = torch.ones(sample.shape, dtype=datatype)

    t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=datatype)
    t_mat = t_mat.repeat((batch_size, 1, 1))

    x_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
    y_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True)
    angle_rad = torch.zeros(batch_size, dtype=datatype, requires_grad=True)

    optimizer = torch.optim.Adam([x_shift, y_shift, angle_rad], lr=mu)
    losses = []
    for i in range(max_iters):
        if verbose and (i == 0 or i % 5 == 4 or i == max_iters - 1):
            print("iteration", i + 1, " out of ", max_iters)
        # prepare transformation matrix
        t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=datatype)
        t_mat = t_mat.repeat((batch_size, 1, 1))
        t_mat[:, 0, 2] = x_shift
        t_mat[:, 1, 2] = y_shift
        t_mat[:, 0, 0] = torch.cos(angle_rad)
        t_mat[:, 0, 1] = -torch.sin(angle_rad)
        t_mat[:, 1, 0] = torch.sin(angle_rad)
        t_mat[:, 1, 1] = torch.cos(angle_rad)
        # transform image
        grid = F.affine_grid(t_mat, sample.shape)
        registered_tens = F.grid_sample(sample, grid, padding_mode="zeros")
        # transform mask
        with torch.no_grad():
            # mask_out = F.grid_sample(mask, grid, padding_mode="zeros")
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


def affine_registration(ref, sample, max_iters=120, mu=0.02, datatype=torch.float32, verbose=False, used_device='cpu'):
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
    height = ref.shape[2]
    width = ref.shape[3]
    registered_tens = None

    ref = ref.to(used_device)
    sample = sample.to(used_device)

    mask = torch.ones(sample.shape, dtype=datatype, device=used_device)

    x_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    y_shift = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    angle_rad = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)

    x_scale = torch.ones(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    y_scale = torch.ones(batch_size, dtype=datatype, requires_grad=True, device=used_device)

    shear_x = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)
    shear_y = torch.zeros(batch_size, dtype=datatype, requires_grad=True, device=used_device)

    optimizer = torch.optim.Adam([x_shift, y_shift, angle_rad, x_scale, y_scale, shear_x, shear_y], lr=mu)
    losses = []
    for i in range(max_iters):
        if verbose and (i == 0 or i % 5 == 4 or i == max_iters - 1):
            print("iteration", i + 1, " out of ", max_iters)
        # prepare transformation matrix
        T = translation_mat(x_shift, y_shift, datatype, used_device)
        R = rotation_mat(angle_rad, datatype, used_device)
        S = scale_mat(x_scale, y_scale, datatype, used_device)
        SH = shear_mat(shear_x, shear_y, datatype, used_device)
        t_mat = T @ R @ S @ SH
        t_mat = t_mat[:, 0:2, :]


        # transform image
        grid = F.affine_grid(t_mat, sample.shape, align_corners=False)
        registered_tens = F.grid_sample(sample, grid, padding_mode="zeros",
                                        align_corners=False)
        # transform mask
        with torch.no_grad():
            mask = F.grid_sample(mask, grid, padding_mode="zeros",
                                 align_corners=False)
        # calculate loss function
        diff = (registered_tens - ref) ** 2
        diff = diff * mask
        loss = diff.mean()

        losses.append(loss.detach().cpu().numpy())
        # adjust transformation parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if verbose:
        print("Registration complete.")
        print("batchsize: ", batch_size, "height: ", height, "width: ", width)

    output_dict = {
        "x_shifts": x_shift.detach().cpu().numpy(),
        "y_shifts": y_shift.detach().cpu().numpy(),
        "angles_rad": angle_rad.detach().cpu().numpy(),
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
