import torch
import torch.nn.functional as F
import numpy as np
from pyramid_registration import translation_mat, scale_mat, shear_mat, rotation_mat
import input_output


# class for generating synthetic samples for registration
class SampleCreator:
    def __init__(self, input_images, verbose=False, from_path=True):
        self.verbose = verbose
        self.datatype = torch.float32
        self.x_shifts = None
        self.y_shifts = None
        self.rotations_deg = None
        self.x_scales = None
        self.y_scales = None
        self.x_shears = None
        self.y_shears = None
        self.used_device = 'cpu'
        if from_path:
            self.input_tensor = input_output.from_folder_to_tensor(input_images)
        else:
            self.input_tensor = input_images
        self.batch_size = self.input_tensor.shape[0]
        self.height = self.input_tensor.shape[2]
        self.width = self.input_tensor.shape[3]
        self.output = None
        self.transf_matrices = None
        self.clear_transf_params()
        if verbose:
            print("Object of class SampleCreator created. Batch size: " + str(self.batch_size) + ", height: " +
                  str(self.height) + ", width:" + str(self.width))

    def check_params_length(self):
        if (len(self.x_shifts) == self.batch_size and len(self.y_shifts) == self.batch_size and
                len(self.rotations_deg) == self.batch_size):
            if self.verbose:
                print("Lengths of transformation parameter arrays are of suitable length.")
            return 0
        print("Error: transformation parameter arrays have wrong lengths.")
        if self.verbose:
            print("x shifts:")
            print(len(self.x_shifts))
            print("y shifts:")
            print(len(self.y_shifts))
            print("rotations:")
            print(len(self.rotations_deg))
            print("batchsize:")
            print(self.batch_size)
        return 1

    def generate_samples(self):
        if self.check_params_length():
            return 1
        output = self.input_tensor.clone()
        t_mats = torch.empty(size=(self.batch_size, 2, 3), dtype=self.datatype)
        x_shift_tens = torch.tensor(self.x_shifts, dtype=self.datatype)
        y_shift_tens = torch.tensor(self.y_shifts, dtype=self.datatype)
        angle_rad_tens = torch.tensor(self.rotations_deg, dtype=self.datatype) / 180 * torch.pi
        x_scale_tens = torch.tensor(self.x_scales, dtype=self.datatype)
        y_scale_tens = torch.tensor(self.y_scales, dtype=self.datatype)
        x_shear_tens = torch.tensor(self.x_shears, dtype=self.datatype)
        y_shear_tens = torch.tensor(self.y_shears, dtype=self.datatype)

        T = translation_mat(x_shift_tens, y_shift_tens, self.datatype, self.used_device)
        R = rotation_mat(angle_rad_tens, self.datatype, self.used_device)
        S = scale_mat(x_scale_tens, y_scale_tens, self.datatype, self.used_device)
        SH = shear_mat(x_shear_tens, y_shear_tens, self.datatype, self.used_device)
        t_mats = T @ R @ S @ SH
        t_mats = t_mats[:, 0:2, :]
        # grid = F.affine_grid(t_mats, [self.batch_size, 1, self.height, self.width])
        grid = F.affine_grid(t_mats, self.input_tensor.shape)
        if self.verbose:
            print("grid shape:")
            print(grid.shape)
            print("output before gird sample shape:")
            print(output.shape)
        output = F.grid_sample(output, grid, padding_mode="zeros")
        if self.verbose:
            print("final output:")
            print(output)
        self.output = output
        self.transf_matrices = t_mats.detach()
        return output

    def get_input_dimensions(self):
        self.batch_size = self.input_tensor.shape[0]
        self.height = self.input_tensor.shape[2]
        self.width = self.input_tensor.shape[3]

    def gen_rand_xshifts(self, min_shift=-0.1, max_shift=0.1):
        self.x_shifts = np.random.uniform(min_shift, max_shift, self.batch_size)

    def gen_rand_yshifts(self, min_shift=-0.1, max_shift=0.1):
        self.y_shifts = np.random.uniform(min_shift, max_shift, self.batch_size)

    def gen_rand_rotations(self, min_shift_deg=-5, max_shift_deg=5):
        self.rotations_deg = np.random.uniform(min_shift_deg, max_shift_deg, self.batch_size)

    def gen_rand_xscales(self, min_scale=0.75, max_scale=1.25):
        self.x_scales = np.random.uniform(min_scale, max_scale, self.batch_size)

    def gen_rand_yscales(self, min_scale=0.75, max_scale=1.25):
        self.y_scales = np.random.uniform(min_scale, max_scale, self.batch_size)

    def gen_rand_xshear(self, min_shear=-0.1, max_shear=0.1):
        self.x_shears = np.random.uniform(min_shear, max_shear, self.batch_size)

    def gen_rand_yshear(self, min_shear=-0.1, max_shear=0.1):
        self.y_shears = np.random.uniform(min_shear, max_shear, self.batch_size)

    def clear_xshifts(self):
        self.x_shifts = np.zeros(self.batch_size)

    def clear_yshifts(self):
        self.y_shifts = np.zeros(self.batch_size)

    def clear_rotations(self):
        self.rotations_deg = np.zeros(self.batch_size)

    def clear_xscales(self):
        self.x_scales = np.ones(self.batch_size)

    def clear_yscales(self):
        self.y_scales = np.ones(self.batch_size)

    def clear_xshear(self):
        self.x_shears = np.zeros(self.batch_size)

    def clear_yshear(self):
        self.y_shears = np.zeros(self.batch_size)

    def gen_rand_transf_params(self):
        self.gen_rand_xshifts()
        self.gen_rand_yshifts()
        self.gen_rand_rotations()

    def clear_transf_params(self):
        self.clear_xshifts()
        self.clear_yshifts()
        self.clear_rotations()
        self.clear_xscales()
        self.clear_yscales()
        self.clear_xshear()
        self.clear_yshear()
