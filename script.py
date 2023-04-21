from registration import grayscale_registration
from SampleCreator import SampleCreator
import input_output
import pandas as pd
import numpy as np
import torch

path_grayscale_folder = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA/xrays_resampled2"
path_grayscale_single = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA/xrays_resampled2/IM-0003-0001.jpeg"
path_rgb_folder = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/kocka"
path_rgb_single = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/kocka/cat.jpg"

gf = input_output.from_folder_to_tensor(path_grayscale_folder)
rf = input_output.from_folder_to_tensor(path_rgb_folder)
gs = input_output.from_image_to_tensor(path_grayscale_single)
rs = input_output.from_image_to_tensor(path_rgb_single)

# print(gf.shape)
# input_output.display_nth_image_from_tensor(gf, 1)
# print(rf.shape)
# input_output.display_nth_image_from_tensor(rf, 1)
# print(gs.shape)
# input_output.display_nth_image_from_tensor(gs, 1)
# print(rs.shape)
# input_output.display_nth_image_from_tensor(rs, 1)

sc_rgb = SampleCreator(rf, from_path=False)
sc_rgb.x_shifts = [-0.05, 0.05]
sc_rgb.y_shifts = [-0.1, 0.1]
sc_rgb.rotations_deg = [-1, 1]
rgb_transf = sc_rgb.generate_samples()
print(rf.shape)
print(type(rf))
rgb_ref = rf.repeat(torch.Size([3, 1, 1, 1]))
print(type(rgb_ref))
print(rgb_ref.size())
reg_res_rgb = grayscale_registration(rgb_ref, rgb_transf)


def make_csv_from_reg_dict(registration_dict, output_path):
    x = registration_dict["x_shifts"]
    y = registration_dict["y_shifts"]
    angle = registration_dict["angles_rad"] * 180 / np.pi
    data = np.stack((x, y, angle), axis=1)
    data = np.transpose(data)
    result = pd.DataFrame(data)
    result.to_csv(output_path, header=["x shift", "y shift", "angle deg"], index=False)
