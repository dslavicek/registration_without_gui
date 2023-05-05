from registration import rigid_registration, affine_registration
from SampleCreator import SampleCreator
import input_output
import pandas as pd
import numpy as np
import torch

path_grayscale_folder = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA/xrays_resampled2"
path_grayscale_single = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA/xrays_resampled2/IM-0003-0001.jpeg"
path_rgb_folder = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/kocka"
path_rgb_single = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/kocka/cat.jpg"

gf = input_output.from_folder_to_tensor(path_grayscale_folder)[0:3]
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

sc_rgb = SampleCreator(rf, from_path=False, verbose=True)
sc_rgb.x_shifts = [0.33, -0.1]
sc_rgb.y_shifts = [0.1, 0]
sc_rgb.rotations_deg = [7, 0]
# sc_rgb.gen_rand_transf_params()
rgb_transf = sc_rgb.generate_samples()
print(sc_rgb.batch_size)
rgb_ref = rf

sc_gf = SampleCreator(gf, from_path=False, verbose=True)
sc_gf.gen_rand_transf_params()
gf_transf = sc_gf.generate_samples()
gf_ref = gf[0:3]
print(f"GF shape: {gf.shape}")

sc_gs = SampleCreator(gs, from_path=False, verbose=True)
sc_gs.gen_rand_transf_params()
gs_transf = sc_gs.generate_samples()
gs_ref = gs
print(f"GS shape: {gs.shape}")

sc_rs = SampleCreator(rs, from_path=False, verbose=True)
sc_rs.gen_rand_transf_params()
rs_transf = sc_rs.generate_samples()
rs_ref = rs
print(f"GS shape: {rs.shape}")
# rgb_ref = rf.repeat(torch.Size([sc_rgb.batch_size, 1, 1, 1]))

reg_res_rgb = affine_registration(rgb_ref, rgb_transf, verbose=True)

print(reg_res_rgb["transformation_matrices"])
print(reg_res_rgb["transformation_matrices"].shape)
reg_res_gf = affine_registration(gf_ref, gf_transf, verbose=True)
reg_res_gs = affine_registration(gs_ref, gs_transf, verbose=True)
reg_res_rs = affine_registration(rs_ref, rs_transf, verbose=True)


input_output.display_nth_image_from_tensor(rgb_ref)
input_output.display_nth_image_from_tensor(reg_res_rgb["registered_tens"])

input_output.display_nth_image_from_tensor(gf_transf)
input_output.display_nth_image_from_tensor(reg_res_gf["registered_tens"])

input_output.display_nth_image_from_tensor(gs_transf)
input_output.display_nth_image_from_tensor(reg_res_gs["registered_tens"])

input_output.display_nth_image_from_tensor(rs_transf)
input_output.display_nth_image_from_tensor(reg_res_rs["registered_tens"])



