import numpy as np
import torch
from SampleCreator import SampleCreator
from pyramid_registration import pyramid_registration
from registration import rigid_registration
from metric import tre
import input_output
import matplotlib.pyplot as plt


# function takes matrices in shape B*2*3 and returns them in shape B*3*3
def add_bottom_row(matrices):
    bottom_row = torch.tensor([0, 0, 1.0])
    bottom_row = bottom_row.repeat((matrices.shape[0], 1, 1))
    matrices = torch.cat((matrices, bottom_row), dim=1)
    return matrices


# function takes transformation matrices from registration results and returns their inverse
# this is useful for assessing registration precision, because transformation matrices calculated during registration
# should be as close as possible to transformation matrices used for generating synthetic samples
def make_inverse_mats(matrices):
    full_mats = add_bottom_row(matrices)
    inv = torch.inverse(full_mats)
    result = inv[:, 0:2, :]
    return result


num_samples = 9
source = "C:\\Users\\slavi\\Documents\\SKOLA\\DIPLOMKA2\\xrays_1024x1280"
# source = "C:\\Users\\slavi\\Documents\\SKOLA\\DIPLOMKA\\xrays_resampled2"
reference = input_output.from_folder_to_tensor(source)[0:num_samples]

# TEST TRANSLATION
sc = SampleCreator(reference, verbose=True, from_path=False)
sc.clear_transf_params()
sc.x_shifts = np.linspace(-0.1, 0.1, num_samples)
samples = sc.generate_samples()
ground_truth = sc.transf_matrices
print(ground_truth)

results_pyr = pyramid_registration(reference, samples, mu=0.02, max_iters=25)
results_pyr2 = pyramid_registration(reference, samples, mu=0.01, max_iters=50)
#results_rig = rigid_registration(reference, samples, mu=0.02)

errors_pyr = tre(ground_truth, make_inverse_mats(results_pyr["transformation_matrices"]))
errors_pyr2 = tre(ground_truth, make_inverse_mats(results_pyr2["transformation_matrices"]))
#errors_rig = tre(ground_truth, make_inverse_mats(results_rig["transformation_matrices"]))
input_output.display_nth_image_from_tensor(results_pyr["registered_tens"])
input_output.display_nth_image_from_tensor(results_pyr["registered_tens"], 1)
input_output.display_nth_image_from_tensor(results_pyr["registered_tens"], 2)
input_output.display_nth_image_from_tensor(results_pyr["registered_tens"], 3)
input_output.display_nth_image_from_tensor(results_pyr["registered_tens"], 4)
print(results_pyr["transformation_matrices"])
print(errors_pyr)
print(results_pyr["x_shifts"])
plt.plot(errors_pyr)
plt.plot(errors_pyr2)
# plt.plot(errors_rig)
plt.show()
plt.plot(results_pyr["losses"])
plt.show()

t_mat = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
t_mat = t_mat.repeat((num_samples, 1, 1))

orig_errors = tre(ground_truth, t_mat)
print(orig_errors)



