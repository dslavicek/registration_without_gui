from pyramid_registration import pyramid_registration
from utils import *
import torch
import sys
# reference_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/ref1"
reference_folder = "../data/ref1"
# sample_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/sam1"
sample_folder = "../data/sam1"
ref_tensor = load_images(reference_folder)
sample_tensor = load_images(sample_folder)
# save_dir = sys.argv[1]
save_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2"

registration_results = pyramid_registration(ref_tensor, sample_tensor)

torch.save(registration_results["transformation_matrices"], save_dir + "/results.pt")




