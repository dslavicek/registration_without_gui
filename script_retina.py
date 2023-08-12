from pyramid_registration3 import pyramid_registration
from utils import *
import torch
from os import listdir
import sys
BATCH_SIZE = 2

input_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/small_datasets/retina_selection_preprocessed"
# input_dir = "../data/FIRE/part_p"

save_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2"
# save_dir = sys.argv[1]

files = listdir(input_dir)
files.sort()
transf_mats = torch.empty(len(files) // 2, 2, 3)
for i in range(0, len(files), BATCH_SIZE * 2):
    references = files[i:i + BATCH_SIZE * 2:2]
    print(references)
    samples = files[i + 1:i + BATCH_SIZE * 2 + 1:2]
    print(samples)
    ref_tens = from_list_of_files_to_tensor(references, input_dir)
    sam_tens = from_list_of_files_to_tensor(samples, input_dir)
    results = pyramid_registration(ref_tens, sam_tens, mu=0.003, max_iters=30, verbose=True)
    transf_mats[i//2:i//2 + BATCH_SIZE, :, :] = results["transformation_matrices"]
    print(f"{i//2+BATCH_SIZE} pairs of images registered, {len(files) // 2 - i//2 - BATCH_SIZE} remaining")

torch.save(transf_mats, save_dir + "/results_retina_cov.pt")
