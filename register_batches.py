from pyramid_registration2 import pyramid_registration
from utils import *
import torch
from os import listdir


def register_batches(input_dir, max_iters=30, mu=0.02, batch_size=8, loss='mse', save=False, verbose=False):
    files = listdir(input_dir)
    files.sort()
    transf_mats = torch.empty(len(files) // 2, 2, 3)
    for i in range(0, len(files), batch_size * 2):
        references = files[i:i + batch_size * 2:2]
        samples = files[i + 1:i + batch_size * 2 + 1:2]
        ref_tens = from_list_of_files_to_tensor(references, input_dir)
        sam_tens = from_list_of_files_to_tensor(samples, input_dir)
        results = pyramid_registration(ref_tens, sam_tens, mu=mu, loss_fcn=loss, max_iters=max_iters)
        transf_mats[i // 2:i // 2 + batch_size, :, :] = results["transformation_matrices"]
        if verbose:
            print(f"{i // 2 + batch_size} pairs of images registered, {len(files) // 2 - i // 2 - batch_size} remaining")
    return transf_mats
