from pyramid_registration3 import pyramid_registration
from utils import *
import torch
import os.path
from os import listdir


def register_batches(input_dir, max_iters=30, mu=0.02, batch_size=8, loss_fcn='mse',
                     masks=None, verbose=False):
    if masks is None:
        masks = os.path.join(input_dir, '..', 'Masks')
    masks_np = load_masks(masks)
    files = listdir(input_dir)
    files.sort()
    transf_mats = torch.empty(len(files) // 2, 2, 3)
    for i in range(0, len(files), batch_size * 2):
        references = files[i:i + batch_size * 2:2]
        samples = files[i + 1:i + batch_size * 2 + 1:2]
        ref_tens = from_list_of_files_to_tensor(references, input_dir)
        sam_tens = from_list_of_files_to_tensor(samples, input_dir)
        results = pyramid_registration(ref_tens, sam_tens, mask_ref=masks_np[0], mask_sample=masks_np[1], mu=mu,
                                       loss_fcn=loss_fcn, max_iters=max_iters, verbose=verbose)
        transf_mats[i // 2:i // 2 + batch_size, :, :] = results["transformation_matrices"]
        # print(f"{i // 2 + batch_size} pairs of images registered, {len(files) // 2 - i // 2 - batch_size} remaining")
    return transf_mats
