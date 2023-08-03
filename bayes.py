# import os
import torch

from bayes_opt import BayesianOptimization, UtilityFunction
from pyramid_registration import pyramid_registration
from utils import load_images
from torch import save as torch_save
from skimage.io import imread
from os import listdir
from pandas import read_csv

reference_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/ref2"
# reference_dir = "../data/ref2"
sample_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/sam2"
# sample_dir = "../data/sam2"
mask_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/mask2"
# mask_dir = "../data/mask2"
save_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2"

ground_truth_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/ground_truth2"
# ground_truth_filenames = listdir(ground_truth_dir)
ground_truth_filenames = [ground_truth_dir + "/" + x for x in listdir(ground_truth_dir)]
ground_truth_filenames.sort()

ref_tensor = load_images(reference_dir)
sample_tensor = load_images(sample_dir)
mask_ref = imread(mask_dir + "/mask.png")
mask_sample = imread(mask_dir + "/feature_mask.png")


# function gets two sets of points and calculates mean distance between corresponding points
# inputs are 2D torch tensors, where first column is x coordinate, second column is y coordinate
def get_mean_distance(points1, points2):
    diff = points1 - points2
    diff_sq = diff ** 2
    dists = torch.sqrt(diff_sq[:, 0] + diff_sq[:, 1])
    return dists.mean()


def register_and_evaluate(iters=30, mu=0.003):
    results = pyramid_registration(ref_tensor, sample_tensor, mask_ref=mask_ref,
                                   mask_sample=mask_sample, mu=mu, max_iters=iters)
    print(results["transformation_matrices"])
    errors = []
    for i in range(ref_tensor.shape[0]):
        # get transformation matrix 3x2 from results
        transf_mat = results["transformation_matrices"][i]
        # add bottom row [0,0,1]
        transf_mat = torch.cat((transf_mat[0:2], torch.tensor([[0, 0, 1.0]])), dim=0)
        # load coordinates of points from ground_truth
        gt = read_csv(ground_truth_filenames[i], sep=' ', names=['x1', 'y1', 'x2', 'y2'])
        # rescale coordinates to <-1,1> from pixels
        point_locations = torch.tensor((gt.iloc[:, 0:2] / 1456 - 1).values, dtype=torch.float32)
        correct_locations = torch.tensor((gt.iloc[:, 2:4] / 1456 - 1).values, dtype=torch.float32)
        # add column with ones for affine transformation
        points_for_transform = torch.cat((point_locations, torch.ones((10, 1))), dim=1)
        # perform affine transformation with calculated matrix
        transformed_points = points_for_transform @ transf_mat.T
        # calculate error
        error = get_mean_distance(transformed_points[:, 0:2], correct_locations)
        errors.append(error)

    return torch.tensor(errors).mean()


def evalueate_no_transform():
    errors = []
    for i in range(ref_tensor.shape[0]):
        # get transformation matrix 3x2 from results
        transf_mat = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        # add bottom row [0,0,1]
        transf_mat = torch.cat((transf_mat[0:2], torch.tensor([[0, 0, 1.0]])), dim=0)
        # load coordinates of points from ground_truth
        gt = read_csv(ground_truth_filenames[i], sep=' ', names=['x1', 'y1', 'x2', 'y2'])
        # rescale coordinates to <-1,1> from pixels
        point_locations = torch.tensor((gt.iloc[:, 0:2] / 1456 - 1).values, dtype=torch.float32)
        correct_locations = torch.tensor((gt.iloc[:, 2:4] / 1456 - 1).values, dtype=torch.float32)
        # add column with ones for affine transformation
        points_for_transform = torch.cat((point_locations, torch.ones((10, 1))), dim=1)
        # perform affine transformation with calculated matrix
        transformed_points = points_for_transform @ transf_mat.T
        # calculate error
        error = get_mean_distance(transformed_points[:, 0:2], correct_locations)
        errors.append(error)

    return torch.tensor(errors).mean()


hyperparameter_bounds = {"mu": [0.001, 0.01],
                         "max_iters": [5, 120]}
optimizer = BayesianOptimization(f=register_and_evaluate,
                                 pbounds=hyperparameter_bounds)

print(evalueate_no_transform())
print(register_and_evaluate())

# registration_results = pyramid_registration(ref_tensor, sample_tensor)
# print(registration_results["transformation_matrices"])
# torch_save(registration_results["transformation_matrices"], save_dir + "/results.pt")
