import torch
import os.path
import matplotlib.pyplot as plt

from register_batches import register_batches
from os import listdir
from pandas import read_csv


def get_mean_distance(points1, points2):
    diff = points1 - points2
    diff_sq = diff ** 2
    dists = torch.sqrt(diff_sq[:, 0] + diff_sq[:, 1])
    return dists.mean()


def register_batches_and_evaluate(input_dir, ground_truth_dir, output_dir='.', params_dict={},
        mats_name="transformation_matrices.pt", fig_name='accuracy_graph.png', verbose=False):
    transf_mats = register_batches(input_dir, **params_dict)
    if mats_name is not None:
        torch.save(transf_mats, os.path.join(output_dir, mats_name))
    ground_truth_files = listdir(ground_truth_dir)
    ground_truth_files.sort()

    errors = []
    for i, file in enumerate(ground_truth_files):
        gt = read_csv(os.path.join(ground_truth_dir, file), sep=' ', names=['x1', 'y1', 'x2', 'y2'])
        correct_locations = torch.tensor((gt.iloc[:, 2:4] / 1456 - 1).values, dtype=torch.float32)
        transf_mat = torch.cat((transf_mats[i], torch.tensor([[0, 0, 1.0]])), dim=0)

        point_locations = torch.tensor((gt.iloc[:, 0:2] / 1456 - 1).values, dtype=torch.float32)
        points_for_transform = torch.cat((point_locations, torch.ones((10, 1))), dim=1)
        transformed_points = points_for_transform @ transf_mat.T
        transformed_points_rescaled = (transformed_points[:, 0:2] + 1) * 1456
        if verbose:
            print("result:")
            print(transformed_points[:, 0:2])
            print("expected:")
            print(correct_locations)
        error = get_mean_distance(transformed_points[:, 0:2], correct_locations)
        errors.append(error)

    error_tens = torch.tensor(errors) * 1496
    len_total = len(error_tens)
    accuracies = []
    for tolerance in range(26):
        accuracies.append(len(error_tens[error_tens < tolerance]) / len_total)
    print(f"AUC: {torch.tensor(accuracies).mean()}")
    if fig_name is not None:
        plt.plot(accuracies)
        plt.xlabel('tolerance [px]')
        plt.ylabel('accuracy [%]')
        plt.title(f"AUC: {torch.tensor(accuracies).mean()}")
        plt.savefig(os.path.join(output_dir, fig_name))
    return torch.tensor(accuracies).mean()


register_batches_and_evaluate("../data/FIRE/FIRE/part_a", "../data/FIRE/FIRE/gt_a", "../results", verbose=True, batch_size=2)
