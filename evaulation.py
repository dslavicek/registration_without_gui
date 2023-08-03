import os.path
import matplotlib.pyplot as plt
import torch

from pandas import read_csv
from os import listdir

ground_truth_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/FIRE/FIRE/Ground Truth"
# results_path = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/results_metacentrum/results_retina_part_p_with_preprocessing.pt"
# results_path = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/results_metacentrum/results_retina_part_s_cov.pt"
results_path = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/results_retina_cov.pt"

ground_truth_files = listdir(ground_truth_dir)
ground_truth_files.sort()
#ground_truth_files = ground_truth_files[0:4]
print(ground_truth_files)
ground_truth_files = [x for x in ground_truth_files if x[15] == 'A'][0:4]
print(ground_truth_files)
transf_matrices = torch.load(results_path)
# transf_matrices = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]).repeat(4, 1, 1)


def get_mean_distance(points1, points2):
    diff = points1 - points2
    diff_sq = diff ** 2
    dists = torch.sqrt(diff_sq[:, 0] + diff_sq[:, 1])
    return dists.mean()


errors = []
for i, file in enumerate(ground_truth_files):
    gt = read_csv(os.path.join(ground_truth_dir, file), sep=' ', names=['x1', 'y1', 'x2', 'y2'])
    correct_locations = torch.tensor((gt.iloc[:, 2:4] / 1456 - 1).values, dtype=torch.float32)
    transf_mat = torch.cat((transf_matrices[i], torch.tensor([[0, 0, 1.0]])), dim=0)

    point_locations = torch.tensor((gt.iloc[:, 0:2] / 1456 - 1).values, dtype=torch.float32)
    points_for_transform = torch.cat((point_locations, torch.ones((10, 1))), dim=1)
    transformed_points = points_for_transform @ transf_mat.T
    transformed_points_rescaled = (transformed_points[:, 0:2] + 1) * 1456
    print("result:")
    print(transformed_points[:, 0:2])
    print("expected:")
    print(correct_locations)
    error = get_mean_distance(transformed_points[:, 0:2], correct_locations)
    errors.append(error)

print(f"Mean error is: {torch.tensor(errors).mean()}")
print(f"In pixels: {torch.tensor(errors).mean() * 1496}")
error_tens = torch.tensor(errors) * 1496
len_total = len(error_tens)
accuracies = []
for tolerance in range(26):
    accuracies.append(len(error_tens[error_tens < tolerance])/len_total)
print(torch.tensor(accuracies).mean())
plt.plot(accuracies)
plt.show()