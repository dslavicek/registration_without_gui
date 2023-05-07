import torch
import logging


# target registration error
def tre(t_mat1, t_mat2, points=None):
    # t_mat1, t_mat2 are tensors of transformation matrices with size B x 3 x 3 or B x 2 x 3, where B is batch size
    if t_mat1.shape[0] != t_mat2.shape[0]:
        logging.error("Input dimensions do not match")

    batch_size = t_mat1.shape[0]

    if points is None:
        points = torch.tensor([[[0.5, 0.5, 1], [0.5, -0.5, 1], [-0.5, 0.5, 1], [-0.5, -0.5, 1]]])
        points = points.repeat([batch_size, 1, 1])

    # if matrices are in shape B x 2 x 3, convert them to B x 3 x 3
    if t_mat1.shape[1] == 2:
        padding = torch.zeros((batch_size, 1, 3))
        padding[:, 0, 2] = 1
        t_mat1 = torch.cat([t_mat1, padding], 1)
    if t_mat2.shape[1] == 2:
        padding = torch.zeros((batch_size, 1, 3))
        padding[:, 0, 2] = 1
        t_mat2 = torch.cat([t_mat2, padding], 1)

    t_points1 = torch.matmul(points, torch.transpose(t_mat1, 1, 2))
    t_points2 = torch.matmul(points, torch.transpose(t_mat2, 1, 2))

    t_points1 = t_points1[:, :, 0:2]
    t_points2 = t_points2[:, :, 0:2]

    diff = t_points2 - t_points1
    diff_sq = diff ** 2

    euc_dist = torch.sqrt(diff_sq.sum(dim=2))
    euc_dist_avg = euc_dist.mean(dim=1)

    return euc_dist_avg

# test
# t_m1 = torch.tensor([[[1, 0, 0.0], [0, 1, 0.0]], [[1, 0, 0.0], [0, 1, 0.0]]])
# t_m2 = torch.tensor([[[1, 0, 0.1], [0, 1, 0.1]], [[1, 0, 0.0], [0, 1, 0.0]]])

# pdm = tre(t_m1, t_m2)
# print(pdm)
