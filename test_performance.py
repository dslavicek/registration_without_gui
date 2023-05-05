import torch
import logging


def points_distance_metric(t_mat1, t_mat2, points=None):
    # t_mat1, t_mat2 are tensors of transformation matrices with size B x 3 x 3 or B x 2 x 3, where B is batch size
    if t_mat1.shape[0] != t_mat2.shape[0]:
        logging.error("Input dimensions do not match")

    batch_size = t_mat1.shape[0]

    if points is None:
        # points = torch.tensor([[[-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, -0.5, 0.5], [1, 1, 1, 1]]])
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
    diff_sq = diff**2


    print(t_points1)
    print(t_points2)

    # print(points[0] @ t_mat2[0].T)


t_m1 = torch.tensor([[[1, 0, 0.0], [0, 1, 0.0]]])
t_m2 = torch.tensor([[[1, 0, 0.1], [0, 1, 0.0]]])

points_distance_metric(t_m1, t_m2)
