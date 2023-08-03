import sys, json

with open('test.json', 'r') as f:
    params = json.load(f)


def bayes(data_dir, gt_dir, pbounds={}, inner_params={}, n_iter=3, init_points=3):
    print(f"pbounds: {pbounds}")
    print(f"inner_params: {inner_params}")
    print(f"n_iter: {n_iter}")
    print(f"init_points: {init_points}")


bayes('.', '.', **params)
