import json, sys

inner_params = {
                 'max_iters': 30,
                 'loss_fcn': 'mse',
                 # 'mu': 0.004
                 }
bayes_params = {
                'pbounds': {'mu': (0.001, 0.01)},
                'n_iter': 4,
                'init_points': 3,
                'inner_params':inner_params
                }


with open("test.json", "w") as f:
    json.dump(bayes_params, f, indent=2)
