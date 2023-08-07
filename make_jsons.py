import json, os.path

base_dir = "jsons"
for n_iters in [30, 50]:
    for loss_fcn in ['mse', 'cov']:
        inner_params = {
            'max_iters': n_iters,
            'loss_fcn': loss_fcn,
            # 'mu': 0.00475
        }
        bayes_params = {
            'pbounds': {'mu': (0.001, 0.01)} if loss_fcn == 'mse' else {'mu': (0.0002, 0.01)},
            'n_iter': 4,
            'init_points': 3,
            'inner_params': inner_params
        }

        with open(os.path.join(base_dir, f"params_{n_iters}i_{loss_fcn}.json"), 'w+') as f:
            json.dump(bayes_params, f, indent=2)
