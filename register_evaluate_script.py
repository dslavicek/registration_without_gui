import sys, json, os.path
from register_and_evaluate import register_batches_and_evaluate
from bayes_opt import BayesianOptimization, UtilityFunction
from register_batches import register_batches
from torch import save
data_dir = sys.argv[1]
gt_dir = sys.argv[2]
output_dir = sys.argv[3]
param_file = sys.argv[4]

with open(param_file, 'r') as f:
    params = json.load(f)
auc = register_batches_and_evaluate(data_dir, gt_dir, output_dir=output_dir, **params)

def bayes(data_dir, gt_dir, output_dir, pbounds={}, n_iter=3, init_points=3, inner_params={}):
    optimizer = BayesianOptimization(
        f=lambda mu: register_batches_and_evaluate(data_dir, gt_dir, output_dir=output_dir, mu=mu, **inner_params),
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(n_iter=n_iter, init_points=init_points)
    # I need to call registration with the best parameters and save transformation matrices to output
    transf_mats = register_batches(data_dir, mu=optimizer.max['params']['mu'])
    save(transf_mats, os.path.join(output_dir, 'tmats.pt'))
    with open(os.path.join(output_dir, 'bayes_result.txt'), 'w') as f:
        f.write(str(optimizer.max))
        f.write(' ')
        f.write(str(optimizer.res))
    print(optimizer.max)


def parse_params_and_run_bayes(data_dir, gt_dir, output_dir='.', param_file='params.json'):
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
        f.write("Parsing params for Bayes\n")
    with open(param_file, 'r') as f:
        params = json.load(f)
    bayes(data_dir, gt_dir, output_dir, **params)
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
        f.write("Bayes complete\n")


parse_params_and_run_bayes(data_dir, gt_dir, output_dir=output_dir, param_file=param_file)
