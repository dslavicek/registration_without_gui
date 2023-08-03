import sys, json
from register_and_evaluate2 import register_batches_and_evaluate
from bayes_opt import BayesianOptimization, UtilityFunction

# data_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/small_datasets/data_bayes"
data_dir = sys.argv[1]
# gt_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/small_datasets/ground_truth3"
gt_dir = sys.argv[2]
output_dir = sys.argv[3]
param_file = sys.argv[4]
with open(param_file, 'r') as f:
    params = json.load(f)


def bayes(data_dir, gt_dir, output_dir, **params):
    optimizer = BayesianOptimization(
        f=lambda mu: register_batches_and_evaluate(data_dir, gt_dir, params_dict=inner_params),
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(n_iter=4, init_points=3)
    print(optimizer.max)
