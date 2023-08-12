import sys, json, os.path
from register_and_evaluate import register_batches_and_evaluate
from bayes_opt import BayesianOptimization, UtilityFunction
from register_batches import register_batches
from torch import save, tensor
data_dir = sys.argv[1]
gt_dir = sys.argv[2]
output_dir = sys.argv[3]
inner_params = {
    "max_iters": 100,
    "loss_fcn": "cov"
}

mu = 0.0005
transf_mats = register_batches(data_dir, mu=mu)
result = register_batches_and_evaluate(data_dir, gt_dir, output_dir=output_dir, mu=mu, **inner_params)
save(transf_mats, os.path.join(output_dir, 'tmats.pt'))
save(tensor(result), os.path.join(output_dir, 'auc.pt'))