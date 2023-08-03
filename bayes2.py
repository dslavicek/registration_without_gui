import sys
from register_and_evaluate import register_batches_and_evaluate
from bayes_opt import BayesianOptimization, UtilityFunction

# data_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/small_datasets/data_bayes"
data_dir = sys.argv[1]
# gt_dir = "C:/Users/slavi/Documents/SKOLA/DIPLOMKA2/data/small_datasets/ground_truth3"
gt_dir = sys.argv[2]
loss = sys.argv[3]
pbounds = {'mu': (0.001, 0.01)}

optimizer = BayesianOptimization(
    f=lambda mu: register_batches_and_evaluate(data_dir, gt_dir, mu=mu, loss=loss),
    pbounds=pbounds,
    random_state=1,
)
optimizer.maximize(n_iter=4, init_points=3)
print(optimizer.max)
