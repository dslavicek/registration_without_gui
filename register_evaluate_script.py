from register_and_evaluate import register_batches_and_evaluate
from sys import argv
#register_batches_and_evaluate("../data/FIRE/part_s", "../data/FIRE/gt_s", output_dir="../vysledky")
register_batches_and_evaluate(argv[1], argv[2], output_dir=argv[3])
