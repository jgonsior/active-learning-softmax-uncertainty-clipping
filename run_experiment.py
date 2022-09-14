import argparse
from ast import Param
from cmath import exp
from collections import OrderedDict
import multiprocessing
import os
from pathlib import Path
import random
import sys
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import ParameterGrid

full_param_grid = {
    "uncertainty_method": [
        "softmax",
        "temp_scaling",
        # "temp_scaling2",
        "label_smoothing",
        "MonteCarlo",
        "inhibited",
        "evidential1",
        # "evidential2",
        # "bayesian",
        # "ensembles",
        "trustscore",
        # "model_calibration",
    ],
    "query_strategy": ["LC", "MM", "Ent", "Rand", "QBC_KLD", "QBC_VE"],
    "exp_name": ["baseline"],  # ["lunchtest"],  # baseline
    "transformer_model_name": ["bert-base-uncased"],
    "dataset": ["trec6", "ag_news", "subj", "rotten", "imdb"],
    "initially_labeled_samples": [25],
    "random_seed": [42, 43, 44, 45, 46],
    "batch_size": [25],
    "num_iterations": [20],
    "uncertainty_clipping": [1.0],
    "lower_is_better": ["True", "False"],
}

dev_param_grid = full_param_grid.copy()
dev_param_grid["num_iterations"] = [2]
dev_param_grid["random_seed"] = [42]
dev_param_grid["exp_name"] = ["lunchtest"]

baselines_param_grid = full_param_grid.copy()
baselines_param_grid["uncertainty_method"] = ["softmax"]

my_methods_param_grid = full_param_grid.copy()
my_methods_param_grid["uncertainty_method"].remove("softmax")


def generate_workload(param_grid):
    done_param_list = []
    open_param_list = []
    for params in list(ParameterGrid(param_grid)):
        if (
            params["query_strategy"] == "Rand"
            and params["uncertainty_method"] != "softmax"
        ):
            continue

        params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))

        # check if has already been run
        exp_results_dir = Path(
            "exp_results/" + "-".join([str(a) for a in params.values()])
        )
        exp_results_dir_metrics = Path(exp_results_dir / "metrics.npz")

        if exp_results_dir_metrics.exists():
            # print("Experiment has already been run, exiting!")
            done_param_list.append((params, exp_results_dir))
            continue
        # print(params)
        open_param_list.append(params)
    return done_param_list, open_param_list


def run_code(
    n_gpus,
    num_iterations,
    batch_size,
    exp_name,
    dataset,
    random_seed,
    query_strategy,
    uncertainty_method,
    initially_labeled_samples,
    transformer_model_name,
    lower_is_better,
    uncertainty_clipping,
):
    gpu_device = random.randint(0, n_gpus - 1)

    cli = f"python test.py --num_iterations {num_iterations} --batch_size {batch_size} --exp_name {exp_name} --dataset {dataset} --random_seed {random_seed} --query_strategy {query_strategy} --uncertainty_method {uncertainty_method} --initially_labeled_samples {initially_labeled_samples} --transformer_model_name {transformer_model_name} --gpu_device {gpu_device} --uncertainty_clipping {uncertainty_clipping} --lower_is_better {lower_is_better}"

    print("#" * 100)
    # print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    # os.system(cli)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="An example that shows active learning "
        "for binary text classification."
    )
    parser.add_argument("--taurus", type=int, default=0)
    parser.add_argument("--workload", type=str, choices=["dev", "baselines", "my"])

    args = parser.parse_args()

    if args.taurus == 1:
        n_jobs = 1
        n_gpus = 1
    else:
        n_gpus = 2
        n_jobs = 10

    if args.workload == "dev":
        _, open_param_list = generate_workload(dev_param_grid)
    elif args.workload == "baselines":
        _, open_param_list = generate_workload(baselines_param_grid)
    elif args.workload == "my":
        _, open_param_list = generate_workload(my_methods_param_grid)
    else:
        exit(-1)

    with parallel_backend("loky", n_jobs=n_jobs):
        Parallel()(delayed(run_code)(n_gpus, **params) for params in open_param_list)
