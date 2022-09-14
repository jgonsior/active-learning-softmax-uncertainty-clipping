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

param_grid = {
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
    # "query_strategy": ["QBC_KLD", "QBC_VE"],
    "exp_name": ["lunchtest"],  # ["lunchtest"],  # baseline
    "transformer_model_name": ["bert-base-uncased"],
    "dataset": ["trec6", "ag_news", "subj", "rotten", "imdb"],
    "initially_labeled_samples": [25],
    "random_seed": [42],  # , 43, 44, 45, 46],
    "batch_size": [25],
    "num_iterations": [2],  # 20 2
}

param_list = []

for params in list(ParameterGrid(param_grid)):
    if params["query_strategy"] == "Rand" and params["uncertainty_method"] != "softmax":
        continue

    params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))

    # check if has already been run
    exp_results_dir = Path("exp_results/" + "-".join([str(a) for a in params.values()]))
    exp_results_dir_metrics = Path(exp_results_dir / "metrics.npz")

    if exp_results_dir_metrics.exists():
        # print("Experiment has already been run, exiting!")
        continue
    print(params)
    param_list.append(params)
# exit(-1)


def run_code(
    num_iterations,
    batch_size,
    exp_name,
    dataset,
    random_seed,
    query_strategy,
    uncertainty_method,
    initially_labeled_samples,
    transformer_model_name,
):
    gpu_device = random.randint(0, 1)
    cli = f"python test.py --num_iterations {num_iterations} --batch_size {batch_size} --exp_name {exp_name} --dataset {dataset} --random_seed {random_seed} --query_strategy {query_strategy} --uncertainty_method {uncertainty_method} --initially_labeled_samples {initially_labeled_samples} --transformer_model_name {transformer_model_name} --gpu_device {gpu_device}"

    print("#" * 100)
    # print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)


with parallel_backend("loky", n_jobs=10):
    Parallel()(delayed(run_code)(**params) for params in param_list)
