import argparse
from collections import OrderedDict
import copy
import os
from pathlib import Path
import random
from joblib import Parallel, delayed, parallel_backend
import numpy as np
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
    "transformer_model_name": [
        "roberta-base"
    ],  # ["bert-base-uncased", "roberta-large"],
    "dataset": ["trec6", "ag_news", "subj", "rotten", "imdb"],
    "initially_labeled_samples": [25],
    "random_seed": [42, 43, 44, 45, 46],
    "batch_size": [25],
    "num_iterations": [20],
    "uncertainty_clipping": [1.0, 0.95, 0.90],
    "lower_is_better": ["True"],  # , "False"],
}

dev_param_grid = copy.deepcopy(full_param_grid)
dev_param_grid["num_iterations"] = [2]
dev_param_grid["random_seed"] = [42]
dev_param_grid["exp_name"] = ["lunchtest"]

baselines_param_grid = copy.deepcopy(full_param_grid)
baselines_param_grid["uncertainty_method"] = ["softmax"]

my_methods_param_grid = copy.deepcopy(full_param_grid)
my_methods_param_grid["uncertainty_method"].remove("softmax")
my_methods_param_grid["query_strategy"] = ["LC"]

# source: https://stackoverflow.com/a/54802737
def _chunks(l, n):
    """Yield n number of striped chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si : si + (d + 1 if i < r else d)]


def generate_workload(
    param_grid,
    array_job_id: int = 0,
    n_array_jobs: int = 1,
):
    print(f"Job Id {array_job_id}")
    done_param_list = []
    open_param_list = []
    full_param_list = []

    for params in list(ParameterGrid(param_grid)):
        if (
            params["query_strategy"] == "Rand"
            and params["uncertainty_method"] != "softmax"
        ):
            continue

        if (
            params["query_strategy"] in ["MM", "Ent", "Rand", "QBC_KLD", "QBC_VE"]
            and params["uncertainty_method"] != "softmax"
        ):
            continue

        params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))

        # check if has already been run
        exp_results_dir = Path(
            "exp_results/" + "-".join([str(a) for a in params.values()])
        )
        exp_results_dir_metrics = Path(exp_results_dir / "metrics.npz")
        full_param_list.append(params)
        if exp_results_dir_metrics.exists():
            # print("Experiment has already been run, exiting!")
            done_param_list.append((params, exp_results_dir))
            continue
        # print(params)
        open_param_list.append(params)

    full_param_list = sorted(
        full_param_list, key=lambda a: "-".join([str(x) for x in a.values()])
    )

    splitted_full_list = list(_chunks(full_param_list, n_array_jobs))

    return done_param_list, open_param_list, splitted_full_list[array_job_id]


def run_code(
    n_gpus,
    dry_run: bool,
    open_param_list,
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
    args = OrderedDict()

    args["num_iterations"] = num_iterations
    args["batch_size"] = batch_size
    args["exp_name"] = exp_name
    args["dataset"] = dataset
    args["random_seed"] = random_seed
    args["query_strategy"] = query_strategy
    args["uncertainty_method"] = uncertainty_method
    args["initially_labeled_samples"] = initially_labeled_samples
    args["transformer_model_name"] = transformer_model_name
    args["lower_is_better"] = lower_is_better
    args["uncertainty_clipping"] = uncertainty_clipping

    args = OrderedDict(sorted(args.items()))

    if args not in open_param_list:
        return

    gpu_device = random.randint(0, n_gpus - 1)

    cli = f"python test.py --num_iterations {num_iterations} --batch_size {batch_size} --exp_name {exp_name} --dataset {dataset} --random_seed {random_seed} --query_strategy {query_strategy} --uncertainty_method {uncertainty_method} --initially_labeled_samples {initially_labeled_samples} --transformer_model_name {transformer_model_name} --gpu_device {gpu_device} --uncertainty_clipping {uncertainty_clipping} --lower_is_better {lower_is_better}"

    # print("#" * 100)
    print(cli)
    # print("#" * 100)
    # print("\n")

    if not dry_run:
        os.system(cli)


if __name__ == "__main__":

    import argparse

    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="An example that shows active learning "
        "for binary text classification."
    )
    parser.add_argument("--taurus", action="store_true")
    parser.add_argument("--workload", type=str, choices=["dev", "baselines", "my"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--array_job_id", type=int, default=0)
    parser.add_argument("--n_array_jobs", type=int, default=1)

    args = parser.parse_args()

    if args.taurus:
        n_jobs = 1
        n_gpus = 1
    else:
        n_gpus = 2
        n_jobs = 10

    if args.workload == "dev":
        _, open_param_list, full_param_list = generate_workload(
            dev_param_grid,
            args.array_job_id,
            args.n_array_jobs,
        )
    elif args.workload == "baselines":
        _, open_param_list, full_param_list = generate_workload(
            baselines_param_grid,
            args.array_job_id,
            args.n_array_jobs,
        )
    elif args.workload == "my":
        _, open_param_list, full_param_list = generate_workload(
            my_methods_param_grid,
            args.array_job_id,
            args.n_array_jobs,
        )
    else:
        exit(-1)

    with parallel_backend("loky", n_jobs=n_jobs):
        Parallel()(
            delayed(run_code)(n_gpus, args.dry_run, open_param_list, **params)
            for params in full_param_list
        )
