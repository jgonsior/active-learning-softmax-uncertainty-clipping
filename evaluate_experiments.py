from collections import OrderedDict
import enum
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np

from run_experiment import (
    full_param_grid,
    dev_param_grid,
    baselines_param_grid,
    my_methods_param_grid,
    generate_workload,
)
from tabulate import tabulate
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")


def runtime_plots():
    pass


def uncertainty_histogram_plots():
    pass


def learning_curve_plots():
    pass


def _convert_config_to_path(config_dict) -> Path:
    params = OrderedDict(sorted(config_dict.items(), key=lambda t: t[0]))

    exp_results_dir = Path("exp_results/" + "-".join([str(a) for a in params.values()]))
    return exp_results_dir


def _convert_config_to_tuple(config_dict) -> Tuple:
    pass


def table_stats(
    exp_name: str,
    transformer_model_name: str,
    dataset: str,
    initially_labeled_samples: int,
    batch_size: int,
    param_grid: Dict[str, Any],
    num_iterations: int,
    metric="test_accs",
):
    # available metrics: train_accs, test_accs, train_eces, test_eces, y_probas_train/test, times_elapsed, times_elapsed_model, queried_indices, acc_bins_train, proba_+ins, confidence scores
    print(f"Metric: {metric}")

    grouped_data = {}
    for query_strategy in param_grid["query_strategy"]:
        for uncertainty_method in param_grid["uncertainty_method"]:
            for lower_is_better in param_grid["lower_is_better"]:
                for uncertainty_clipping in param_grid["uncertainty_clipping"]:
                    key = f"{query_strategy} ({uncertainty_method}) {lower_is_better}/{uncertainty_clipping}"
                    grouped_data[key] = []
                    for random_seed in param_grid["random_seed"]:
                        # check if this configuration is available
                        exp_results_dir = _convert_config_to_path(
                            {
                                "uncertainty_method": uncertainty_method,
                                "query_strategy": query_strategy,
                                "exp_name": exp_name,
                                "transformer_model_name": transformer_model_name,
                                "dataset": dataset,
                                "initially_labeled_samples": initially_labeled_samples,
                                "random_seed": random_seed,
                                "batch_size": batch_size,
                                "num_iterations": num_iterations,
                                "uncertainty_clipping": uncertainty_clipping,
                                "lower_is_better": lower_is_better,
                            }
                        )
                        if exp_results_dir.exists():
                            metrics = np.load(exp_results_dir / "metrics.npz")
                            args = json.loads(
                                Path(exp_results_dir / "args.json").read_text()
                            )
                            metric_values = metrics[metric].tolist()
                            grouped_data[key].append(metric_values)

                    if len(grouped_data[key]) == 0:
                        del grouped_data[key]

    def _learning_curves_plot(data):
        df_data = []
        for k, v in grouped_data.items():
            for i, value in enumerate(v):
                for j, val in enumerate(value):
                    df_data.append((k, val, i, j))

        data_df = pd.DataFrame(
            df_data, columns=["Strategy", metric, "Random Seed", "Iteration"]
        )

        sns.lineplot(x="Iteration", y=metric, hue="Strategy", data=data_df)

        plots_path = Path("plots/")
        plots_path.mkdir(exist_ok=True)

        plt.savefig(
            f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.jpg"
        )
        plt.savefig(
            f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.pdf",
            dpi=300,
        )
        plt.clf()

    def _barplot(data):
        pass

    _learning_curves_plot(grouped_data)

    table_data = []

    for k, v in grouped_data.items():
        table_data.append((k, v, np.mean(v), np.std(v)))

    df = pd.DataFrame(table_data, columns=["Strategy", "Values", "Mean", "Std"])
    df["Values"] = df["Values"].apply(lambda x: [sum(v) / len(v) for v in x])
    df.sort_values(by="Mean", inplace=True)
    print(tabulate(df, headers="keys"))


def display_run_experiment_stats(param_grid):
    done_param_list, open_param_list, full_param_list = generate_workload(param_grid)
    print("Open:")
    print(tabulate(open_param_list))

    print()
    print("Done:")
    done_param_list_without_folders = [params[0] for params in done_param_list]
    print(tabulate(done_param_list_without_folders))

    print()
    print("full grid:")
    print(tabulate(param_grid, floatfmt=".2f", numalign="right", headers="keys"))


# display_run_experiment_stats(param_grid=dev_param_grid)
# display_run_experiment_stats(param_grid=my_methods_param_grid)
# display_run_experiment_stats(param_grid=baselines_param_grid)


def tables_plots(param_grid):
    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for dataset in param_grid["dataset"]:
                for initially_labeled_samples in param_grid[
                    "initially_labeled_samples"
                ]:
                    for batch_size in param_grid["batch_size"]:
                        for num_iteration in param_grid["num_iterations"]:
                            print(
                                f"{exp_name} - {transformer_model_name} - {dataset} - {initially_labeled_samples} - {batch_size} - {num_iteration}"
                            )
                            table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric="test_accs",
                            )

                            table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric="train_accs",
                            )

                            table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric="times_elapsed",
                            )

                            table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric="times_elapsed_model",
                            )
                            print()
                            exit(-1)


# tables_plots(baselines_param_grid)
# tables_plots(my_methods_param_grid)
tables_plots(full_param_grid)
# tables_plots(dev_param_grid)
