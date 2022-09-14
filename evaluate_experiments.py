from pathlib import Path
from typing import Any, Dict, Tuple

from sklearn.decomposition import randomized_svd
from run_parallel_dev import done_param_list, open_param_list, param_grid
from tabulate import tabulate


def runtime_plots():
    pass


def uncertainty_histogram_plots():
    pass


def learning_curve_plots():
    pass


def _convert_config_to_path(config_dict) -> Path:
    pass


def _convert_config_to_tuple(config_dict) -> Tuple:
    pass


def table_stats(
    exp_name: str,
    transformer_model_name: str,
    dataset: str,
    initially_labeled_samples: int,
    batch_size: int,
    param_grid: Dict[str, Any],
):
    # group by query_strategy - uncertainty_method - lower_is_better - uncertainty_clipping in combination
    for query_strategy in param_grid["query_strategy"]:
        for uncertainty_method in param_grid["uncertainty_method"]:
            for lower_is_better in param_grid["lower_is_better"]:
                for uncertainty_clipping in param_grid["uncertainty_clipping"]:
                    # check if this configuration is available
                    _convert_config_to_path(
                        {
                            "uncertainty_method": uncertainty_method,
                            "query_strategy": query_strategy,
                            "exp_name": exp_name
                            "transformer_model_name": transformer_model_name
                            "dataset": dataset,
                            "initially_labeled_samples": initially_labeled_samples
                            "random_seed": random_seed,
                            "batch_size": batch_size,
                            "num_iterations": num_iterations,
                            "uncertainty_clipping": uncertainty_clipping
                            "lower_is_better": lower_is_better
                        }
                    )

                    # if so -> load it, if not:
                    pass


def display_run_experiment_stats():
    print("Open:")
    print(tabulate(open_param_list))

    print()
    print("Done:")
    done_param_list_without_folders = [params[0] for params in done_param_list]
    print(tabulate(done_param_list_without_folders))

    print()
    print("full grid:")
    print(tabulate(param_grid, headers="keys"))


display_run_experiment_stats()

for exp_name in param_grid["exp_name"]:
    for transformer_model_name in param_grid["transformer_model_name"]:
        for dataset in param_grid["dataset"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    table_stats(
                        exp_name,
                        transformer_model_name,
                        dataset,
                        initially_labeled_samples,
                        batch_size,
                        param_grid,
                    )
