from ast import FormattedValue
from collections import OrderedDict, defaultdict
from copy import deepcopy
import copy
from dis import findlinestarts
import enum
from heapq import nlargest
import itertools
import json
from pathlib import Path
import random
from turtle import title
from typing import Any, Dict, Tuple
from joblib import Parallel, delayed, parallel_backend
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from regex import D
from sklearn.metrics import jaccard_score
from transformers import DefaultFlowCallback
from sklearn.neighbors import KernelDensity
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
import statsmodels.api as sm                                      
from scipy.signal import argrelextrema, find_peaks, argrelmax, argrelmin

from dataset_loader import load_my_dataset
from small_text import data

font_size = 5.8

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    # "text.usetex": False,
    "font.family": "times",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": font_size,
    "font.size": font_size,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "xtick.bottom": True,
    # "figure.autolayout": True,
}

sns.set_style("white")
sns.set_context("paper")
plt.rcParams.update(tex_fonts)  # type: ignore

"""import scipy

proba = np.random.rand(100000, 5)
step = 0.05
# proba = (
#    np.mgrid[
#        0 : 1 + step : step,
#        0 : 1 + step : step,
#        0 : 1 + step : step,
#        0 : 1 + step : step,
#    ]
#    .reshape(4, -1)
#    .T
# )
proba = proba[1:]
proba = proba / np.sum(proba, axis=1)[:, None]
ent = np.apply_along_axis(lambda x: scipy.stats.entropy(x), 1, proba)
sns.histplot(data=ent)
plt.savefig(
    "plots/entropy.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
plt.clf()

# plt.hist2d(x=proba[:, 0], y=proba[:, 1])
sns.histplot(data=np.max(proba, axis=1))
plt.savefig(
    "plots/entropy_raw.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
exit(-1)
"""


# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_matplotlib_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


width = 505.89
width = 358.5049


def queried_samples_table(
    exp_name: str,
    transformer_model_name: str,
    dataset: str,
    initially_labeled_samples: int,
    batch_size: int,
    param_grid: Dict[str, Any],
    num_iterations: int,
    metric,
    table_title_prefix: str,
):
    # available metrics: train_accs, test_acc, train_eces, test_ece, y_probas_train/test, times_elapsed, times_elapsed_model, queried_indices, acc_bins_train, proba_+ins, confidence scores
    print(f"Metric: {metric}")
    grouped_data = _load_grouped_data(
        exp_name,
        transformer_model_name,
        dataset,
        initially_labeled_samples,
        batch_size,
        param_grid,
        num_iterations,
        metric,
    )
    table_data = []
    for strat_a, strat_b in itertools.combinations(grouped_data.keys(), 2):
        print(f"{strat_a} vs {strat_b}")
        jaccards = []
        for random_seed_data_a, random_seed_data_b in zip(
            grouped_data[strat_a], grouped_data[strat_b]
        ):
            queried_a = np.array(random_seed_data_a).flatten()
            queried_b = np.array(random_seed_data_b).flatten()
            jaccard = len(np.intersect1d(queried_a, queried_b)) / len(
                np.union1d(queried_a, queried_b)
            )
            jaccards.append(jaccard)
        table_data.append(
            (f"{strat_a} vs {strat_b}", jaccards, np.mean(jaccards), np.std(jaccards))
        )

    df = pd.DataFrame(table_data, columns=["Strategy", "Jaccard", "Mean", "Std"])
    df.sort_values(by="Mean", inplace=True)
    print(tabulate(df, headers="keys"))

    table_file = Path(
        f"tables/queried_samples_data_{table_title_prefix}-{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.tex"
    )
    table_file.parent.mkdir(parents=True, exist_ok=True)
    table_file.write_text(tabulate(df, headers="keys", tablefmt="latex_booktabs"))


def runtime_plots(
    exp_name: str,
    transformer_model_name: str,
    dataset: str,
    initially_labeled_samples: int,
    batch_size: int,
    param_grid: Dict[str, Any],
    num_iterations: int,
    metric,
    table_title_prefix,
):
    # available metrics: train_accs, test_acc, train_eces, test_ece, y_probas_train/test, times_elapsed, times_elapsed_model, queried_indices, acc_bins_train, proba_+ins, confidence scores
    print(f"Metric: {metric}")
    grouped_data = _load_grouped_data(
        exp_name,
        transformer_model_name,
        dataset,
        initially_labeled_samples,
        batch_size,
        param_grid,
        num_iterations,
        metric,
    )
    if len(grouped_data) == 0:
        return

    # sum up elapsed times
    df_data = []
    for k, v in grouped_data.items():
        for value in v:
            df_data.append((k, sum(value)))
    data_df = pd.DataFrame(df_data, columns=["Strategy", metric])
    print(data_df)
    fig = plt.figure(figsize=set_matplotlib_size(width, fraction=1.0))
    sns.catplot(data=data_df, y="Strategy", x=metric, kind="bar")

    plots_path = Path("plots/")
    plots_path.mkdir(exist_ok=True)

    plt.savefig(
        f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.savefig(
        f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.clf()
    plt.close("all")

    table_data = []

    for k, v in grouped_data.items():
        v = [sum(value) for value in v]
        table_data.append((k, v, np.mean(v), np.std(v)))
    df = pd.DataFrame(table_data, columns=["Strategy", "Values", "Mean", "Std"])
    df.sort_values(by="Mean", inplace=True)
    print(tabulate(df, headers="keys"))

    table_file = Path(
        f"tables/runtime_{table_title_prefix}-{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.tex"
    )
    table_file.parent.mkdir(parents=True, exist_ok=True)
    table_file.write_text(tabulate(df, headers="keys", tablefmt="latex_booktabs"))


def uncertainty_histogram_plots(
    exp_name: str,
    transformer_model_name: str,
    dataset: str,
    initially_labeled_samples: int,
    batch_size: int,
    param_grid: Dict[str, Any],
    num_iterations: int,
    metric,
    table_title_prefix: str,
):
    # available metrics: train_accs, test_acc, train_eces, test_ece, y_probas_train/test, times_elapsed, times_elapsed_model, queried_indices, acc_bins_train, proba_+ins, confidence scores
    print(f"Metric: {metric}")
    grouped_data = _load_grouped_data(
        exp_name,
        transformer_model_name,
        dataset,
        initially_labeled_samples,
        batch_size,
        param_grid,
        num_iterations,
        metric,
    )
    if len(grouped_data) == 0:
        return
    # print(grouped_data)
    df_data = []
    for k, v in grouped_data.items():
        for random_seed in v:
            # print(random_seed)
            for i, iteration in enumerate(random_seed):
                for v in iteration:
                    if metric != "confidence_scores":
                        v = np.max(v)
                    if v < 0:
                        v = v * (-1)
                    df_data.append((k, v, i))
    df = pd.DataFrame(data=df_data, columns=["Strategy", metric, "iteration"])

    """sns.displot(
        data=df,
        x=metric,
        col="Strategy",
        row="iteration",
        # binwidth=3,
        # height=3,
        facet_kws=dict(margin_titles=True),
    )
    plt.savefig(
        f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}_grouped.jpg"
    )
    plt.clf()
    plt.close('all')
    """

    for strat in df["Strategy"].unique():
        mv = df.loc[df["Strategy"] == strat][metric].astype(np.float16)
        if np.nanmax(mv) == np.inf:
            max_value = np.iinfo(np.int16).max
        else:
            max_value = np.nanmax(mv)
        if np.nanmin(mv) == 0 and max_value == 0:
            continue
        counts, bins = np.histogram(mv, bins=100, range=(np.nanmin(mv), max_value))

        fig = plt.figure(figsize=set_matplotlib_size(width, fraction=0.5))
        plt.hist(
            bins[:-1],
            weights=counts,
            bins=bins,
        )
        plt.title(f"{strat}")
        plot_path = Path(
            f"./plots/{table_title_prefix}-{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}"
        )
        plot_path.mkdir(exist_ok=True, parents=True)

        plt.savefig(
            plot_path / f"{strat.replace('/', '-')}.jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.savefig(
            plot_path / f"{strat.replace('/', '-')}.pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()
        plt.close("all")

        for iteration in df["iteration"].unique():
            mv = df.loc[(df["Strategy"] == strat) & (df["iteration"] == iteration)][
                metric
            ].astype(np.float16)
            if np.nanmax(mv) == np.inf:
                max_value = np.iinfo(np.int16).max
            else:
                max_value = np.nanmax(mv)
            if np.nanmin(mv) == 0 and max_value == 0:
                continue
            counts, bins = np.histogram(mv, bins=100, range=(np.nanmin(mv), max_value))

            fig = plt.figure(figsize=set_matplotlib_size(width, fraction=0.5))
            plt.hist(
                bins[:-1],
                weights=counts,
                bins=bins,
            )

            #  fig = plt.figure(figsize=set_matplotlib_size(width, fraction=1.0))
            #  sns.histplot(
            #  data=df.loc[(df["Strategy"] == strat) & (df["iteration"] == iteration)],
            #  x=metric,
            #  )
            plt.title(f"{strat}: {iteration}")
            plot_path = Path(
                f"./plots/{table_title_prefix}-{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}/{strat.replace('/', '-')}/"
            )
            plot_path.mkdir(exist_ok=True, parents=True)

            plt.savefig(
                plot_path / f"{iteration}.jpg", bbox_inches="tight", pad_inches=0
            )
            plt.savefig(
                plot_path / f"{iteration}.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.clf()
            plt.close("all")


def _convert_config_to_path(config_dict) -> Path:
    params = OrderedDict(sorted(config_dict.items(), key=lambda t: t[0]))

    exp_results_dir = Path(
        # "exp_results_taurus_with_class_weights/"
        "exp_results/"
        + "-".join([str(a) for a in params.values()])
    )
    return exp_results_dir


def _load_grouped_data(
    exp_name: str,
    transformer_model_name: str,
    dataset: str,
    initially_labeled_samples: int,
    batch_size: int,
    param_grid: Dict[str, Any],
    num_iterations: int,
    metric="test_acc",
    ignore_clipping_for_random_and_passive=True,
    clipping_on_which_data="all",
):
    grouped_data = {}
    for query_strategy in param_grid["query_strategy"]:
        for uncertainty_method in param_grid["uncertainty_method"]:
            for lower_is_better in param_grid["lower_is_better"]:
                for uncertainty_clipping in param_grid["uncertainty_clipping"]:
                    if (
                        query_strategy in ["passive", "Rand"]
                        and uncertainty_clipping != 1.0
                        and ignore_clipping_for_random_and_passive
                    ):
                        continue
                    elif (
                        query_strategy in ["passive", "Rand"]
                        and uncertainty_clipping != 1.0
                        and not ignore_clipping_for_random_and_passive
                    ):
                        uncertainty_clipping = 1.0
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
                                "clipping_on_which_data": clipping_on_which_data,
                            }
                        )
                        if not exp_results_dir.exists():
                            #print(exp_results_dir)
                            continue
                        else:
                            #print(exp_results_dir)
                            #continue
                            metrics = np.load(
                                exp_results_dir / "metrics.npz", allow_pickle=True
                            )
                            # print(metrics.files)
                            # args = json.loads(
                            #    Path(exp_results_dir / "args.json").read_text()
                            # )
                            metric_values = metrics[metric].tolist()
                            grouped_data[key].append(metric_values)

                    if len(grouped_data[key]) == 0:
                        del grouped_data[key]
    return grouped_data


def table_stats(
    exp_name: str,
    transformer_model_name: str,
    dataset: str,
    initially_labeled_samples: int,
    batch_size: int,
    param_grid: Dict[str, Any],
    num_iterations: int,
    metric="test_acc",
    table_title_prefix="",
    consider_last_n=21,
):
    # available metrics: train_accs, test_acc, train_eces, test_ece, y_probas_train/test, times_elapsed, times_elapsed_model, queried_indices, acc_bins_train, proba_+ins, confidence scores
    print(f"Metric: {metric}")
    grouped_data = _load_grouped_data(
        exp_name,
        transformer_model_name,
        dataset,
        initially_labeled_samples,
        batch_size,
        param_grid,
        num_iterations,
        metric,
    )

    def _learning_curves_plot(data):
        df_data = []
        for k, v in grouped_data.items():
            for i, value in enumerate(v):
                for j, val in enumerate(value):
                    df_data.append((k, val, i, j))

        data_df = pd.DataFrame(
            df_data, columns=["Strategy", metric, "Random Seed", "Iteration"]
        )
        fig = plt.figure(figsize=set_matplotlib_size(width, fraction=1.0))
        sns.lineplot(x="Iteration", y=metric, hue="Strategy", data=data_df)

        plots_path = Path("plots/")
        plots_path.mkdir(exist_ok=True)

        plt.savefig(
            f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.savefig(
            f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()
        plt.close("all")

    _learning_curves_plot(grouped_data)

    table_data = []

    for k, v in grouped_data.items():
        v = [x[-consider_last_n:] for x in v]
        table_data.append((k, v, np.mean(v), np.std(v)))

    df = pd.DataFrame(table_data, columns=["Strategy", "Values", "Mean", "Std"])
    df["Values"] = df["Values"].apply(lambda x: [sum(v) / len(v) for v in x])
    df.sort_values(by="Mean", inplace=True)
    print(tabulate(df, headers="keys"))

    table_file = Path(
        f"tables/auc_stats_{table_title_prefix}-{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.tex"
    )
    table_file.parent.mkdir(parents=True, exist_ok=True)
    table_file.write_text(tabulate(df, headers="keys", tablefmt="latex_booktabs"))


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


def _filter_out_param(param_grid, param, values_to_delete):
    for v in values_to_delete:
        if v in param_grid[param]:
            param_grid[param].remove(v)
    return param_grid


def _execute_parallel(param_grid, dataset: str):
    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        print(
                            f"{exp_name} - {transformer_model_name} - {dataset} - {initially_labeled_samples} - {batch_size} - {num_iteration}"
                        )

                        def _with_without_clipping(pg, clipping=True):
                            if clipping:
                                table_title_prefix = ""
                                param_grid_new = _filter_out_param(
                                    pg, "uncertainty_clipping", [0.95, 0.9, 0.7]
                                )
                            else:
                                table_title_prefix = "clipped"
                                param_grid_new = _filter_out_param(pg, "", [])

                            table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="test_acc",
                                table_title_prefix=table_title_prefix + "_last1",
                                consider_last_n=1,
                            )
                            table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="test_acc",
                                table_title_prefix=table_title_prefix + "_last5",
                                consider_last_n=5,
                            )

                            table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="test_acc",
                                table_title_prefix=table_title_prefix,
                                consider_last_n=21,
                            )

                            """table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="test_ece",
                                table_title_prefix=table_title_prefix + "_last5",
                                consider_last_n=5,
                            )

                            table_stats(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="test_ece",
                                table_title_prefix=table_title_prefix,
                                consider_last_n=21,
                            )

                            runtime_plots(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="times_elapsed",
                                table_title_prefix=table_title_prefix,
                            )

                            queried_samples_table(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="queried_indices",
                                table_title_prefix=table_title_prefix,
                            )"""

                            """uncertainty_histogram_plots(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="confidence_scores",
                                table_title_prefix=table_title_prefix,
                            )

                            uncertainty_histogram_plots(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid_new,
                                num_iteration,
                                metric="y_proba_test_active",
                                table_title_prefix=table_title_prefix,
                            )"""

                            """runtime_plots(
                                            exp_name,
                                            transformer_model_name,
                                            dataset,
                                            initially_labeled_samples,
                                            batch_size,
                                            param_grid_new,
                                            num_iteration,
                                            metric="times_elapsed_model",
                                        )"""
                            print()

                        _with_without_clipping(copy.deepcopy(param_grid), clipping=True)
                        _with_without_clipping(
                            copy.deepcopy(param_grid), clipping=False
                        )


def tables_plots(param_grid):
    for dataset in param_grid["dataset"]:
        _execute_parallel(param_grid, dataset)
    # with parallel_backend("loky", n_jobs=20):
    #    Parallel()(
    #        delayed(_execute_parallel)(param_grid, dataset)
    #        for dataset in param_grid["dataset"]
    #    )


def _rename_dataset_name(dataset):
    if dataset == "trec6":
        dataset2 = "TR"
    elif dataset == "ag_news":
        dataset2 = "AG"
    elif dataset == "subj":
        dataset2 = "SU"
    elif dataset == "rotten":
        dataset2 = "RT"
    elif dataset == "imdb":
        dataset2 = "IM"
    elif dataset == "sst2":
        dataset2 = "S2"
    elif dataset == "cola":
        dataset2 = "CL"
    return dataset2


def _rename_strat(strategy, clipping=True):
    strategy = strategy.replace("1.0", "100")
    strategy = strategy.replace("0.95", "95")
    strategy = strategy.replace("0.9", "90")
    strategy = strategy.replace("trustscore (softmax)", "TrSc")
    # rename strategies
    strategy = strategy.replace("True/", "")
    strategy = strategy.replace("MM (softmax)", "MM")
    strategy = strategy.replace("LC (softmax)", "LC")
    strategy = strategy.replace("LC (inhibited)", "IS")
    strategy = strategy.replace("LC (MonteCarlo)", "MC")
    strategy = strategy.replace("Ent (softmax)", "Ent")
    strategy = strategy.replace("Rand (softmax)", "Rand")
    strategy = strategy.replace("passive (softmax)", "Pass")

    strategy = strategy.replace("QBC_VE (softmax)", "VE")
    strategy = strategy.replace("QBC_KLD (softmax)", "KLD")

    strategy = strategy.replace("LC (evidential1)", "Evi")

    strategy = strategy.replace("LC (label_smoothing)", "LS")
    strategy = strategy.replace("LC (temp_scaling)", "TeSc")

    if not clipping:
        strategy = strategy.replace(" leftmost_peak90", "")
        strategy = strategy.replace(" valley_after_peak90", "")
        strategy = strategy.replace(" 100", "")
        strategy = strategy.replace(" 95", "")
        strategy = strategy.replace(" 90", "")
        strategy = strategy.replace(" leftmost_peak", "")
        strategy = strategy.replace(" valley_after_peak", "")
    else:
        strategy = strategy.replace(" 100", "")

    return strategy


def full_boxplot(pg, clipping=True, metric="test_acc", consider_last_n=21):
    if clipping:
        table_title_prefix = ""
        param_grid = _filter_out_param(pg, "uncertainty_clipping", [0.95, 0.9, 0.7])
    else:
        table_title_prefix = "clipped"
        param_grid = _filter_out_param(pg, "", [])

    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        datasets = param_grid["dataset"]
                        table_file = Path(
                            f"final/merge_datasets_{metric}_{table_title_prefix}_{exp_name}_{transformer_model_name}_{consider_last_n}_{initially_labeled_samples}_{batch_size}_{num_iteration}.tex"
                        )
                        plot_file = Path(
                            f"final/boxplots_{metric}_{table_title_prefix}_{exp_name}_{transformer_model_name}_{consider_last_n}_{initially_labeled_samples}_{batch_size}_{num_iteration}.pdf"
                        )
                        print(table_file)

                        groups = []
                        for dataset in datasets:
                            grouped_data = _load_grouped_data(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric,
                            )
                            groups.append((dataset, grouped_data))

                        table_data = []

                        for dataset, group in groups:
                            dataset2 = _rename_dataset_name(dataset)

                            for k, v in group.items():
                                print(k)
                                k = _rename_strat(k, clipping=clipping)
                                if k == "Passive":
                                    continue
                                v = [x[-consider_last_n:] for x in v]
                                v = np.mean(v, axis=1)
                                formatted_value = np.mean(v) * 100
                                table_data.append((k, formatted_value))

                        df = pd.DataFrame(table_data, columns=["Method", "Acc"])
                        df2 = df.groupby(["Method"]).mean().sort_values("Acc")

                        fig = plt.figure(
                            figsize=set_matplotlib_size(width, fraction=1.0)
                        )
                        ax = sns.boxplot(data=df, x="Acc", y="Method", order=df2.index)
                        plt.xlabel("")
                        plt.ylabel("")

                        plt.tight_layout()
                        plt.savefig(
                            plot_file, dpi=300, bbox_inches="tight", pad_inches=0
                        )
                        # plt.show()
                        plt.clf()
                        plt.close("all")


def full_violinplot(pg, metric="test_acc", consider_last_n=21, clipping_method="clipping"):
    param_grid = _filter_out_param(pg, "uncertainty_clipping", [0.9, 0.7])

    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        for clipping_on_which_data in param_grid["clipping_on_which_data"]:
                            datasets = param_grid["dataset"]
                            plot_file = Path(
                                f"final/violinplots_{metric}_{exp_name}_{transformer_model_name}_{consider_last_n}_{initially_labeled_samples}_{batch_size}_{num_iteration}_{clipping_on_which_data}_{clipping_method}.pdf"
                            )
                            #print(plot_file)

                            groups = []
                            for dataset in datasets:
                                grouped_data = _load_grouped_data(
                                    exp_name,
                                    transformer_model_name,
                                    dataset,
                                    initially_labeled_samples,
                                    batch_size,
                                    param_grid,
                                    num_iteration,
                                    metric,
                                    clipping_on_which_data=clipping_on_which_data
                                )
                                groups.append((dataset, grouped_data))

                            table_data = []
                            table_data2 = []
                            stick_data = {}

                            for dataset, group in groups:
                                dataset2 = _rename_dataset_name(dataset)
                                stick_data[dataset2] = []

                                for k, v in group.items():
                                    if k[-3:] == "1.0":
                                        clipping = "Original"
                                    elif k[-3:] == "0.9":
                                        clipping = "90%"
                                    elif k[-4:] == "0.95":
                                        clipping = "95\%"
                                    elif k.endswith("valley_after_peak"):
                                        clipping="vap"
                                    elif k.endswith("leftmost_peak"):
                                        clipping="lp"
                                    else:
                                        print("\n"*10)
                                        print(f"{k} - {v}")
                                        print("help" * 100)
                                        print("\n"*10)

                                    k = _rename_strat(k, clipping=False)
                                    #    continue
                                    v = [x[-consider_last_n:] for x in v]
                                    v = np.mean(v, axis=1)

                                    for formatted_value in v:
                                        formatted_value *= 100
                                        table_data.append((k, formatted_value, clipping))

                                    formatted_value = np.mean(v) * 100
                                    table_data2.append((k, formatted_value, clipping))

                                    if clipping == "95\%" or k in ["Rand", "Pass"]:
                                        table_data2.append((k, formatted_value, "0%"))
                                    # if k in ["Rand", "Pass"]:
                                    #    table_data2.append((k, formatted_value, "95\%"))
                                    stick_data[dataset2].append(formatted_value)
                            df = pd.DataFrame(
                                table_data, columns=["Method", "Acc", "clipping"]
                            )
                            df2 = (
                                df.groupby(["Method", "clipping"]).mean().sort_values("Acc")
                            )

                            df3 = pd.DataFrame(
                                table_data2, columns=["Method", "Acc", "clipping"]
                            )
                            df4 = (
                                df3.loc[df3["clipping"] == "0%"]
                                .groupby(["Method", "clipping"])
                                .mean()
                                .sort_values("Acc", ascending=False)
                            )
                            df3 = df3.loc[df3["clipping"] != "0%"]

                            ordering = [o[0] for o in df4.index.tolist()]

                            # move pass to the right, rand to the left

                            if "Pass" in ordering:
                                ordering.remove("Pass")
                                ordering = ["Pass"] + ordering 

                            if "Rand" in ordering:
                                ordering.remove("Rand")
                                ordering =  ordering + ["Rand"]

                            if clipping_method == "clipping":
                                df3 = df3[~df3["clipping"].isin(["vap", "lp"])]
                                df = df[~df["clipping"].isin(["vap", "lp"])]
                            elif clipping_method == "leftmost_peak":
                                df3 = df3[~df3["clipping"].isin(["95\%", "lp"])]
                                df = df[~df["clipping"].isin(["95\%", "lp"])]
                            elif clipping_method == "valley_after_peak":
                                df3 = df3[~df3["clipping"].isin(["vap", "95\%"])]
                                df = df[~df["clipping"].isin(["vap", "95\%"])]
                            
                            print(df3)
                            #print(df)

                            fig_dim = set_matplotlib_size(width, fraction=1.0)
                            fig_dim = (fig_dim[0], fig_dim[1] * 0.6)
                            fig = plt.figure(figsize=fig_dim)
                            ax = sns.violinplot(
                                data=df3,
                                y="Acc",
                                x="Method",
                                order=ordering,
                                hue="clipping",
                                split=True,
                                inner="stick",
                                bw=0.05,
                            )

                            violins = [
                                art
                                for art in ax.get_children()
                                if isinstance(art, PolyCollection)
                            ]

                            for violin in violins:
                                violin.set_alpha(0)

                            """ax2 = sns.violinplot(
                                data=df,
                                y="Acc",
                                x="Method",
                                order=ordering,
                                hue="clipping",
                                inner=None,
                                split=True,
                                bw=0.4,
                                palette=[".85", ".4"],
                                # cut=1,
                                # ax=ax,
                            )"""
                            ax2 = sns.boxplot(
                                data=df,
                                x="Method",
                                y="Acc",
                                hue="clipping",
                                order=ordering,
                                # color="white",
                                palette=[".85", ".4"],
                                width=0.3,
                                meanline=False,
                                medianprops={"linewidth": 0},
                                showmeans=True,
                                meanprops={
                                    "marker": "D",
                                    "markersize": 1,
                                    "markeredgecolor": "#fff7aa",
                                    "markerfacecolor": "#fff7aa",
                                    "zorder": 100,
                                },
                                flierprops={"marker": "D", "markersize": 1}
                                # boxprops={"zorder": 20},
                                # ax=ax3,
                            )
                            old_handles, old_labels = ax2.get_legend_handles_labels()
                            old_handles = old_handles[2:]

                            dataset_colors = {
                                dataset_name: sns.color_palette(
                                    palette="tab10", n_colors=len(stick_data.keys())
                                )[ix]
                                for ix, dataset_name in enumerate(stick_data.keys())
                            }
                            dataset_transparent_colors = copy.deepcopy(dataset_colors)
                            for k, v in dataset_colors.items():
                                dataset_colors[k] = (*v, 1)
                                dataset_transparent_colors[k] = (*v, 0.8)

                            dataset_colors = dataset_transparent_colors

                            for l in ax.lines:
                                for dataset_ix, dataset_name in enumerate(
                                    stick_data.keys()
                                ):
                                    if len(l.get_data()[1]) == 0:
                                        continue
                                    if l.get_data()[1][0] in stick_data[dataset_name]:
                                        orig_data = l.get_data()
                                        new_data = orig_data
                                        if new_data[0][1] == int(new_data[0][1]):
                                            # right sticks
                                            new_data[0][0] = -0.3 + new_data[0][1]
                                        else:
                                            # left sticks
                                            new_data[0][1] = 0.3 + new_data[0][0]
                                        l.set_data(new_data)

                                        l.set_color(dataset_colors[dataset_name])
                                        l.set_linewidth(1.5)
                                        # l.set_linestyle((dataset_ix, (1, 0.5)))
                                        l.set_linestyle("solid")
                                        l.set_solid_capstyle("butt")
                                        if dataset_name in ["AG", "TR"]:
                                            l.set(zorder=50)
                                        else:
                                            l.set(zorder=10)

                                        # copy the line, and then insert it with a different offset, and much higher opacity
                                        """line2 = copy.copy(l)
                                        line2.set_linestyle((dataset_ix + 1, (1, 0.5)))
                                        line2.set_color(
                                            dataset_transparent_colors[dataset_name]
                                        )
                                        ax.add_line(line2)"""

                            dataset_legend_handles = []
                            for dataset, dataset_color in dataset_colors.items():
                                dataset_legend_handles.append(
                                    Line2D(
                                        [0],
                                        [0],
                                        color=dataset_color,
                                        # linestyle=(0, (1, 0.5)),
                                        linestyle="solid",
                                        lw=2,
                                        label=dataset,
                                        solid_capstyle="butt",
                                    ),
                                )

                            dataset_legend_handles = [*dataset_legend_handles, *old_handles]

                            ax.legend(
                                handles=dataset_legend_handles,
                                loc="lower center",
                                ncol=2 + len(dataset_colors.keys()),
                                handlelength=1.5,
                            )

                            for tick in ax.get_xticklabels():
                                if tick._text in ["Ent", "LC", "MM"]:
                                    tick.set_color("#4c9dc9")
                                    tick.set_text("$\mathit{" + tick._text + "}")
                                elif tick._text in ["Rand", "Pass"]:
                                    tick.set_color("#ff5858")
                                    tick.set_text("$\mathit{" + tick._text + "}")
                            plt.xlabel("")
                            plt.ylabel("$acc_{last5}$ (\%)", labelpad=1)

                            # random hline
                            plt.axhline(
                                y=df4.loc["Rand", "0%"]["Acc"],
                                color="#797979",
                                linestyle="--",
                                linewidth=0.5,
                            )

                            plt.ylim(61, 99)

                            plt.tight_layout()
                            plt.savefig(
                                plot_file, dpi=300, bbox_inches="tight", pad_inches=0.01
                            )
                            plt.clf()
                            plt.close("all")


def full_table_stat(pg, clipping=True, metric="test_acc", consider_last_n=5, clipping_method="clipping"):
    """if clipping:
        table_title_prefix = ""
        param_grid = _filter_out_param(pg, "uncertainty_clipping", [1.0, 0.9, 0.7])
    else:
        table_title_prefix = "clipped"
        param_grid = _filter_out_param(pg, "uncertainty_clipping", [0.95, 0.9, 0.7])
    """
    table_title_prefix = ""
    param_grid = pg #_filter_out_param(pg, "uncertainty_clipping", [0.9, 0.7])
    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        datasets = param_grid["dataset"]
                        table_file = Path(
                            f"final/merge_datasets_{metric}_{table_title_prefix}_{exp_name}_{transformer_model_name}_{consider_last_n}_{initially_labeled_samples}_{batch_size}_{num_iteration}.tex"
                        )
                        print(table_file)

                        groups = []
                        for dataset in datasets:
                            grouped_data = _load_grouped_data(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric,
                            )
                            groups.append((dataset, grouped_data))

                        table_data = {}

                        for dataset, group in groups:
                            dataset2 = _rename_dataset_name(dataset)

                            for k, v in group.items():
                                v = [x[-consider_last_n:] for x in v]
                                v = np.mean(v, axis=1)
                                std_v = np.std(v) * 100
                                mean_v = np.mean(v) * 100

                                formatted_value = f"{mean_v:0.2f} +- ({std_v:0.2f})"

                                if k not in table_data.keys():
                                    table_data[k] = {dataset2: formatted_value}
                                else:
                                    table_data[k][dataset2] = formatted_value

                        for k, v in table_data.items():
                            table_data[k]["Mean"] = "{0:.2f}".format(
                                np.mean([float(x[:5]) for x in v.values()])
                            )
                        df = pd.DataFrame(table_data)
                        

                        df = df.T
                        df.reset_index(inplace=True)
                        df = df.rename(columns={"index": "Method"})
                        df.sort_values(by="Mean", inplace=True)

                        df["Method"] = df["Method"].apply(
                            lambda x: _rename_strat(x, clipping=clipping)
                        )
                        
                        #print(df)
                       
                        # df = df.set_index("Method")
                        #print(df)

                        print(
                            tabulate(
                                df,
                                headers="keys",
                                floatfmt=("0.2f"),
                            )
                        )

                        table_file.parent.mkdir(parents=True, exist_ok=True)
                        table_file.write_text(
                            tabulate(
                                df,
                                headers="keys",
                                tablefmt="latex_booktabs",
                                showindex=False,
                                floatfmt=("0.2f"),
                            )
                        )

def merged_stats_table(pg, clipping=True, metric="test_acc", consider_last_n=5):
    """if clipping:
        table_title_prefix = ""
        param_grid = _filter_out_param(pg, "uncertainty_clipping", [1.0, 0.9, 0.7])
    else:
        table_title_prefix = "clipped"
        param_grid = _filter_out_param(pg, "uncertainty_clipping", [0.95, 0.9, 0.7])
    """
    table_title_prefix = ""
    param_grid = pg #_filter_out_param(pg, "uncertainty_clipping", [0.9, 0.7])
    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        for clipping_on_which_data in param_grid["clipping_on_which_data"]:

                            datasets = param_grid["dataset"]
                            table_file = Path(
                                f"final/merged_stats_datasets_{metric}_{clipping_on_which_data}_{table_title_prefix}_{exp_name}_{transformer_model_name}_{consider_last_n}_{initially_labeled_samples}_{batch_size}_{num_iteration}.tex"
                            )
                            print(table_file)

                            groups = []
                            for dataset in datasets:
                                grouped_data = _load_grouped_data(
                                    exp_name,
                                    transformer_model_name,
                                    dataset,
                                    initially_labeled_samples,
                                    batch_size,
                                    param_grid,
                                    num_iteration,
                                    metric,
                                    clipping_on_which_data=clipping_on_which_data
                                )
                                groups.append((dataset, grouped_data))

                            table_data = []
                            table_data2 = []

                            for dataset, group in groups:
                                for k, v in group.items():
                                    if k[-3:] == "1.0":
                                        clipping = "Original"
                                    elif k[-3:] == "0.9":
                                        clipping = "90%"
                                    elif k[-4:] == "0.95":
                                        clipping = "95\%"
                                    elif k.endswith("valley_after_peak"):
                                        clipping="vap"
                                    elif k.endswith("leftmost_peak"):
                                        clipping="lp"
                                    elif k.endswith("valley_after_peak90"):
                                        clipping="vap90"
                                    elif k.endswith("leftmost_peak90"):
                                        clipping="lp90"
                                    else:
                                        print("\n"*10)
                                        print(f"{k} - {v}")
                                        print("help" * 100)
                                        print("\n"*10)

                                    k = _rename_strat(k, clipping=False)
                                    #    continue
                                    v = [x[-consider_last_n:] for x in v]
                                    #v = np.mean(v, axis=1)

                                    v = np.mean(v, axis=1)
                                    #std_v = np.std(v) * 100
                                    #mean_v = np.mean(v) * 100

                                    

                                    for formatted_value in v:
                                        formatted_value *= 100
                                        #formatted_value2 = f"{mean_v:0.2f} +- ({std_v:0.2f})"
                                        table_data.append((k, formatted_value, clipping))
                                        #table_data2.append((k, formatted_value2, clipping))

                                    #formatted_value = np.mean(v) * 100
                            df = pd.DataFrame(
                                table_data, columns=["Method", "Acc", "clipping"]
                            )        

                            # Pass, Rand, VE, KLD, Etn
                            #df = df[~df["Method"].isin(["Pass", "Rand", "Ent", "KLD","VE", "90%"])]
                            df = df[~df["Method"].isin(["Pass", "Rand", "90%", "KLD", "Ent"])]
                            
                            #df = df.groupby(['Method', "clipping"]).mean()     
                            df = df.drop("Method", axis=1)

                            print(df)
                            df2 = df.groupby(["clipping"]).agg({'Acc':list}) 
                            df2['Acc'] = df2['Acc'].apply(lambda r: f"{np.mean(r):0.2f} +-{np.std(r):0.2f}")
                            df = df.groupby(["clipping"]).mean()     

                            
                            df = df.sort_values(by="Acc")  
                            df2 = df2.sort_values(by="Acc")  

                            print(df)
                            print(df2)

                            exit(-1)         

                            
                            print(
                                tabulate(
                                    df,
                                    headers="keys",
                                    floatfmt=("0.2f"),
                                )
                            )

                            table_file.parent.mkdir(parents=True, exist_ok=True)
                            table_file.write_text(
                                tabulate(
                                    df,
                                    headers="keys",
                                    tablefmt="latex_booktabs",
                                    showindex=False,
                                    floatfmt=("0.2f"),
                                )
                            )


def strat_dataset_improvements_table(pg, clipping=True, metric="test_acc", consider_last_n=5):
    """
    1. for each dataset: take original, unclipped as "base value"
    2. calculate improvement/dispromvent for each clipped version compared to base value for this dataset
    3. sum up relative improvements per dataset (percentages)
    4. display averaged relative improvements per strat and per clipping method
    """
    table_title_prefix = ""
    param_grid = pg
    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        for clipping_on_which_data in param_grid["clipping_on_which_data"]:

                            datasets = param_grid["dataset"]
                            table_file = Path(
                                f"final/strat_dataset_improvements_table_{metric}_{clipping_on_which_data}_{table_title_prefix}_{exp_name}_{transformer_model_name}_{consider_last_n}_{initially_labeled_samples}_{batch_size}_{num_iteration}.tex"
                            )
                            print(table_file)

                            groups = []
                            for dataset in datasets:
                                grouped_data = _load_grouped_data(
                                    exp_name,
                                    transformer_model_name,
                                    dataset,
                                    initially_labeled_samples,
                                    batch_size,
                                    param_grid,
                                    num_iteration,
                                    metric,
                                    clipping_on_which_data=clipping_on_which_data
                                )
                                groups.append((dataset, grouped_data))

                            base_values = {}

                            for dataset, group in groups:
                                for k,v in group.items():
                                    print(k)
                                    if k.endswith("1.0"):  
                                        k = _rename_strat(k, clipping=False)

                                        if (dataset,k) in base_values.keys():
                                            base_values[(dataset, k)].append([vvv[-consider_last_n:] for vvv in v])
                                        else:
                                            base_values[(dataset, k)] = [[vvv[-consider_last_n:] for vvv in v]]
                            for k in base_values.keys():
                                base_values[k] = np.mean(base_values[k])

                            table_data = []
                            for dataset, group in groups:
                                for k, v in group.items():
                                    if k[-3:] == "1.0":
                                        clipping = "Original"
                                    elif k[-3:] == "0.9":
                                        clipping = "90%"
                                    elif k[-4:] == "0.95":
                                        clipping = "95\%"
                                    elif k.endswith("valley_after_peak"):
                                        clipping="vap"
                                    elif k.endswith("leftmost_peak"):
                                        clipping="lp"
                                    elif k.endswith("valley_after_peak90"):
                                        clipping="vap90"
                                    elif k.endswith("leftmost_peak90"):
                                        clipping="lp90"
                                    else:
                                        print("\n"*10)
                                        print(f"{k} - {v}")
                                        print("help" * 100)
                                        print("\n"*10)

                                    k = _rename_strat(k, clipping=False)
                                    #    continue
                                    v = [x[-consider_last_n:] for x in v]
                                    #v = np.mean(v, axis=1)

                                    v = np.mean(v)
                                    if (dataset,k) in base_values.keys():
                                        v = v - base_values[(dataset, k)]
                                    else:
                                        v = -99
                                    #std_v = np.std(v) * 100
                                    #mean_v = np.mean(v) * 100

                                    

                                    table_data.append((k, v*100, clipping))
                                        
                            df = pd.DataFrame(
                                table_data, columns=["Method", "Acc", "clipping"]
                            )        

                            # Pass, Rand, VE, KLD, Etn
                            #df = df[~df["Method"].isin(["Pass", "Rand", "Ent", "KLD","VE", "90%"])]
                            #df = df[~df["Method"].isin(["Pass", "Rand", "90%", "KLD", "Ent"])]
                            df = df[~df["Method"].isin(["Pass", "Rand"])]
                            
                            df = pd.pivot_table(df, values="Acc", index="clipping", columns="Method")
                            print(df)
                            print("todo: sortierung!")

                            exit(-1)
                            
                            print(
                                tabulate(
                                    df,
                                    headers="keys",
                                    floatfmt=("0.2f"),
                                )
                            )

                            table_file.parent.mkdir(parents=True, exist_ok=True)
                            table_file.write_text(
                                tabulate(
                                    df,
                                    headers="keys",
                                    tablefmt="latex_booktabs",
                                    showindex=False,
                                    floatfmt=("0.2f"),
                                )
                            )



def full_significance_tests(
    pg, metric="test_acc", consider_last_n=5, transformer_model_name="bert-base-uncased"
):
    # a) vergleich pro datensatz, wie in CAL
    # b) vergleich zwischen unc_clipping und nicht unc_clipping
    # c) alle datenpunkte zwischen allen datensätzen zusammenschmeißen -> ist es im mittel besser unc_clipping zu nehmen? -> das ganze pro spalte machen?!
    # d) schauen, ob meine 10 runs auserichend waren

    pg_clipped = _filter_out_param(
        copy.deepcopy(pg), "uncertainty_clipping", [1.0, 0.9, 0.7]
    )
    pg_unclipped = _filter_out_param(
        copy.deepcopy(pg), "uncertainty_clipping", [0.95, 0.9, 0.7]
    )

    def _get_signifcance_data(param_grid):
        for exp_name in param_grid["exp_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        datasets = param_grid["dataset"]

                        groups = []
                        for dataset in datasets:
                            grouped_data = _load_grouped_data(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric,
                            )
                            groups.append((dataset, grouped_data))
                        table_data = {}

                        for dataset, group in groups:
                            dataset2 = _rename_dataset_name(dataset)

                            for k, v in group.items():
                                k = _rename_strat(k)
                                v = [x[-consider_last_n:] for x in v]
                                formatted_value = np.mean(v, axis=1)

                                if k not in table_data.keys():
                                    table_data[k] = {dataset2: formatted_value}
                                else:
                                    table_data[k][dataset2] = formatted_value

                        # for k, v in table_data.items():
                        #    table_data[k]["Mean"] = np.array(list(v.values())).flatten()
                        df = pd.DataFrame(table_data)
                        return df.T

    clipping_data = _get_signifcance_data(pg_clipped)
    unclipping_data = _get_signifcance_data(pg_unclipped)

    print(clipping_data)

    from deepsig import aso

    per_dataset = [["Method", *clipping_data.columns.to_list()]]
    for strat in unclipping_data.index:
        if strat in ["Rand", "Pass"]:
            continue

        a = clipping_data.loc[strat + " 95"].to_numpy()
        b = unclipping_data.loc[strat].to_numpy()
        aso_values = [
            aso(
                a1,
                b1,
                confidence_level=0.95,
                num_comparisons=len(clipping_data.columns),
                show_progress=False,
                num_jobs=8,
            )
            for a1, b1 in zip(a, b)
        ]

        # aso_values = [
        #    "\\textbf{" + av + "}" if float(av) < 0.5 else av for av in aso_values
        # ]
        per_dataset.append([strat, *aso_values])

        df = pd.DataFrame(per_dataset)
        df.to_csv(f"final/_data_significance-{transformer_model_name}.csv", index=False)


def full_significance_heatmaps(transformer_model_name="bert-base-uncased"):
    df = pd.read_csv(
        f"final/_data_significance-{transformer_model_name}.csv",
        index_col=0,
    )
    print(df)

    df.drop("VE", axis=0, inplace=True)
    df.drop("KLD", axis=0, inplace=True)
    df.drop("Ent", axis=0, inplace=True)

    fig_dim = set_matplotlib_size(
        width,
        fraction=0.5,
    )
    fig = plt.figure(figsize=fig_dim)

    sns.heatmap(
        df,
        annot=True,
        cmap="coolwarm",
        center=0.5,
        fmt=".1f",
    )

    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig(
        f"final/significance-{transformer_model_name}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()
    plt.clf()
    plt.close("all")


def show_values_on_bars(axs, h_v="v", space=4.0, xlim_additional=0):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + float(xlim_additional)
                value = int(p.get_height())

                ax.text(_x, _y, value, ha="center")

                current_ylim = ax.get_ylim()
                current_ylim = (current_ylim[0], current_ylim[1] + xlim_additional)
                ax.set_ylim(current_ylim)
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - 0.2
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")
                current_xlim = ax.get_xlim()
                current_xlim = (current_xlim[0], current_xlim[1] + xlim_additional)
                ax.set_xlim(current_xlim)

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def full_runtime_stats(pg, clipping=True, metric="times_elapsed", consider_last_n=21, clipping_on_which_data="all", clipping_method="clipping"):
    if clipping:
        table_title_prefix = ""
        param_grid = _filter_out_param(pg, "uncertainty_clipping", [0.95, 0.9, 0.7])
    else:
        table_title_prefix = "clipped"
        param_grid = _filter_out_param(pg, "", [])

    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        datasets = param_grid["dataset"]
                        table_file = Path(
                            f"final/merge_datasets_{metric}_{table_title_prefix}_{exp_name}_{clipping_on_which_data}_{clipping_method}_{transformer_model_name}_{consider_last_n}_{initially_labeled_samples}_{batch_size}_{num_iteration}.pdf"
                        )
                        print(table_file)

                        groups = []
                        for dataset in datasets:
                            grouped_data = _load_grouped_data(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric,
                                clipping_on_which_data=clipping_on_which_data,
                            )
                            groups.append((dataset, grouped_data))

                        if len(grouped_data) == 0:
                            return

                        # sum up elapsed times
                        df_data = defaultdict(lambda: 0)
                        for dataset, group in groups:
                            for k, v in group.items():
                                if not k.endswith(clipping_method):
                                    continue
                                df_data[k] = []
                                for value in v:
                                    df_data[k].append(sum(value))
                                df_data[k] = np.mean(df_data[k])


                        df_data2 = []
                        for k, v in df_data.items():
                            if k in [
                                "Rand (softmax) True/1.0",
                                "passive (softmax) True/1.0",
                            ]:
                                continue
                            df_data2.append([_rename_strat(k, clipping=False), v])

                        data_df = pd.DataFrame(df_data2, columns=["Strategy", metric])

                        data_df.sort_values(by=metric, inplace=True)

                        
                        # fig = plt.figure(figsize=(width * 0.5 / 72.27, 100 / 72.27))
                        fig = plt.figure(
                            figsize=set_matplotlib_size(width, fraction=0.5)
                        )
                        ax = sns.barplot(data=data_df, y="Strategy", x=metric)

                        show_values_on_bars(ax, "h", xlim_additional=10)

                        for tick in itertools.chain(
                            ax.get_xticklabels(), ax.get_yticklabels()
                        ):
                            if tick._text in ["Ent", "LC", "MM"]:
                                tick.set_color("#4c9dc9")
                                tick.set_text("$\mathit{" + tick._text + "}")
                            elif tick._text in ["Rand", "Pass"]:
                                tick.set_color("#e18e4c")
                                tick.set_text("$\mathit{" + tick._text + "}")

                        plt.xlabel("")
                        plt.ylabel("")

                        plt.tight_layout()
                        plt.savefig(
                            table_file, dpi=300, bbox_inches="tight", pad_inches=0
                        )
                        # plt.show()
                        plt.clf()
                        plt.close("all")


def full_passive_comparison(
    pg, clipping=True, metric="times_elapsed", consider_last_n=21
):
    # get list of outliers based on passive
    # test, which strategy has queried the most outliers
    # does it change when using uncertainty clipping?!

    if clipping:
        table_title_prefix = ""
        param_grid = _filter_out_param(pg, "uncertainty_clipping", [0.95, 0.9, 0.7])
    else:
        table_title_prefix = "clipped"
        param_grid = _filter_out_param(pg, "", [])

    for exp_name in param_grid["exp_name"]:
        for transformer_model_name in param_grid["transformer_model_name"]:
            for initially_labeled_samples in param_grid["initially_labeled_samples"]:
                for batch_size in param_grid["batch_size"]:
                    for num_iteration in param_grid["num_iterations"]:
                        datasets = param_grid["dataset"]
                        table_file = Path(
                            f"final/merge_datasets_{metric}_{table_title_prefix}_{exp_name}_{transformer_model_name}_{consider_last_n}_{initially_labeled_samples}_{batch_size}_{num_iteration}.pdf"
                        )
                        print(table_file)

                        groups = []
                        for dataset in datasets:
                            grouped_data = _load_grouped_data(
                                exp_name,
                                transformer_model_name,
                                dataset,
                                initially_labeled_samples,
                                batch_size,
                                param_grid,
                                num_iteration,
                                metric,
                            )
                            groups.append((dataset, grouped_data))

                        if len(grouped_data) == 0:
                            return

                        # sum up elapsed times
                        df_data = defaultdict(lambda: 0)
                        for dataset, group in groups:
                            for k, v in group.items():
                                for value in v:
                                    df_data[k] += sum(value)

                        df_data2 = []
                        for k, v in df_data.items():
                            df_data2.append(
                                [_rename_strat(k, clipping=clipping), v / 60]
                            )

                        data_df = pd.DataFrame(df_data2, columns=["Strategy", metric])

                        data_df.sort_values(by=metric, inplace=True)

                        fig = plt.figure(
                            figsize=set_matplotlib_size(width, fraction=0.5)
                        )
                        ax = sns.barplot(data=data_df, y="Strategy", x=metric)

                        show_values_on_bars(ax, "h", xlim_additional=3)

                        plt.xlabel("")
                        plt.ylabel("")

                        plt.tight_layout()
                        plt.savefig(
                            table_file, dpi=300, bbox_inches="tight", pad_inches=0
                        )
                        # plt.show()
                        plt.clf()
                        plt.close("all")


def _plot_class_heatmap(data, ax, title, v_min, v_max):
    print(np.min(data.to_numpy()))
    print(np.max(data.to_numpy()))

    ax = sns.heatmap(
        data,
        annot=True,
        # cmap=sns.color_palette("husl", as_cmap=True),
        center=0,
        # square=True,
        fmt=".1f",
        ax=ax,
        vmin=v_min,
        vmax=v_max,
        xticklabels=True,
        yticklabels=True,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(100, 0))

    # ax.set_title(f"{title[0]}-{title[1]}")

    return ax


def full_class_distribution(
    pg,
    datasets_to_consider=["trec6", "ag_news"],
    clippings=[1.0, 0.95],
    transformer_model_name="bert-base-uncased",
      clipping_on_which_data="all", 
):
    # train/test class distribution
    # class distribution per strategy
    pg["dataset"] = datasets_to_consider

    def _count_unique_percentages(Ys):
        uniques = np.unique(Ys, return_counts=True)[1]
        return [counts / np.sum(uniques) for counts in uniques]

    results = []
    datasets = datasets_to_consider
    for clipping in clippings:
        param_grid = copy.deepcopy(pg)
        param_grid["uncertainty_clipping"] = [clipping]
        param_grid["transformer_model_name"] = [transformer_model_name]

        for exp_name in param_grid["exp_name"]:
            for transformer_model_name in param_grid["transformer_model_name"]:
                for initially_labeled_samples in param_grid[
                    "initially_labeled_samples"
                ]:
                    for batch_size in param_grid["batch_size"]:
                        for num_iteration in param_grid["num_iterations"]:
                            dataset_counts_test = {}
                            dataset_counts_train = {}
                            Y_trains = {}
                            Y_tests = {}

                            for dataset in datasets:
                                # print(dataset)
                                train, test, _ = load_my_dataset(
                                    dataset, "bert-base-uncased", tokenization=False
                                )
                                train_Y = train["label"]
                                test_Y = test["label"]

                                train_Y_uniques = _count_unique_percentages(train_Y)
                                test_Y_uniques = _count_unique_percentages(test_Y)

                                dataset_counts_test[dataset] = test_Y_uniques
                                dataset_counts_train[dataset] = train_Y_uniques

                                Y_trains[dataset] = train_Y
                                Y_tests[dataset] = test_Y

                            groups = {}
                            for dataset in datasets:
                                grouped_data = _load_grouped_data(
                                    exp_name,
                                    transformer_model_name,
                                    dataset,
                                    initially_labeled_samples,
                                    batch_size,
                                    param_grid,
                                    num_iteration,
                                    metric="queried_indices",
                                    clipping_on_which_data=clipping_on_which_data
                                )
                                grouped_data = {
                                    _rename_strat(k, clipping=False): v
                                    for k, v in grouped_data.items()
                                }
                                groups[dataset] = grouped_data

                            if len(grouped_data) == 0:
                                return

                            for dataset in datasets:
                                queried_percentages = {}
                                for strategy in groups[dataset].keys():
                                    queried_percentages[strategy] = []
                                    for random_seed in range(
                                        0, np.shape(groups[dataset][strategy])[0]
                                    ):
                                        queriend_indices = np.array(
                                            groups[dataset][strategy][random_seed]
                                        ).flatten()

                                        queried_Ys = np.array(Y_trains[dataset])[
                                            queriend_indices
                                        ]

                                        queried_percentages[strategy] = [
                                            *queried_percentages[strategy],
                                            *queried_Ys,
                                        ]

                                    queried_percentages[
                                        strategy
                                    ] = _count_unique_percentages(
                                        queried_percentages[strategy]
                                    )

                                df = pd.DataFrame(queried_percentages)
                                df["Test"] = dataset_counts_test[dataset]
                                df["Train"] = dataset_counts_train[dataset]

                                # normalize data using true distribution in train test
                                df = df.apply(lambda col: col - df["Train"])
                                df = df.apply(lambda x: x * 100)

                                del df["Train"]
                                if "Pass" in df.columns:
                                    del df["Pass"]

                                df = df.T
                                df = df.rename(columns=lambda c: chr(ord("A") + c))

                                results.append((dataset, clipping, df))

    # create 4 supblots
    fig, axs = plt.subplots(
        2,
        2,
        figsize=set_matplotlib_size(width, fraction=1.0),
    )

    min_ag_news = -20
    max_ag_news = 20

    min_trec = -15
    max_trec = 15

    ax00 = _plot_class_heatmap(
        results[0][2],
        ax=axs[0, 0],
        title=results[0],
        v_min=min_trec,
        v_max=max_trec,
    )

    ax01 = _plot_class_heatmap(
        results[1][2],
        ax=axs[0, 1],
        title=results[1],
        v_min=min_ag_news,
        v_max=max_ag_news,
    )
    ax10 = _plot_class_heatmap(
        results[2][2],
        ax=axs[1, 0],
        title=results[2],
        v_min=min_trec,
        v_max=max_trec,
    )
    ax11 = _plot_class_heatmap(
        results[3][2],
        ax=axs[1, 1],
        title=results[3],
        v_min=min_ag_news,
        v_max=max_ag_news,
    )

    for axa in [ax00, ax01, ax10, ax11]:
        axa.set_xlabel("")
        axa.set_ylabel("")
        axa.tick_params(axis="x", bottom=False)

    ax00.set_ylabel("No Clipping")
    ax10.set_ylabel("Uncertainty Clipping 95\%")

    # remove unecessary yaxis
    ax01.yaxis.set_visible(False)
    ax11.yaxis.set_visible(False)
    ax00.xaxis.set_ticks([])
    ax01.xaxis.set_ticks([])

    ax00.set_xlabel("TREC6")
    ax00.xaxis.set_label_position("top")
    ax01.set_xlabel("AG_NEWS")
    ax01.xaxis.set_label_position("top")
    plt.tight_layout()

    for ax in [ax00, ax01, ax10, ax11]:
        for tick in itertools.chain(ax.get_xticklabels(), ax.get_yticklabels()):
            if tick._text in ["Ent", "LC", "MM"]:
                tick.set_color("#4c9dc9")
                tick.set_text("$\mathit{" + tick._text + "}")
            elif tick._text in ["Rand", "Pass", "Test", "Wrong"]:
                tick.set_color("#e18e4c")
                tick.set_text("$\mathit{" + tick._text + "}")

    table_file = Path(f"final/class_distributions.pdf")
    print(table_file)
    plt.savefig(table_file, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close("all")


def _flatten(list_to_flatten):
    return [num for elem in list_to_flatten for num in elem]


def _vector_indice_heatmap(data, ax, title, vmin, vmax, other_data=None):
    results = []
    for a, b in itertools.product(data.keys(), repeat=2):
        outliers_per_random_seed_a = set(_flatten(data[a]))
        outliers_per_random_seed_b = set(_flatten(data[b]))

        results.append(
            (
                a,
                b,
                len(outliers_per_random_seed_a.intersection(outliers_per_random_seed_b))
                / len(outliers_per_random_seed_a.union(outliers_per_random_seed_b))
                * 100,
            )
        )

    if other_data:
        other_results = []
        for a, b in itertools.product(other_data.keys(), repeat=2):
            outliers_per_random_seed_a = set(_flatten(other_data[a]))
            outliers_per_random_seed_b = set(_flatten(other_data[b]))

            other_results.append(
                (
                    a,
                    b,
                    len(
                        outliers_per_random_seed_a.intersection(
                            outliers_per_random_seed_b
                        )
                    )
                    / len(outliers_per_random_seed_a.union(outliers_per_random_seed_b))
                    * 100,
                )
            )
        new_df = pd.DataFrame(results, columns=["a", "b", "agreement"])
        new_df = new_df.pivot(index="a", columns="b", values="agreement")
        original_df = pd.DataFrame(other_results, columns=["a", "b", "agreement"])
        original_df = original_df.pivot(index="a", columns="b", values="agreement")

        annotation_dataframe = new_df - original_df  # - new_df

        annotation = annotation_dataframe
    else:
        annotation = True

    result_df = pd.DataFrame(results, columns=["a", "b", "agreement"])

    print("smallest: ", result_df["agreement"].min())
    print("second largest: ", np.unique(result_df["agreement"].to_numpy())[-2])

    result_df = result_df.pivot(index="a", columns="b", values="agreement")
    mask = np.zeros_like(result_df.to_numpy())
    mask[np.diag_indices_from(mask)] = True

    ax = sns.heatmap(
        result_df,
        annot=annotation,
        mask=mask,
        cmap=sns.color_palette("husl", as_cmap=True),
        # square=True,
        fmt=".0f",
        ax=ax,
        # linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
        xticklabels=True,
        yticklabels=True,
    )
    # ax.set_title(f"{title[0]}-{title[1]}")

    return ax


def full_outlier_comparison(
    pg,clipping_method=0.95,clipping_on_which_data="all"
):
    results = []

    for clipping in [1.0, clipping_method]:
        param_grid = copy.deepcopy(pg)
        param_grid["uncertainty_clipping"] = [clipping]
        for transformer_model_name in param_grid["transformer_model_name"]:
            for exp_name in param_grid["exp_name"]:
                for initially_labeled_samples in param_grid[
                    "initially_labeled_samples"
                ]:
                    for batch_size in param_grid["batch_size"]:
                        for num_iteration in param_grid["num_iterations"]:
                            datasets = param_grid["dataset"]

                            groups = {}
                            for dataset in datasets:
                                grouped_data = _load_grouped_data(
                                    exp_name,
                                    transformer_model_name,
                                    dataset,
                                    initially_labeled_samples,
                                    batch_size,
                                    param_grid,
                                    num_iteration,
                                    metric="queried_indices",
                                    ignore_clipping_for_random_and_passive=False,
                                    clipping_on_which_data=clipping_on_which_data
                                )
                                grouped_data = {
                                    _rename_strat(k, clipping=False): v
                                    for k, v in grouped_data.items()
                                }

                                groups[dataset] = grouped_data

                            if len(grouped_data) == 0:
                                return

                            outliers = {}
                            queried_indices = {}

                            for dataset in datasets:
                                outliers[dataset] = []
                                queried_indices[dataset] = {}

                                for strategy in groups[dataset].keys():
                                    if strategy == "Pass":
                                        strategy = "Wrong"
                                        queried_indices[dataset][strategy] = []
                                        for random_seed in param_grid["random_seed"]:
                                            passive_path = _convert_config_to_path(
                                                {
                                                    "uncertainty_method": "softmax",
                                                    "query_strategy": "passive",
                                                    "exp_name": exp_name,
                                                    "transformer_model_name": transformer_model_name,
                                                    "dataset": dataset,
                                                    "initially_labeled_samples": initially_labeled_samples,
                                                    "random_seed": random_seed,
                                                    "batch_size": batch_size,
                                                    "num_iterations": num_iteration,
                                                    "uncertainty_clipping": "1.0",
                                                    "lower_is_better": True,
                                                    "clipping_on_which_data": clipping_on_which_data
                                                }
                                            )
                                            if passive_path.exists():
                                                metrics = np.load(
                                                    passive_path / "metrics.npz",
                                                    allow_pickle=True,
                                                )
                                                outliers[dataset].append(
                                                    metrics["passive_outlier"].tolist()[
                                                        0
                                                    ][0]
                                                )

                                                queried_indices[dataset][
                                                    strategy
                                                ].append(
                                                    [
                                                        int(q)
                                                        for q in metrics[
                                                            "passive_outlier"
                                                        ][0][0].tolist()
                                                    ]
                                                )
                                            else:
                                                print("Alarm"*100)
                                    else:
                                        queried_indices[dataset][strategy] = []
                                        for random_seed in range(
                                            0, np.shape(groups[dataset][strategy])[0]
                                        ):
                                            queried_indices[dataset][strategy].append(
                                                np.array(
                                                    groups[dataset][strategy][
                                                        random_seed
                                                    ]
                                                )
                                                .flatten()
                                                .tolist()
                                            )

                            merged_strats = {}

                            for ix, dataset in enumerate(queried_indices.keys()):
                                ix += 1
                                for strat in queried_indices[dataset].keys():
                                    if not strat in merged_strats.keys():
                                        merged_strats[strat] = []
                                    indices_of_dataset = _flatten(
                                        queried_indices[dataset][strat]
                                    )

                                    # make indices of each dataset different
                                    indices_of_dataset = [
                                        iod + 100**ix for iod in indices_of_dataset
                                    ]

                                    merged_strats[strat].append(indices_of_dataset)
                            results.append(
                                (clipping, transformer_model_name, merged_strats)
                            )
    
    # create 4 supblots
    fig, axs = plt.subplots(
        2,
        2,
        figsize=set_matplotlib_size(width, fraction=1.0),
        sharex=True,
        sharey=True,
    )

    v_min = 6
    v_max = 70

    ax00 = _vector_indice_heatmap(
        results[0][2], ax=axs[0, 0], title=results[0], vmin=v_min, vmax=v_max
    )
    ax01 = _vector_indice_heatmap(
        results[2][2],
        ax=axs[0, 1],
        title=results[2],
        vmin=v_min,
        vmax=v_max,
        other_data=results[0][2],
    )
    ax10 = _vector_indice_heatmap(
        results[1][2], ax=axs[1, 0], title=results[1], vmin=v_min, vmax=v_max
    )
    ax11 = _vector_indice_heatmap(
        results[3][2],
        ax=axs[1, 1],
        title=results[3],
        vmin=v_min,
        vmax=v_max,
        other_data=results[0][2],
    )

    """cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(100, 0))                      
    """
    for axa in [ax00, ax01, ax10, ax11]:
        axa.set_xlabel("")
        axa.set_ylabel("")
        axa.tick_params(axis="x", bottom=False)

    from matplotlib.ticker import LogFormatter

    ax00.set_ylabel("BERT")
    ax10.set_ylabel("RoBERTa")

    ax00.set_xlabel("No Clipping")
    ax00.xaxis.set_label_position("top")
    ax01.set_xlabel("Uncertainty Clipping: 95\%")
    ax01.xaxis.set_label_position("top")

    # remove colorbars
    for axa in [ax00, ax01, ax10]:
        cbar = axa.collections[0].colorbar
        cbar.remove()

    cbar11 = ax11.collections[0].colorbar

    plt.tight_layout()

    plt.subplots_adjust(bottom=0.11, right=0.94, top=0.95)
    cax = plt.axes([0.95, 0, 0.02, 1.0])
    cbar = fig.colorbar(
        ax11.collections[0],
        cax=cax,
    )
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(100, 0))

    cbar11.remove()

    for ax in [ax00, ax01, ax10, ax11]:
        for tick in itertools.chain(ax.get_xticklabels(), ax.get_yticklabels()):
            if tick._text in ["Ent", "LC", "MM"]:
                tick.set_color("#4c9dc9")
                tick.set_text("$\mathit{" + tick._text + "}")
            elif tick._text in ["Rand", "Pass", "Wrong"]:
                tick.set_color("#e18e4c")
                tick.set_text("$\mathit{" + tick._text + "}")

    table_file = Path(f"final/vector_indices_all_datasets.pdf")
    print(table_file)
    plt.savefig(table_file, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close("all")


def full_uncertainty_plots(
    param_grid,
    # metric="confidence_scores",
    metric="y_proba_test_active",
    datasets=["trec6"],  # , "ag_news"],
    strategies=[
        # "Rand (softmax) True/1.0",
        # "passive (softmax) True/1.0",
        "trustscore (softmax) True/1.0",
        "LC (label_smoothing) True/1.0",
        # "LC (evidential1) True/1.0",
    ],
    transformer_model_name="bert-base-uncased",
):
    for dataset in datasets:
        grouped_data = _load_grouped_data(
            exp_name=param_grid["exp_name"][0],
            transformer_model_name=transformer_model_name,
            dataset=dataset,
            initially_labeled_samples=param_grid["initially_labeled_samples"][0],
            batch_size=param_grid["batch_size"][0],
            param_grid=param_grid,
            num_iterations=param_grid["num_iterations"][0],
            metric=metric,
        )
        if len(grouped_data) == 0:
            return
        
        df_data = []
        for k, v in grouped_data.items():
            #if k not in strategies:
            #    continue

            for random_seed in v:
                # print(random_seed)
                for i, iteration in enumerate(random_seed):
                    for v in iteration:
                        if metric != "confidence_scores":
                            v = np.max(v)
                        if v < 0:
                            print(k)
                            # passive classifier
                            v = 1 + v
                        # v = 1 - v
                        df_data.append((k, v, i))
        df = pd.DataFrame(data=df_data, columns=["Strategy", metric, "iteration"])

        for strat in df["Strategy"].unique():
            plot_path = Path(f"./plots/{metric}_{transformer_model_name}_{dataset}")

            #if plot_path != Path("plots/y_proba_test_active_bert-base-uncased_trec6"):
            #    continue
            
            mv = df.loc[df["Strategy"] == strat][metric].astype(np.float16)
            if np.nanmax(mv) == np.inf:
                max_value = np.iinfo(np.int16).max
            else:
                max_value = np.nanmax(mv)
            if np.nanmin(mv) == 0 and max_value == 0:
                continue
            counts, bins = np.histogram(mv, bins=70, range=(np.nanmin(mv), max_value))

            fig = plt.figure(figsize=set_matplotlib_size(width, fraction=0.33))
            plt.hist(
                bins[:-1],
                weights=counts,
                bins=bins,
            )
            # plt.title(f"{strat}")
            plt.title("")
            plot_path.mkdir(exist_ok=True, parents=True)

            plt.savefig(
                plot_path / f"{strat.replace('/', '-')}.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.savefig(
                plot_path / f"{strat.replace('/', '-')}.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.clf()
            plt.close("all")
            for iteration in df["iteration"].unique():
                #if iteration != 5:
                #    continue
                
                plot_path = Path(
                    f"./plots/{metric}_{transformer_model_name}_{dataset}/{strat.replace('/', '-')}/"
                )

                #if plot_path != Path("plots/y_proba_test_active_bert-base-uncased_trec6/passive (softmax) True-1.0"):
                #    continue
                if Path(plot_path/ f"{iteration}.jpg").exists():
                    continue

                mv = df.loc[(df["Strategy"] == strat) & (df["iteration"] == iteration)][
                    metric
                ].astype(np.float16)

                if len(mv) == 0:
                    continue

                if np.nanmax(mv) == np.inf:
                    max_value = np.iinfo(np.int16).max
                else:
                    max_value = np.nanmax(mv)
                if np.nanmin(mv) == 0 and max_value == 0:
                    continue
                counts, bins = np.histogram(
                    mv, bins=70, range=(np.nanmin(mv), max_value)
                )

                fig = plt.figure(figsize=set_matplotlib_size(width, fraction=0.33))
                plt.hist(
                    bins[:-1],
                    weights=counts,
                    bins=bins,
                )
                #  plt.title(f"{strat}: {iteration}")
                plt.title("")
                plt.ylabel("Freq")
                plot_path.mkdir(exist_ok=True, parents=True)

                plt.savefig(
                    plot_path / f"{iteration}.jpg", bbox_inches="tight", pad_inches=0
                )
                plt.savefig(
                    plot_path / f"{iteration}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                print(plot_path / f"{iteration}.jpg")
                plt.clf()
                plt.close("all")

def uncertainty_advanced_clipping_test_plots(
    param_grid,
    metric="confidence_scores",
    #metric="y_proba_test_active",
    datasets=["trec6", "ag_news", "subj", "rotten", "imdb", "cola", "sst2"],
    transformer_model_name="bert-base-uncased",
    bins = 100,
    ignore_clipping_for_random_and_passive=True
):    
    for dataset in datasets:
        for query_strategy in param_grid["query_strategy"]:
            for uncertainty_method in param_grid["uncertainty_method"]:
                for lower_is_better in param_grid["lower_is_better"]:
                    for uncertainty_clipping in param_grid["uncertainty_clipping"]:
                        if uncertainty_clipping != 1.0:
                            continue
                        if (
                            query_strategy in ["passive", "Rand"]
                            and uncertainty_clipping != 1.0
                            and ignore_clipping_for_random_and_passive
                        ):
                            continue
                        elif (
                            query_strategy in ["passive", "Rand"]
                            and uncertainty_clipping != 1.0
                            and not ignore_clipping_for_random_and_passive
                        ):
                            uncertainty_clipping = 1.0

                        for clipping_on_which_data in param_grid["clipping_on_which_data"]:
                            for random_seed in param_grid["random_seed"]:
                                # check if this configuration is available
                                exp_results_dir = _convert_config_to_path(
                                    {
                                        "uncertainty_method": uncertainty_method,
                                        "query_strategy": query_strategy,
                                        "exp_name": param_grid["exp_name"][0],
                                        "transformer_model_name": transformer_model_name,
                                        "dataset": dataset,
                                        "initially_labeled_samples": param_grid["initially_labeled_samples"][0],
                                        "random_seed": random_seed,
                                        "batch_size": param_grid["batch_size"][0],
                                        "num_iterations": param_grid["num_iterations"][0],
                                        "uncertainty_clipping": uncertainty_clipping,
                                        "lower_is_better": lower_is_better,
                                        "clipping_on_which_data": clipping_on_which_data
                                    }
                                )

                                if exp_results_dir.exists():
                                    metrics = np.load(
                                        exp_results_dir / "metrics.npz", allow_pickle=True
                                    )
                                    
                                    metric_values = metrics[metric][1:]


                                    plot_path = Path(
                                        f"./plots/{exp_results_dir.name}/"
                                    )
                                    
                                    for iteration, mv in enumerate(metric_values):
                                        if Path(plot_path/ f"{iteration:02d}.jpg").exists():
                                            continue

                                        if len(mv) == 0:
                                            continue

                                        """if np.nanmax(mv) == np.inf:
                                            max_value = np.iinfo(np.int16).max
                                        else:
                                            max_value = np.nanmax(mv)
                                        """
                                        if np.nanmin(mv) == 0 and max_value == 0:
                                            continue

                                        clipping_threshold95 = np.percentile(
                                            mv, (1 - 0.95) * 100
                                        )

                                        clipping_threshold90 = np.percentile(
                                            mv, (1 - 0.90) * 100
                                        )

                                        """counts, bins = np.histogram(
                                            mv, bins=bins, range=(0,1)#range=(np.nanmin(mv), max_value)
                                        )

                                        
                                        print(mv)
                                        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(mv.reshape(-1,1))
                                        X_plot = np.linspace(-1,1,1000)[:,np.newaxis]
                                        log_dens = kde.score_samples(X_plot)
                                        """

                                        fig,ax = plt.subplots(figsize=set_matplotlib_size(width, fraction=2))

                                        sns_plot = sns.histplot(data=mv, kde=True, element="step", fill=True,ax=ax)

                                        kde = sm.nonparametric.KDEUnivariate(mv).fit()
                                        kde_x, kde_y = (kde.support, kde.density)
                                        
                                        # find leftmost maximum point
                                        # find leftmost minimum point
                                        
                                       
                                        print(kde_y[:100])


                                        # for local maxima
                                        local_maxima = [(kde_x[abcde], kde_y[abcde]) for abcde in argrelmax(kde_y)[0]]

                                        # for local minima
                                        local_minima = [(kde_x[abcde], kde_y[abcde]) for abcde in argrelmin(kde_y)[0]]

                                        print(clipping_threshold95)
                                        print(local_maxima)
                                        print(local_minima)


                                        # calculate using a heuristic where to clip based on extremas
                                        #find leftmost peak, then leftmost minima, and then clip at the minima, or use 5%!

                                        leftmost_peak = local_maxima[0]
                                        next_valley = None

                                        for valley in local_minima:
                                            if valley[0] > leftmost_peak[0]:
                                                next_valley = valley
                                                break
                                        
                                        if next_valley is None:
                                            continue
                                        
                                        plt.axvline(x=leftmost_peak[0], color="yellow")
                                        plt.axvline(x=next_valley[0], color="darkblue")


                                        if leftmost_peak[0] > clipping_threshold95:
                                            continue


                                        #https://stackoverflow.com/questions/19936033/finding-turning-points-of-an-array-in-python
                                        #plt.axvline(x=leftmost_maximum_point[0],color="yellow")
                                        #plt.axvline(x=leftmost_minimum_point_after_maximum[0],color="darkblue")

                                        ax.set_xlim(0,1)
                                        #plt.fill(X_plot[:,0], np.exp(log_dens), fc="#AAAAFF")
                                        plt.axvline(x=clipping_threshold95, color="darkred")
                                        plt.axvline(x=clipping_threshold90, color="cyan")

                                        """plt.hist(
                                            bins[:-1],
                                            weights=counts,
                                            bins=bins,
                                        histtype="step"
                                        )"""
                                        #  plt.title(f"{strat}: {iteration}")
                                        plt.title("")
                                        plt.ylabel("Freq")
                                        plot_path.mkdir(exist_ok=True, parents=True)

                                        plt.savefig(
                                            plot_path / f"{iteration:02d}.jpg", bbox_inches="tight", pad_inches=0, dpi=300
                                        )
                                        plt.savefig(
                                            plot_path / f"{iteration:02d}.pdf",
                                            dpi=300,
                                            bbox_inches="tight",
                                            pad_inches=0,
                                        )
                                        print(plot_path / f"{iteration:02d}.jpg")
                                        plt.clf()
                                        plt.close("all")



def _generate_al_strat_abbreviations_table(pg):
    param_grid = copy.deepcopy(pg)
    param_grid["uncertainty_clipping"] = [1.0]

    data = _load_grouped_data(
        exp_name=param_grid["exp_name"][0],
        transformer_model_name="bert-base-uncased",
        dataset="trec6",
        initially_labeled_samples=param_grid["initially_labeled_samples"][0],
        batch_size=param_grid["batch_size"][0],
        param_grid=param_grid,
        num_iterations=param_grid["num_iterations"][0],
        metric="test_acc",
    )

    strats = []
    for key, _ in data.items():
        strats.append((key, _rename_strat(key)))
    df = pd.DataFrame(strats, columns=["AL strategy", "Abbreviation"])
    print(tabulate(df, headers="keys", showindex=False, tablefmt="latex_booktabs"))


# full_param_grid["dataset"].remove("cola")
# full_param_grid["dataset"].remove("sst2")


#_generate_al_strat_abbreviations_table(full_param_grid)
#full_uncertainty_plots(full_param_grid, metric="confidence_scores")
#full_uncertainty_plots(full_param_grid)
#full_param_grid["clipping_on_which_data"] = ["unlabeled"]
#full_param_grid["clipping_on_which_data"] = ["all"]
full_param_grid["clipping_on_which_data"] = ["all", "unlabeled"]

#full_violinplot(copy.deepcopy(full_param_grid), consider_last_n=5, clipping_method="clipping")
#full_violinplot(copy.deepcopy(full_param_grid), consider_last_n=5, clipping_method="valley_after_peak")
#full_violinplot(copy.deepcopy(full_param_grid), consider_last_n=5, clipping_method="leftmost_peak")
#uncertainty_advanced_clipping_test_plots(full_param_grid)

strat_dataset_improvements_table(full_param_grid, clipping=True,  consider_last_n=5)
exit(-2)
merged_stats_table(full_param_grid, clipping=True,  consider_last_n=5)

full_class_distribution(copy.deepcopy(full_param_grid),clippings=[1.0,"leftmost_peak"], clipping_on_which_data="unlabeled")
full_runtime_stats(copy.deepcopy(full_param_grid), metric="times_elapsed", clipping_method="leftmost_peak", clipping_on_which_data="unlabeled")
full_outlier_comparison(copy.deepcopy(full_param_grid), clipping_method="leftmost_peak", clipping_on_which_data="unlabeled")
full_violinplot(copy.deepcopy(full_param_grid), consider_last_n=5, clipping_method="leftmost_peak")


exit(-1)
#exit(-5)
#full_table_stat(copy.deepcopy(full_param_grid), clipping=False)

#full_reproducability(copy.deepcopy(full_param_grid), clipping=False)
"""full_significance_tests(copy.deepcopy(full_param_grid))
full_significance_tests(
    copy.deepcopy(full_param_grid), transformer_model_name="roberta-base"
)"""
exit(-1)


full_uncertainty_plots(full_param_grid)




# full_violinplot(copy.deepcopy(full_param_grid))

# tables_plots(baselines_param_grid)
# tables_plots(my_methods_param_grid)
# tables_plots(full_param_grid)
# tables_plots(dev_param_grid)
# tables_plots(finetuning_test_grid)
