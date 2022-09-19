from ast import FormattedValue
from collections import OrderedDict, defaultdict
from copy import deepcopy
import copy
import enum
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from joblib import Parallel, delayed, parallel_backend
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score

from run_experiment import (
    full_param_grid,
    dev_param_grid,
    baselines_param_grid,
    my_methods_param_grid,
    generate_workload,
    large_test_grid,
    finetuning_test_grid,
)
from tabulate import tabulate
import pandas as pd
import seaborn as sns

font_size = 6

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
    "figure.autolayout": True,
}

sns.set_style("white")
sns.set_context("paper")
plt.rcParams.update(tex_fonts)  # type: ignore


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
    for (strat_a, strat_b) in itertools.combinations(grouped_data.keys(), 2):
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
        f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.jpg"
    )
    plt.savefig(
        f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.pdf",
        dpi=300,
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
        counts, bins = np.histogram(df.loc[df["Strategy"] == strat][metric])

        fig = plt.figure(figsize=set_matplotlib_size(width, fraction=1.0))
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

        plt.savefig(plot_path / f"{strat.replace('/', '-')}.jpg")
        plt.savefig(plot_path / f"{strat.replace('/', '-')}.pdf", dpi=300)
        plt.clf()
        plt.close("all")

        for iteration in df["iteration"].unique():
            counts, bins = np.histogram(
                df.loc[(df["Strategy"] == strat) & (df["iteration"] == iteration)][
                    metric
                ]
            )

            fig = plt.figure(figsize=set_matplotlib_size(width, fraction=1.0))
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

            plt.savefig(plot_path / f"{iteration}.jpg")
            plt.savefig(plot_path / f"{iteration}.pdf", dpi=300)
            plt.clf()
            plt.close("all")


def _convert_config_to_path(config_dict) -> Path:
    params = OrderedDict(sorted(config_dict.items(), key=lambda t: t[0]))

    exp_results_dir = Path("exp_results/" + "-".join([str(a) for a in params.values()]))
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
):
    grouped_data = {}
    for query_strategy in param_grid["query_strategy"]:
        for uncertainty_method in param_grid["uncertainty_method"]:
            for lower_is_better in param_grid["lower_is_better"]:
                for uncertainty_clipping in param_grid["uncertainty_clipping"]:
                    if (
                        query_strategy in ["passive", "Rand"]
                        and uncertainty_clipping != 1.0
                    ):
                        continue
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
            f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.jpg"
        )
        plt.savefig(
            f"plots/{metric}_{exp_name}_{transformer_model_name}_{dataset}_{initially_labeled_samples}_{batch_size}_{num_iterations}.pdf",
            dpi=300,
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
        dataset2 = "TREC6"
    elif dataset == "ag_news":
        dataset2 = "AGN"
    elif dataset == "subj":
        dataset2 = "SUBJ"
    elif dataset == "rotten":
        dataset2 = "RT"
    elif dataset == "imdb":
        dataset2 = "IMDB"
    return dataset2


def _rename_strat(strategy, clipping=True):
    strategy = strategy.replace("trustscore (softmax) True/1.0", "LC (trustscore)")
    # rename strategies
    strategy = strategy.replace("True/", "")
    strategy = strategy.replace("MM (softmax)", "MM")
    strategy = strategy.replace("LC (softmax)", "LC")
    strategy = strategy.replace("Ent (softmax)", "Ent")
    strategy = strategy.replace("Rand (softmax)", "Rand")
    strategy = strategy.replace("passive (softmax)", "Passive")

    strategy = strategy.replace("QBC_VE (softmax)", "LC (ensemble, VE)")
    strategy = strategy.replace("QBC_KLD (softmax)", "LC (ensemble, KLD)")

    strategy = strategy.replace("LC (evidential1)", "LC (evidential)")

    strategy = strategy.replace("LC (label_smoothing)", "LC (label smoothing)")

    if clipping:
        strategy = strategy.replace(" 1.0", "")
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

                        for (dataset, group) in groups:
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
                            plot_file,
                            dpi=300,
                        )
                        # plt.show()
                        plt.clf()
                        plt.close("all")


def full_table_stat(pg, clipping=True, metric="test_acc", consider_last_n=21):
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

                        table_data = {}

                        for (dataset, group) in groups:
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
                            table_data[k]["Mean"] = f"{0:.2f}".format(
                                [float(x[:5]) for x in v.values()]
                            )

                        df = pd.DataFrame(table_data)
                        df = df.T
                        df.reset_index(inplace=True)
                        df = df.rename(columns={"index": "Method"})
                        df.sort_values(by="Mean", inplace=True)

                        df["Method"] = df["Method"].apply(
                            lambda x: _rename_strat(x, clipping=clipping)
                        )

                        # df = df.set_index("Method")
                        print(df)

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


def show_values_on_bars(axs, h_v="v", space=4.0, xlim_additional=0):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center")
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


def full_runtime_stats(pg, clipping=True, metric="times_elapsed", consider_last_n=21):
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
                        for (dataset, group) in groups:
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
                            table_file,
                            dpi=300,
                        )
                        # plt.show()
                        plt.clf()
                        plt.close("all")


full_boxplot(full_param_grid, clipping=False)
exit(-1)
full_boxplot(full_param_grid)
exit(-1)
full_runtime_stats(full_param_grid)
full_table_stat(full_param_grid, clipping=False)

# tables_plots(baselines_param_grid)
# tables_plots(my_methods_param_grid)
# tables_plots(full_param_grid)
# tables_plots(dev_param_grid)
# tables_plots(finetuning_test_grid)
