"""Example of a binary active learning text classification.
"""
from collections import OrderedDict
import gc
import json
from pathlib import Path
import random
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import logging

from small_text.active_learner import PoolBasedActiveLearner
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory
from small_text.initialization.strategies import random_initialization_stratified
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling
from timeit import default_timer as timer


from small_text.active_learner import PoolBasedActiveLearner

from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import (
    UncertaintyBasedClassificationFactory,
)
from small_text.query_strategies import (
    BreakingTies,
    RandomSampling,
    PredictionEntropy,
    LeastConfidence,
    QBC_VE,
    QBC_KLD,
    Trustscore2,
    TemperatureScalingStrat,
)

from small_text.integrations.transformers import TransformerModelArguments


from dataset_loader import load_my_dataset


def main(
    num_iterations: int,
    batch_size: int,
    dataset: str,
    transformer_model_name: str,
    initially_labeled_samples: int,
    query_strategy_name: str,
    uncertainty_method: str,
    gpu_device: int,
    lower_is_better: bool,
    uncertainty_clipping: float,
):
    train, test, num_classes = load_my_dataset(dataset, transformer_model_name)

    cpu_cuda = "cpu"
    if torch.cuda.is_available():
        cpu_cuda = "cuda:" + str(gpu_device)
        print(f"cuda available, using {cpu_cuda}")

    transformer_model = TransformerModelArguments(transformer_model_name)
    clf_factory = UncertaintyBasedClassificationFactory(
        transformer_model,
        num_classes,
        uncertainty_method=uncertainty_method,
        kwargs=dict(
            {
                "device": cpu_cuda,
                "mini_batch_size": 64,
                "class_weight": "balanced",
            }
        ),
    )

    query_strategy: ConfidenceBasedQueryStrategy

    if query_strategy_name == "LC":
        query_strategy = LeastConfidence(
            lower_is_better=lower_is_better, uncertainty_clipping=uncertainty_clipping
        )
    elif query_strategy_name == "Rand":
        query_strategy = RandomSampling()
    elif query_strategy_name == "Ent":
        query_strategy = PredictionEntropy(
            lower_is_better=lower_is_better, uncertainty_clipping=uncertainty_clipping
        )
    elif query_strategy_name == "MM":
        query_strategy = BreakingTies(
            lower_is_better=lower_is_better, uncertainty_clipping=uncertainty_clipping
        )
    elif query_strategy_name == "QBC_VE":
        query_strategy = QBC_VE(
            lower_is_better=lower_is_better,
            uncertainty_clipping=uncertainty_clipping,
            clf_factory=clf_factory,
        )
    elif query_strategy_name == "QBC_KLD":
        query_strategy = QBC_KLD(
            lower_is_better=lower_is_better,
            uncertainty_clipping=uncertainty_clipping,
            clf_factory=clf_factory,
        )
    elif query_strategy_name == "trustscore":
        query_strategy = Trustscore2(
            uncertainty_clipping=uncertainty_clipping,
        )
    # error: can't optimize a non-leaf Tensor
    """elif query_strategy_name == "temp_scaling2":
        query_strategy = TemperatureScalingStrat(
            uncertainty_clipping=uncertainty_clipping,
            clf_factory=clf_factory,
        )"""

        # "model_calibration",
        # "bayesian",
        # "evidential2",

    else:
        print("Query Strategy not found")
        exit(-1)

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)

    active_learner.query_strategy.predict_proba_with_labeled_data = False

    labeled_indices = initialize_active_learner(
        active_learner, train.y, initially_labeled_samples
    )

    if query_strategy_name == "trustscore":
        query_strategy._clsUNLABELED = active_learner._clf.embed(
            train, embedding_method="cls"
        )
    try:
        return perform_active_learning(
            active_learner, train, labeled_indices, test, num_iterations, batch_size
        )
    except PoolExhaustedException:
        print("Error! Not enough samples left to handle the query.")
    except EmptyPoolException:
        print("Error! No more samples left. (Unlabeled pool is empty)")


def _evaluate(active_learner, train, test):
    y_pred_train = active_learner.classifier.predict(train)
    y_proba_train = active_learner.classifier.predict_proba(train)

    y_pred_test = active_learner.classifier.predict(test)
    y_proba_test = active_learner.classifier.predict_proba(test)

    test_acc = accuracy_score(y_pred_test, test.y)
    train_acc = accuracy_score(y_pred_train, train.y)
    test_ece, acc_bins_test, proba_bins_test = _expected_calibration_error(
        y_pred_test, y_proba_test, test.y
    )
    train_ece, acc_bins_train, proba_bins_train = _expected_calibration_error(
        y_pred_train, y_proba_train, train.y
    )

    print(f"Train acc: {train_acc}")
    print(f"Test acc: {test_acc}")
    print(f"Test ece: {train_ece}")
    print(f"Test ece: {test_ece}")

    print("---")
    return (
        train_acc,
        test_acc,
        train_ece,
        test_ece,
        y_proba_test,
        y_proba_train,
        acc_bins_train,
        proba_bins_train,
        acc_bins_test,
        proba_bins_test,
    )


def _expected_calibration_error(y_pred, probas, y_true, n_bins=10):
    proba = np.amax(probas, axis=1)

    intervals = np.linspace(0, 1, n_bins + 1)
    accuracy = (y_pred == y_true).astype(int)

    num_predictions = y_pred.shape[0]
    error = 0

    acc_bins = []
    proba_bins = []

    for lower, upper in zip(intervals[:-1], intervals[1:]):
        mask = np.logical_and(proba > lower, proba <= upper)
        bin_size = mask.sum()
        if bin_size > 0:
            proba_bin_mean = proba[mask].mean()
            acc_bin_mean = accuracy[mask].mean()

            error += bin_size / num_predictions * np.abs(acc_bin_mean - proba_bin_mean)

            acc_bins.append(acc_bin_mean)
            proba_bins.append(proba_bin_mean)
        else:
            acc_bins.append(np.NaN)
            proba_bins.append(np.NaN)

    return error, acc_bins, proba_bins


def perform_active_learning(
    active_learner, train, indices_labeled, test, num_iterations, batch_size
):
    test_accs = []
    train_accs = []
    test_eces = []
    train_eces = []
    y_probas_test = []
    y_probas_train = []
    times_elapsed = []
    times_elapsed_model = []
    queried_indices = []
    acc_binss_train = []
    proba_binss_train = []
    acc_binss_test = []
    proba_binss_test = []
    confidence_scores = []

    # calculate passive accuracy before
    print("Initial Performance")
    start = timer()

    (
        train_acc,
        test_acc,
        train_ece,
        test_ece,
        y_proba_test,
        y_proba_train,
        acc_bins_train,
        proba_bins_train,
        acc_bins_test,
        proba_bins_test,
    ) = _evaluate(active_learner, train[indices_labeled], test)
    end = timer()

    time_elapsed = end - start

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    train_eces.append(train_ece)
    test_eces.append(test_ece)
    y_probas_test.append(y_proba_test)
    y_probas_train.append(y_proba_train)
    times_elapsed.append(time_elapsed)
    times_elapsed_model.append(0)
    queried_indices.append(indices_labeled)
    acc_binss_train.append(acc_bins_train)
    acc_binss_test.append(acc_bins_test)
    proba_binss_test.append(proba_bins_test)
    proba_binss_train.append(proba_bins_train)
    confidence_scores.append(np.empty(1))

    for i in range(num_iterations):
        # free memory
        torch.cuda.empty_cache()
        gc.collect()

        start = timer()

        indices_queried = active_learner.query(num_samples=batch_size, save_scores=True)
        end = timer()

        time_elapsed = end - start

        confidence_scores.append(active_learner.last_scores)

        y = train.y[indices_queried]

        start = timer()
        active_learner.update(y)
        end = timer()
        time_elapsed_model = end - start

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print(
            "Iteration #{:d} ({} samples {})".format(
                i, len(indices_labeled), time_elapsed
            )
        )
        (
            train_acc,
            test_acc,
            train_ece,
            test_ece,
            y_proba_test,
            y_proba_train,
            acc_bins_train,
            proba_bins_train,
            acc_bins_test,
            proba_bins_test,
        ) = _evaluate(active_learner, train[indices_labeled], test)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_eces.append(train_ece)
        test_eces.append(test_ece)
        y_probas_test.append(y_proba_test)
        y_probas_train.append(y_proba_train)
        times_elapsed.append(time_elapsed)
        times_elapsed_model.append(time_elapsed_model)
        queried_indices.append(indices_queried)
        acc_binss_train.append(acc_bins_train)
        acc_binss_test.append(acc_bins_test)
        proba_binss_test.append(proba_bins_test)
        proba_binss_train.append(proba_bins_train)

    return (
        train_accs,
        test_accs,
        train_eces,
        test_eces,
        y_probas_train,
        y_probas_test,
        times_elapsed,
        times_elapsed_model,
        queried_indices,
        acc_binss_train,
        proba_binss_train,
        acc_binss_test,
        proba_binss_test,
        confidence_scores,
    )


def initialize_active_learner(active_learner, y_train, initially_labeled_samples: int):
    indices_initial = random_initialization_stratified(
        y_train, n_samples=initially_labeled_samples
    )
    active_learner.initialize_data(indices_initial, y_train[indices_initial])
    print(indices_initial)
    return indices_initial


if __name__ == "__main__":
    import argparse

    logger = logging.getLogger()
    # logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="An example that shows active learning "
        "for binary text classification."
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="number of active learning iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--initially_labeled_samples",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ag_news", "trec6", "subj", "rotten", "imdb"],
        default="20newsgroups",
    )
    parser.add_argument(
        "--transformer_model_name",
        type=str,
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--query_strategy",
        type=str,
        default="LC",
        choices=[
            "LC",
            "MM",
            "Ent",
            "Rand",
            "QBC_KLD",
            "QBC_VE",
            "trustscore",
            "model_calibration",
            "bayesian",
            "evidential2",
            "temp_scaling2",
        ],
    )

    parser.add_argument(
        "--lower_is_better", type=str, default="True", choices=["True", "False"]
    )

    parser.add_argument("--uncertainty_clipping", type=float, default=1.0)

    parser.add_argument(
        "--uncertainty_method",
        type=str,
        default="softmax",
        choices=[
            "softmax",
            "temp_scaling",
            # "temp_scaling2",
            "label_smoothing",
            "MonteCarlo",
            "inhibited",
            "evidential1",
            # "evidential2",
            # "bayesian",
            # "model_calibration",
        ],
    )
    parser.add_argument("--gpu_device", type=int, choices=[0, 1])

    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))

    # set random seed
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    arg_dict = OrderedDict(sorted(vars(args).items(), key=lambda t: t[0]))
    del arg_dict["gpu_device"]

    exp_results_dir = Path(
        "exp_results/" + "-".join([str(a) for a in arg_dict.values()])
    )
    exp_results_dir_args = Path(exp_results_dir / "args.json")
    exp_results_dir_metrics = Path(exp_results_dir / "metrics.npz")
    print(exp_results_dir_metrics)
    if exp_results_dir_metrics.exists():
        print(arg_dict)
        print(exp_results_dir_metrics)
        print("Experiment has already been run, exiting!")
        exit(0)

    if args.lower_is_better == "True":
        args.lower_is_better = True
    else:
        args.lower_is_better = False

    (
        train_accs,
        test_accs,
        train_eces,
        test_eces,
        y_probas_train,
        y_probas_test,
        times_elapsed,
        times_elapsed_model,
        queried_indices,
        acc_binss_train,
        proba_binss_train,
        acc_binss_test,
        proba_binss_test,
        confidence_scores,
    ) = main(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        dataset=args.dataset,
        transformer_model_name=args.transformer_model_name,
        initially_labeled_samples=args.initially_labeled_samples,
        query_strategy_name=args.query_strategy,
        uncertainty_method=args.uncertainty_method,
        gpu_device=args.gpu_device,
        lower_is_better=args.lower_is_better,
        uncertainty_clipping=args.uncertainty_clipping,
    )

    # create exp_results_dir
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    # save args
    exp_results_dir_args.write_text(
        json.dumps(
            {
                "args": vars(args),
            },
            indent=4,
        )
    )

    np.savez_compressed(
        exp_results_dir_metrics,
        train_accs=train_accs,
        test_accs=test_accs,
        train_eces=train_eces,
        test_eces=test_eces,
        y_probas_train=y_probas_train,
        y_probas_test=y_probas_test,
        times_elapsed=times_elapsed,
        times_elapsed_model=times_elapsed_model,
        queried_indices=queried_indices,
        acc_bins_train=acc_binss_train,
        proba_bins_train=proba_binss_train,
        acc_bins_test=acc_binss_test,
        proba_bins_test=proba_binss_test,
        confidence_scores=confidence_scores,
    )
