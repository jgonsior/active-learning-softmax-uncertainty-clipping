"""Example of a binary active learning text classification.
"""
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
    # DeepEnsemble,
    # TrustScore2,
    # TrustScore,
    # EvidentialConfidence2,
    # BT_Temp,
    # TemperatureScaling,
    BreakingTies,
    RandomSampling,
    PredictionEntropy,
    # FalkenbergConfidence2,
    # FalkenbergConfidence,
    LeastConfidence,
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
):
    train, test, num_classes = load_my_dataset(dataset, transformer_model_name)

    cpu_cuda = "cpu"
    if torch.cuda.is_available():
        cpu_cuda = "cuda"
        print("cuda available")

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

    query_strategy: QueryStrategy

    match query_strategy_name:
        case "LC":
            query_strategy = LeastConfidence()
        case "Rand":
            query_strategy = RandomSampling()
        case "Ent":
            query_strategy = PredictionEntropy()
        case "MM":
            query_strategy = BreakingTies()
        case _:
            print("Query Stategy not found")
            exit(-1)
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    labeled_indices = initialize_active_learner(
        active_learner, train.y, initially_labeled_samples
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
    y_pred_test = active_learner.classifier.predict(test)
    y_proba_test = active_learner.classifier.predict_proba(test)
    y_proba_train = active_learner.classifier.predict_proba(train)

    test_acc = accuracy_score(y_pred_test, test.y)
    train_acc = accuracy_score(y_pred_train, train.y)
    test_ece = _expected_calibration_error(y_pred_test, y_proba_test, test.y)
    train_ece = _expected_calibration_error(y_pred_train, y_proba_train, train.y)

    print(f"Train acc: {train_acc}")
    print(f"Test acc: {test_acc}")
    print(f"Test ece: {train_ece}")
    print(f"Test ece: {test_ece}")

    print("---")
    return (train_acc, test_acc, train_ece, test_ece)


def _expected_calibration_error(y_pred, probas, y_true, n_bins=10):
    proba = np.amax(probas, axis=1)

    intervals = np.linspace(0, 1, n_bins + 1)
    accuracy = (y_pred == y_true).astype(int)

    num_predictions = y_pred.shape[0]
    error = 0

    for lower, upper in zip(intervals[:-1], intervals[1:]):
        mask = np.logical_and(proba > lower, proba <= upper)
        bin_size = mask.sum()
        if bin_size > 0:
            proba_bin_mean = proba[mask].mean()
            acc_bin_mean = accuracy[mask].mean()

            error += bin_size / num_predictions * np.abs(acc_bin_mean - proba_bin_mean)

    return error


def perform_active_learning(
    active_learner, train, indices_labeled, test, num_iterations, batch_size
):
    test_accs = []
    train_accs = []
    test_eces = []
    train_eces = []
    times_elapsed = []
    times_elapsed_model = []

    # calculate passive accuracy before
    print("Initial Performance")
    start = timer()
    train_acc, test_acc, train_ece, test_ece = _evaluate(
        active_learner, train[indices_labeled], test
    )
    end = timer()

    time_elapsed = end - start

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    train_eces.append(train_ece)
    test_eces.append(test_ece)
    times_elapsed.append(time_elapsed)
    times_elapsed_model.append(0)

    for i in range(num_iterations):
        start = timer()
        indices_queried = active_learner.query(num_samples=batch_size)
        end = timer()

        time_elapsed = end - start

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
        train_acc, test_acc, train_ece, test_ece = _evaluate(
            active_learner, train[indices_labeled], test
        )

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_eces.append(train_ece)
        test_eces.append(test_ece)
        times_elapsed.append(time_elapsed)
        times_elapsed_model.append(time_elapsed_model)

    return train_accs, test_accs, times_elapsed, times_elapsed_model


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
        choices=["LC", "MM", "Ent", "Rand"],
    )
    parser.add_argument(
        "--uncertainty_method",
        type=str,
        default="softmax",
        choices=[
            "softmax",
            "temp_scaling",
            "label_smoothing",
            "MonteCarlo",
            "inhibited",
            "evidential1",
            "evidential2",
            "bayesian",
            "ensembles",
            "trustscore",
            "model_calibration",
        ],
    )

    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))

    # set random seed
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    exp_results_dir = Path(
        "exp_results/" + "-".join([str(a) for a in vars(args).values()])
    )
    exp_results_dir_data = Path(exp_results_dir / "data.json")

    if exp_results_dir_data.exists():
        print("Experiment has already been run, exiting!")
        exit(0)

    train_accs, test_accs, times_elapsed, times_elapsed_model = main(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        dataset=args.dataset,
        transformer_model_name=args.transformer_model_name,
        initially_labeled_samples=args.initially_labeled_samples,
        query_strategy_name=args.query_strategy,
        uncertainty_method=args.uncertainty_method,
    )

    # create exp_results_dir
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    # save args
    exp_results_dir_data.write_text(
        json.dumps(
            {
                "args": vars(args),
                "train_accs": train_accs,
                "test_accs": test_accs,
                "times_elapsed_query_strategy": times_elapsed,
                "times_elapsed_model": times_elapsed_model,
            },
            indent=4,
        )
    )
