"""Example of a binary active learning text classification.
"""
import numpy as np

from small_text.active_learner import PoolBasedActiveLearner
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling

from examplecode.data.example_data_binary import get_train_test, preprocess_data
from examplecode.shared import evaluate


def main():
    # Prepare some data: The data is a 2-class subset of 20news (baseball vs. hockey)
    text_train, text_test = get_train_test()
    train, test = preprocess_data(text_train, text_test)
    num_classes = 2

    # Active learning parameters
    clf_template = ConfidenceEnhancedLinearSVC()
    clf_factory = SklearnClassifierFactory(clf_template, num_classes)
    query_strategy = RandomSampling()

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    labeled_indices = initialize_active_learner(active_learner, train.y)

    try:
        perform_active_learning(active_learner, train, labeled_indices, test)
    except PoolExhaustedException:
        print("Error! Not enough samples left to handle the query.")
    except EmptyPoolException:
        print("Error! No more samples left. (Unlabeled pool is empty)")


def perform_active_learning(active_learner, train, indices_labeled, test):
    """
    This is the main loop in which we perform 10 iterations of active learning.
    During each iteration 20 samples are queried and then updated.

    The update step reveals the true label to the active learner, i.e. this is a simulation,
    but in a real scenario the user input would be passed to the update function.
    """
    # Perform 10 iterations of active learning...
    for i in range(10):
        # ...where each iteration consists of labelling 20 samples
        indices_queried = active_learner.query(num_samples=20)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print("Iteration #{:d} ({} samples)".format(i, len(indices_labeled)))
        evaluate(active_learner, train[indices_labeled], test)


def initialize_active_learner(active_learner, y_train):
    # Initialize the model. This is required for model-based query strategies.
    indices_pos_label = np.where(y_train == 1)[0]
    indices_neg_label = np.where(y_train == 0)[0]

    indices_initial = np.concatenate(
        [
            np.random.choice(indices_pos_label, 10, replace=False),
            np.random.choice(indices_neg_label, 10, replace=False),
        ]
    )

    indices_initial = indices_initial.astype(int)
    y_initial = np.array([y_train[i] for i in indices_initial])

    active_learner.initialize_data(indices_initial, y_initial)

    return indices_initial


if __name__ == "__main__":
    main()
