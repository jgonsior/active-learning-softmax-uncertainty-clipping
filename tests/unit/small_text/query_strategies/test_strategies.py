import unittest

import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.preprocessing import normalize
from unittest.mock import patch, Mock

from small_text.classifiers import ConfidenceEnhancedLinearSVC, SklearnClassifier
from small_text.data.datasets import SklearnDataset
from small_text.query_strategies import EmptyPoolException, PoolExhaustedException
from small_text.query_strategies import (
    BreakingTies,
    ContrastiveActiveLearning,
    RandomSampling,
    SubsamplingQueryStrategy,
    LeastConfidence,
    PredictionEntropy,
    EmbeddingBasedQueryStrategy,
    EmbeddingKMeans,
)


DEFAULT_QUERY_SIZE = 10


def query_random_data(
    strategy, num_samples=100, n=10, use_embeddings=False, embedding_dim=100
):

    x = np.random.rand(num_samples, 10)
    kwargs = dict()

    if use_embeddings:
        kwargs["embeddings"] = np.random.rand(
            SamplingStrategiesTests.DEFAULT_NUM_SAMPLES, embedding_dim
        )

    indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
    indices_unlabeled = np.array(
        [i for i in range(x.shape[0]) if i not in set(indices_labeled)]
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    return strategy.query(None, x, indices_unlabeled, indices_labeled, y, n=n, **kwargs)


class SamplingStrategiesTests(object):

    DEFAULT_NUM_SAMPLES = 100

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        raise NotImplementedError()

    def test_simple_query(self):
        """Tests a query of n=5."""
        indices = self._query(
            self._get_query_strategy(), num_samples=self.DEFAULT_NUM_SAMPLES, n=5
        )
        self.assertEqual(5, len(indices))

    def test_default_query(self):
        """Tests the query with default args."""
        indices = self._query(
            self._get_query_strategy(), num_samples=self.DEFAULT_NUM_SAMPLES
        )
        self.assertEqual(DEFAULT_QUERY_SIZE, len(indices))

    def test_query_takes_remaining_pool(self):
        indices = self._query(
            self._get_query_strategy(), num_samples=self.DEFAULT_NUM_SAMPLES, n=10
        )
        self.assertEqual(DEFAULT_QUERY_SIZE, len(indices))

    def test_query_exhausts_pool(self):
        """Tests for PoolExhaustedException."""
        with self.assertRaises(PoolExhaustedException):
            self._query(self._get_query_strategy(), n=11)

    def _query(self, strategy, num_samples=20, n=10, **kwargs):
        x = np.random.rand(num_samples, 10)

        indices_labeled = np.random.choice(
            np.arange(num_samples), size=10, replace=False
        )
        indices_unlabeled = np.array(
            [i for i in range(x.shape[0]) if i not in set(indices_labeled)]
        )

        clf_mock = self._get_clf()
        if clf_mock is not None:
            proba = np.random.random_sample((num_samples, 2))
            clf_mock.predict_proba = Mock(return_value=proba)

        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(indices_labeled.shape[0], y.shape[0])

        return strategy.query(
            clf_mock, x, indices_unlabeled, indices_labeled, y, n=n, **kwargs
        )


class RandomSamplingTest(unittest.TestCase, SamplingStrategiesTests):
    def _get_clf(self):
        return None

    def _get_query_strategy(self):
        return RandomSampling()

    def test_random_sampling_str(self):
        strategy = RandomSampling()
        self.assertEqual("RandomSampling()", str(strategy))

    def test_random_sampling_query_default(self):
        indices = query_random_data(self._get_query_strategy())
        self.assertEqual(10, len(indices))

    def test_random_sampling_empty_pool(self, num_samples=20, n=10):
        strategy = RandomSampling()

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = []
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with self.assertRaises(EmptyPoolException):
            strategy.query(None, dataset, indices_unlabeled, indices_labeled, y, n=n)


class BreakingTiesTest(unittest.TestCase, SamplingStrategiesTests):
    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return BreakingTies()

    def test_breaking_ties_str(self):
        strategy = self._get_query_strategy()
        self.assertEqual("BreakingTies()", str(strategy))

    def test_breaking_ties_binary(self):
        proba = np.array([[0.1, 0.9], [0.45, 0.55], [0.5, 0.5], [0.7, 0.3]])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        dataset = SklearnDataset(
            np.random.rand(proba.shape[0], 10), np.random.randint(0, high=2, size=10)
        )
        strategy = self._get_query_strategy()
        indices = strategy.query(
            clf_mock, dataset, np.arange(0, 4), np.array([]), np.array([]), n=2
        )

        expected = np.array([2, 1])
        assert_array_equal(expected, indices)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.8, 0.1, 0, 0.4]), strategy.scores_)

    def test_breaking_ties_multiclass(self):
        proba = np.array(
            [[0.1, 0.75, 0.15], [0.45, 0.5, 0.05], [0.1, 0.8, 0.1], [0.7, 0.15, 0.15]]
        )
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        dataset = SklearnDataset(
            np.random.rand(proba.shape[0], 10), np.random.randint(0, high=2, size=10)
        )
        strategy = self._get_query_strategy()
        indices = strategy.query(
            clf_mock, dataset, np.arange(0, 4), np.array([]), np.array([]), n=2
        )

        expected = np.array([1, 3])
        assert_array_equal(expected, indices)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.6, 0.05, 0.7, 0.55]), strategy.scores_)


class LeastConfidenceTest(unittest.TestCase, SamplingStrategiesTests):
    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return LeastConfidence()

    def test_least_confidence_str(self):
        strategy = self._get_query_strategy()
        self.assertEqual("LeastConfidence()", str(strategy))

    def test_least_confidence_binary(self):
        proba = np.array([[0.1, 0.9], [0.45, 0.55], [0.2, 0.8], [0.7, 0.3]])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        dataset = SklearnDataset(
            np.random.rand(proba.shape[0], 10), np.random.randint(0, high=2, size=10)
        )
        strategy = self._get_query_strategy()
        indicies = strategy.query(
            clf_mock, dataset, np.arange(0, 4), np.array([]), np.array([]), n=2
        )

        expected = np.array([1, 3])
        assert_array_equal(expected, indicies)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.9, 0.55, 0.8, 0.7]), strategy.scores_)

    def test_least_confidence_multiclass(self):
        proba = np.array(
            [[0.1, 0.75, 0.15], [0.45, 0.5, 0.05], [0.1, 0.8, 0.1], [0.7, 0.15, 0.15]]
        )
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        dataset = SklearnDataset(
            np.random.rand(proba.shape[0], 10), np.random.randint(0, high=2, size=10)
        )
        strategy = self._get_query_strategy()
        indicies = strategy.query(
            clf_mock, dataset, np.arange(0, 4), np.array([]), np.array([]), n=2
        )

        expected = np.array([1, 3])
        assert_array_equal(expected, indicies)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.75, 0.5, 0.8, 0.7]), strategy.scores_)


class PredictionEntropyTest(unittest.TestCase, SamplingStrategiesTests):
    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return PredictionEntropy()

    def test_prediction_entropy_str(self):
        strategy = self._get_query_strategy()
        self.assertEqual("PredictionEntropy()", str(strategy))

    def test_prediction_entropy_binary(self):
        proba = np.array([[0.1, 0.9], [0.45, 0.55], [0.5, 0.5], [0.7, 0.3]])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        dataset = SklearnDataset(
            np.random.rand(proba.shape[0], 10), np.random.randint(0, high=2, size=10)
        )
        strategy = self._get_query_strategy()
        indicies = strategy.query(
            clf_mock, dataset, np.arange(0, 4), np.array([]), np.array([]), n=2
        )

        expected = np.array([2, 1])
        assert_array_equal(expected, indicies)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(
            np.array([0.325083, 0.688139, 0.693147, 0.610864]), strategy.scores_
        )

    def test_prediction_entropy_multiclass(self):
        proba = np.array(
            [[0.1, 0.8, 0.1], [0.45, 0.25, 0.3], [0.33, 0.33, 0.34], [0.7, 0.3, 0]]
        )
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        dataset = SklearnDataset(
            np.random.rand(proba.shape[0], 10), np.random.randint(0, high=2, size=10)
        )
        strategy = self._get_query_strategy()
        indicies = strategy.query(
            clf_mock, dataset, np.arange(0, 4), np.array([]), np.array([]), n=2
        )

        expected = np.array([2, 1])
        assert_array_equal(expected, indicies)

        assert_array_almost_equal(
            np.array([0.639032, 1.067094, 1.098513, 0.610864]), strategy.scores_
        )


class SubSamplingTest(unittest.TestCase, SamplingStrategiesTests):
    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return SubsamplingQueryStrategy(RandomSampling(), 20)

    def test_subsampling_str(self):
        strategy = SubsamplingQueryStrategy(RandomSampling(), subsample_size=20)
        expected_str = (
            "SubsamplingQueryStrategy(base_query_strategy=RandomSampling(), "
            "subsample_size=20)"
        )
        self.assertEqual(expected_str, str(strategy))

    def test_subsampling_query_default(self):
        indices = query_random_data(self._get_query_strategy())
        self.assertEqual(10, len(indices))

    def test_subsampling_empty_pool(self, num_samples=20, n=10):
        strategy = self._get_query_strategy()

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = []
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with self.assertRaises(EmptyPoolException):
            strategy.query(None, dataset, indices_unlabeled, indices_labeled, y, n=n)

    def test_scores_property(self):
        num_samples = 20
        scores = np.random.rand(num_samples, 1)

        strategy = self._get_query_strategy()

        strategy.base_query_strategy.scores_ = scores
        assert_array_equal(scores, strategy.scores_)

        strategy = self._get_query_strategy()
        self.assertIsNone(strategy.scores_)


class EmbeddingBasedQueryStrategyImplementation(EmbeddingBasedQueryStrategy):
    def sample(
        self,
        clf,
        dataset,
        indices_unlabeled,
        indices_labeled,
        y,
        n,
        embeddings,
        embeddings_proba=None,
    ):
        pass


class SklearnClassifierWithRandomEmbeddings(SklearnClassifier):
    def embed(self, dataset, embed_dim=5, pbar=None):
        self.embeddings_ = np.random.rand(len(dataset), embed_dim)
        return self.embeddings_


class SklearnClassifierWithRandomEmbeddingsAndProba(SklearnClassifier):
    def embed(self, dataset, return_proba=False, embed_dim=5, pbar=None):

        self.embeddings_ = np.random.rand(len(dataset), embed_dim)
        if return_proba:
            self.proba_ = np.random.rand(len(dataset))
            return self.embeddings_, self.proba_

        return self.embeddings_


class EmbeddingBasedQueryStrategyTest(unittest.TestCase):
    def test_str(self):
        query_strategy = EmbeddingBasedQueryStrategyImplementation()
        self.assertEqual("EmbeddingBasedQueryStrategy()", str(query_strategy))

    def test_query_with_precomputed_embeddings(self, num_samples=20):
        clf = SklearnClassifierWithRandomEmbeddingsAndProba(
            ConfidenceEnhancedLinearSVC, 2
        )
        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )
        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices = np.arange(num_samples)
        mask = np.isin(indices, indices_labeled)
        indices_unlabeled = indices[~mask]
        y = np.random.randint(0, 2, size=num_samples)
        n = 10
        embeddings = None

        query_strategy = EmbeddingBasedQueryStrategyImplementation()

        with patch.object(
            query_strategy, "sample", wraps=query_strategy.sample
        ) as sample_spy:
            query_strategy.query(
                clf,
                dataset,
                indices_unlabeled,
                indices_labeled,
                y,
                n=n,
                embeddings=embeddings,
            )

            sample_spy.assert_called()

    def test_query_when_embed_has_return_proba(self, num_samples=20):
        clf = SklearnClassifierWithRandomEmbeddingsAndProba(
            ConfidenceEnhancedLinearSVC, 2
        )
        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )
        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices = np.arange(num_samples)
        mask = np.isin(indices, indices_labeled)
        indices_unlabeled = indices[~mask]
        y = np.random.randint(0, 2, size=num_samples)
        n = 10
        embeddings = None

        query_strategy = EmbeddingBasedQueryStrategyImplementation()

        with patch.object(
            query_strategy, "sample", wraps=query_strategy.sample
        ) as sample_spy:
            query_strategy.query(
                clf,
                dataset,
                indices_unlabeled,
                indices_labeled,
                y,
                n=n,
                embeddings=embeddings,
            )
            sample_spy.assert_called_once_with(
                clf,
                dataset,
                indices_unlabeled,
                indices_labeled,
                y,
                n,
                clf.embeddings_,
                embeddings_proba=clf.proba_,
            )

    def test_query_when_embed_has_no_return_proba(self, num_samples=20):
        clf = SklearnClassifierWithRandomEmbeddings(ConfidenceEnhancedLinearSVC, 2)
        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )
        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices = np.arange(num_samples)
        mask = np.isin(indices, indices_labeled)
        indices_unlabeled = indices[~mask]
        y = np.random.randint(0, 2, size=num_samples)
        n = 10
        embeddings = None

        query_strategy = EmbeddingBasedQueryStrategyImplementation()

        with patch.object(
            query_strategy, "sample", wraps=query_strategy.sample
        ) as sample_spy:
            query_strategy.query(
                clf,
                dataset,
                indices_unlabeled,
                indices_labeled,
                y,
                n=n,
                embeddings=embeddings,
            )
            sample_spy.assert_called_once_with(
                clf, dataset, indices_unlabeled, indices_labeled, y, n, clf.embeddings_
            )

    def test_query_with_nonexistent_embed_kwargs_and_no_return_proba(
        self, num_samples=20
    ):
        clf = SklearnClassifierWithRandomEmbeddings(ConfidenceEnhancedLinearSVC, 2)
        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )
        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices = np.arange(num_samples)
        mask = np.isin(indices, indices_labeled)
        indices_unlabeled = indices[~mask]
        y = np.random.randint(0, 2, size=num_samples)
        n = 10
        embeddings = None

        query_strategy = EmbeddingBasedQueryStrategyImplementation()

        with self.assertRaises(TypeError):
            query_strategy.query(
                clf,
                dataset,
                indices_unlabeled,
                indices_labeled,
                y,
                n=n,
                embeddings=embeddings,
                embed_kwargs={"does": "not exist"},
            )

    def test_query_with_nonexistent_embed_kwargs_and_return_proba(self, num_samples=20):
        clf = SklearnClassifierWithRandomEmbeddingsAndProba(
            ConfidenceEnhancedLinearSVC, 2
        )
        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )
        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices = np.arange(num_samples)
        mask = np.isin(indices, indices_labeled)
        indices_unlabeled = indices[~mask]
        y = np.random.randint(0, 2, size=num_samples)
        n = 10
        embeddings = None

        query_strategy = EmbeddingBasedQueryStrategyImplementation()

        with self.assertRaises(TypeError):
            query_strategy.query(
                clf,
                dataset,
                indices_unlabeled,
                indices_labeled,
                y,
                n=n,
                embeddings=embeddings,
                embed_kwargs={"does": "not exist"},
            )


class EmbeddingKMeansTest(unittest.TestCase):
    def test_query(self, n=10, num_samples=100, num_classes=2):
        query_strategy = EmbeddingKMeans()
        clf = SklearnClassifierWithRandomEmbeddingsAndProba(
            ConfidenceEnhancedLinearSVC, num_classes
        )

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        indices = query_strategy.query(
            clf, dataset, indices_unlabeled, indices_labeled, y, n
        )

        self.assertIsNotNone(indices)
        self.assertEqual(n, indices.shape[0])

    @patch("sklearn.preprocessing.normalize", wraps=normalize)
    def test_sample(self, normalize_mock, n=10, num_samples=100, embedding_dim=60):
        query_strategy = EmbeddingKMeans()
        query_strategy._get_nearest_to_centers_iterative = Mock(
            wraps=query_strategy._get_nearest_to_centers_iterative
        )
        clf = ConfidenceEnhancedLinearSVC()

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        embeddings = np.random.rand(num_samples, embedding_dim)

        # make sure we hit the "default" case
        query_strategy._get_nearest_to_centers = Mock(
            return_value=np.random.choice(indices_unlabeled, 10, replace=False)
        )
        indices = query_strategy.sample(
            clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings
        )
        self.assertIsNotNone(indices)
        self.assertEqual(n, indices.shape[0])

        normalize_mock.assert_called()
        np.testing.assert_array_equal(embeddings, normalize_mock.call_args[0][0])
        query_strategy._get_nearest_to_centers_iterative.assert_not_called()

    @patch("sklearn.preprocessing.normalize", wraps=normalize)
    def test_sample_with_normalize_false(
        self, normalize_mock, n=10, num_samples=100, embedding_dim=20
    ):
        query_strategy = EmbeddingKMeans(normalize=False)
        query_strategy._get_nearest_to_centers_iterative = Mock(
            wraps=query_strategy._get_nearest_to_centers_iterative
        )
        clf = ConfidenceEnhancedLinearSVC()

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        embeddings = np.random.rand(num_samples, embedding_dim)

        indices = query_strategy.sample(
            clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings
        )
        self.assertIsNotNone(indices)
        self.assertEqual(n, indices.shape[0])

        normalize_mock.assert_not_called()

    def test_sample_with_fallback(self, n=10, num_samples=100, embedding_dim=20):
        query_strategy = EmbeddingKMeans()
        query_strategy._get_nearest_to_centers = Mock(return_value=np.zeros(n))
        query_strategy._get_nearest_to_centers_iterative = Mock(
            wraps=query_strategy._get_nearest_to_centers_iterative
        )

        clf = ConfidenceEnhancedLinearSVC()

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        embeddings = np.random.rand(num_samples, embedding_dim)

        indices = query_strategy.sample(
            clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings
        )
        self.assertIsNotNone(indices)
        self.assertEqual(n, indices.shape[0])

        query_strategy._get_nearest_to_centers_iterative.assert_called()

    def test_str(self):
        query_strategy = EmbeddingKMeans()
        self.assertEqual("EmbeddingKMeans(normalize=True)", str(query_strategy))

    def test_str_with_normalize_false(self):
        query_strategy = EmbeddingKMeans(normalize=False)
        self.assertEqual("EmbeddingKMeans(normalize=False)", str(query_strategy))


class ContrastiveActiveLearningTest(unittest.TestCase):
    def test_query(self, n=10, num_samples=100, num_classes=2):
        query_strategy = ContrastiveActiveLearning()
        clf = SklearnClassifierWithRandomEmbeddingsAndProba(
            ConfidenceEnhancedLinearSVC, num_classes
        )

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        indices = query_strategy.query(
            clf, dataset, indices_unlabeled, indices_labeled, y, n
        )

        self.assertIsNotNone(indices)
        self.assertEqual(n, indices.shape[0])

    def test_query_with_precomputed_embeddings(
        self, n=10, num_samples=100, embedding_dim=20
    ):

        query_strategy = ContrastiveActiveLearning()
        clf = SklearnClassifierWithRandomEmbeddings(ConfidenceEnhancedLinearSVC, 2)

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )
        embeddings = np.random.rand(num_samples, embedding_dim)

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with self.assertRaisesRegex(
            ValueError, "Error: embeddings_proba is None. This strategy"
        ):
            query_strategy.query(
                clf,
                dataset,
                indices_unlabeled,
                indices_labeled,
                y,
                n,
                embeddings=embeddings,
            )

    @patch("small_text.query_strategies.strategies.normalize", wraps=normalize)
    def test_sample(self, normalize_mock, n=10, num_samples=100, embedding_dim=60):
        query_strategy = ContrastiveActiveLearning()
        clf = SklearnClassifierWithRandomEmbeddingsAndProba(
            ConfidenceEnhancedLinearSVC, 2
        )

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        embeddings = np.random.rand(num_samples, embedding_dim)
        embeddings_proba = np.random.random_sample((num_samples, 2))

        indices = query_strategy.sample(
            clf,
            dataset,
            indices_unlabeled,
            indices_labeled,
            y,
            n,
            embeddings,
            embeddings_proba=embeddings_proba,
        )
        self.assertIsNotNone(indices)
        self.assertEqual(n, indices.shape[0])

        normalize_mock.assert_called()
        np.testing.assert_array_equal(embeddings, normalize_mock.call_args[0][0])

    @patch("small_text.query_strategies.strategies.normalize", wraps=normalize)
    def test_sample_with_normalize_false(
        self, normalize_mock, n=10, num_samples=100, embedding_dim=20
    ):
        query_strategy = ContrastiveActiveLearning(normalize=False)
        clf = SklearnClassifierWithRandomEmbeddingsAndProba(
            ConfidenceEnhancedLinearSVC, 2
        )

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        embeddings = np.random.rand(num_samples, embedding_dim)
        embeddings_proba = np.random.random_sample((num_samples, 2))

        indices = query_strategy.sample(
            clf,
            dataset,
            indices_unlabeled,
            indices_labeled,
            y,
            n,
            embeddings,
            embeddings_proba=embeddings_proba,
        )
        self.assertIsNotNone(indices)
        self.assertEqual(n, indices.shape[0])

        normalize_mock.assert_not_called()

    def test_sample_with_clf_that_does_not_return_proba(
        self, n=10, num_samples=100, embedding_dim=20
    ):
        query_strategy = ContrastiveActiveLearning()
        clf = ConfidenceEnhancedLinearSVC()

        dataset = SklearnDataset(
            np.random.rand(num_samples, 10), np.random.randint(0, high=2, size=10)
        )

        indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        embeddings = np.random.rand(num_samples, embedding_dim)

        with self.assertRaisesRegex(
            ValueError, "Error: embeddings_proba is None. This strategy"
        ):
            query_strategy.sample(
                clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings
            )

    def test_str(self):
        query_strategy = ContrastiveActiveLearning()
        self.assertEqual(
            "ContrastiveActiveLearning(k=10, embed_kwargs={}, normalize=True)",
            str(query_strategy),
        )

    def test_str_with_normalize_false(self):
        query_strategy = ContrastiveActiveLearning(normalize=False)
        self.assertEqual(
            "ContrastiveActiveLearning(k=10, embed_kwargs={}, normalize=False)",
            str(query_strategy),
        )
