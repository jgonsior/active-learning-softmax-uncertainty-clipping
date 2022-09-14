from abc import ABC, abstractmethod
import abc
import collections
import random

import numpy as np
from scipy.stats import entropy
from sklearn import ensemble
from sklearn.preprocessing import normalize
import torch

from small_text.query_strategies.exceptions import (
    EmptyPoolException,
    PoolExhaustedException,
)


class QueryStrategy(ABC):
    """Abstract base class for Query Strategies."""

    @abstractmethod
    def query(
        self,
        clf,
        dataset,
        indices_unlabeled,
        indices_labeled,
        y,
        n=10,
        save_scores=False,
    ):
        """
        Queries instances from the unlabeled pool.

        A query selects instances from the unlabeled pool.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.
        n : int
            Number of samples to query.

        Returns
        -------
        indices : numpy.ndarray
            Indices relative to `dataset` which were selected.
        """
        pass

    @staticmethod
    def _validate_query_input(indices_unlabeled, n):
        if len(indices_unlabeled) == 0:
            raise EmptyPoolException(
                "No unlabeled indices available. Cannot query an empty pool."
            )

        if n > len(indices_unlabeled):
            raise PoolExhaustedException(
                "Pool exhausted: {} available / {} requested".format(
                    len(indices_unlabeled), n
                )
            )


class RandomSampling(QueryStrategy):
    """Randomly selects instances."""

    def query(
        self,
        clf,
        _dataset,
        indices_unlabeled,
        indices_labeled,
        y,
        n=10,
        save_scores=False,
    ):
        self._validate_query_input(indices_unlabeled, n)
        return np.random.choice(indices_unlabeled, size=n, replace=False)

    def __str__(self):
        return "RandomSampling()"


class ConfidenceBasedQueryStrategy(QueryStrategy):
    """A base class for confidence-based querying.

    To use this class, create a subclass and implement `get_confidence()`.
    """

    def __init__(self, lower_is_better=False, uncertainty_clipping=1.0):
        self.lower_is_better = lower_is_better
        self.scores_ = None
        self.uncertainty_clipping = uncertainty_clipping

    def query(
        self,
        clf,
        dataset,
        indices_unlabeled,
        indices_labeled,
        y,
        n=10,
        save_scores=False,
    ):
        self.save_scores = save_scores
        self.last_scores = None
        self._validate_query_input(indices_unlabeled, n)

        confidence = self.score(clf, dataset, indices_unlabeled, indices_labeled, y)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_partitioned = np.argpartition(confidence[indices_unlabeled], n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_partitioned])

    def score(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        """Assigns a confidence score to each instance.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.

        Returns
        -------
        confidence : np.ndarray[float]
            Array of confidence scores in the shape (n_samples, n_classes).
            If `self.lower_is_better` the confiden values are flipped to negative so that
            subsequent methods do not need to differentiate maximization/minimization.
        """

        confidence = self.get_confidence(
            clf, dataset, indices_unlabeled, indices_labeled, y
        )

        # TODO: uncertainty clipping Z301-304, war früher in "get_confidence" klasse
        """         
        proba = np.amax(proba, axis=1)

        
        wert = np.percentile(proba, 5) # schwelle ab den 10% - 90% ist
        for x in range(len(proba)):
            if proba[x] < wert:
                proba[x] = 1 #lower score bekommen hohen wert
        """

        self.scores_ = confidence
        if not self.lower_is_better:
            confidence = -confidence

        if self.save_scores:
            self.last_scores = confidence
        return confidence

    @abstractmethod
    def get_confidence(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        """Computes a confidence score for each of the given instances.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.
        Returns
        -------
        confidence : ndarray[float]
            Array of confidence scores in the shape (n_samples, n_classes).
        """
        pass

    def __str__(self):
        return "ConfidenceBasedQueryStrategy()"


class BreakingTies(ConfidenceBasedQueryStrategy):
    """Selects instances which have a small margin between their most likely and second
    most likely predicted class [LUO05]_.
    """

    def __init__(self, lower_is_better=True, uncertainty_clipping=1.0):
        super().__init__(
            lower_is_better=lower_is_better, uncertainty_clipping=uncertainty_clipping
        )

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        if self.predict_proba_with_labeled_data:
            clf.tell_me_so_far_labeled_data(
                X=dataset[_indices_labeled], Y=_y[_indices_labeled]
            )
        proba = clf.predict_proba(dataset)
        return np.apply_along_axis(lambda x: self._best_versus_second_best(x), 1, proba)

    @staticmethod
    def _best_versus_second_best(proba):
        ind = np.argsort(proba)
        return proba[ind[-1]] - proba[ind[-2]]

    def __str__(self):
        return "BreakingTies()"


class LeastConfidence(ConfidenceBasedQueryStrategy):
    """Selects instances with the least prediction confidence (regarding the most likely class)
    [LG94]_."""

    def __init__(self, lower_is_better=True, uncertainty_clipping=1.0):
        super().__init__(
            lower_is_better=lower_is_better, uncertainty_clipping=uncertainty_clipping
        )

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        if self.predict_proba_with_labeled_data:
            clf.tell_me_so_far_labeled_data(X=dataset.x, Y=_y)
        proba = clf.predict_proba(dataset)

        return np.amax(proba, axis=1)

    def __str__(self):
        return "LeastConfidence()"


class PredictionEntropy(ConfidenceBasedQueryStrategy):
    """Selects instances with the largest prediction entropy [HOL08]_."""

    def __init__(self, lower_is_better=True, uncertainty_clipping=1.0):
        super().__init__(
            lower_is_better=lower_is_better, uncertainty_clipping=uncertainty_clipping
        )

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        if self.predict_proba_with_labeled_data:
            clf.tell_me_so_far_labeled_data(
                X=dataset[_indices_labeled], Y=_y[_indices_labeled]
            )
        proba = clf.predict_proba(dataset)
        return np.apply_along_axis(lambda x: entropy(x), 1, proba)

    def __str__(self):
        return "PredictionEntropy()"


class QBC_Base(ConfidenceBasedQueryStrategy):
    def __init__(
        self,
        clf_factory,
        lower_is_better=True,
        uncertainty_clipping=1.0,
        amount_of_ensembles=5,
    ):
        super().__init__(
            lower_is_better=lower_is_better, uncertainty_clipping=uncertainty_clipping
        )
        self.amount_of_ensembles = amount_of_ensembles
        self.factory = clf_factory

    @abc.abstractmethod
    def calculate_vote(self, ensemble_probas):
        raise NotImplementedError

    @abc.abstractmethod
    def get_probas(self, clf, dataset):
        raise NotImplementedError

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        ensemble_probas = [self.get_probas(clf, dataset)]

        current_seed = random.randint(0, 1000000)

        for i in range(self.amount_of_ensembles - 1):
            seed = current_seed + i
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            new_classifier = self.factory.new()
            new_classifier.fit(dataset[_indices_labeled])
            new_probas = self.get_probas(new_classifier, dataset)
            ensemble_probas.append(new_probas)

            del new_classifier

        seed = current_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        proba = self.calculate_vote(ensemble_probas)

        return np.array(proba)

    def __str__(self):
        return "PredictionEntropy()"


class QBC_KLD(QBC_Base):
    def get_probas(self, clf, dataset):
        return clf.predict_proba(dataset)

    # source: alipy
    def calculate_vote(self, ensemble_probas):

        """Calculate the average Kullback-Leibler (KL) divergence for measuring the
        level of disagreement in QBC.
        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.
        Returns
        -------
        score: list
            Score for each instance. Shape [n_samples]
        References
        ----------
        [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning for
            text classification. In Proceedings of the International Conference on Machine
            Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
        """
        score = []

        committee_size = len(ensemble_probas)
        input_shape = np.shape(ensemble_probas[0])
        if len(input_shape) == 2:
            label_num = input_shape[1]
            # calc kl div for each instance
            for i in range(input_shape[0]):
                instance_mat = np.array(
                    [X[i, :] for X in ensemble_probas if X is not None]
                )
                tmp = 0
                # calc each label
                for lab in range(label_num):
                    committee_consensus = (
                        np.sum(instance_mat[:, lab]) / committee_size + 1e-9
                    )
                    for committee in range(committee_size):
                        tmp += instance_mat[committee, lab] * np.log(
                            (instance_mat[committee, lab] + 1e-9) / committee_consensus
                        )
                score.append(tmp)
        else:
            raise Exception(
                "A 2D probabilistic prediction matrix must be provided, with the shape like [n_samples, n_class]"
            )
        return score


class QBC_VE(QBC_Base):
    def get_probas(self, clf, dataset):
        return clf.predict(dataset)

    def calculate_vote(self, ensemble_probas):
        # source: alipy
        """Calculate the vote entropy for measuring the level of disagreement in QBC.
        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.
        Returns
        -------
        score: list
            Score for each instance. Shape [n_samples]
        References
        ----------
        [1] I. Dagan and S. Engelson. Committee-based sampling for training probabilistic
            classifiers. In Proceedings of the International Conference on Machine
            Learning (ICML), pages 150-157. Morgan Kaufmann, 1995.
        """
        score = []
        committee_size = len(ensemble_probas)
        input_shape = np.shape(ensemble_probas[0])

        if len(input_shape) == 2:
            ele_uni = np.unique(ensemble_probas)
            if not (len(ele_uni) == 2 and 0 in ele_uni and 1 in ele_uni):
                raise ValueError("The predicted label matrix must only contain 0 and 1")
            # calc each instance
            for i in range(input_shape[0]):
                instance_mat = np.array(
                    [X[i, :] for X in ensemble_probas if X is not None]
                )
                voting = np.sum(instance_mat, axis=0)
                tmp = 0
                # calc each label
                for vote in voting:
                    if vote != 0:
                        tmp += (
                            vote
                            / len(ensemble_probas)
                            * np.log((vote + 1e-9) / len(ensemble_probas))
                        )
                score.append(-tmp)
        else:
            input_mat = np.array([X for X in ensemble_probas if X is not None])
            # label_arr = np.unique(input_mat)
            # calc each instance's score
            for i in range(input_shape[0]):
                count_dict = collections.Counter(input_mat[:, i])
                tmp = 0
                for key in count_dict:
                    tmp += (
                        count_dict[key]
                        / committee_size
                        * np.log((count_dict[key] + 1e-9) / committee_size)
                    )
                score.append(-tmp)
        return score


class SubsamplingQueryStrategy(QueryStrategy):
    """A decorator that first subsamples randomly from the unlabeled pool and then applies
    the `base_query_strategy` on the sampled subset.
    """

    def __init__(self, base_query_strategy, subsample_size=4096):
        """
        Parameters
        ----------
        base_query_strategy : QueryStrategy
            Base query strategy to which the querying is being delegated after subsampling.
        subsample_size : int, default=4096
            Size of the subsampled set.
        """
        self.base_query_strategy = base_query_strategy
        self.subsample_size = subsample_size

        self.subsampled_indices_ = None

    def query(
        self,
        clf,
        dataset,
        indices_unlabeled,
        indices_labeled,
        y,
        n=10,
        save_scores=False,
    ):
        self._validate_query_input(indices_unlabeled, n)

        subsampled_indices = np.random.choice(
            indices_unlabeled, self.subsample_size, replace=False
        )
        subset = dataset[np.concatenate([subsampled_indices, indices_labeled])]
        subset_indices_unlabeled = np.arange(self.subsample_size)
        subset_indices_labeled = np.arange(
            self.subsample_size, self.subsample_size + indices_labeled.shape[0]
        )

        indices = self.base_query_strategy.query(
            clf, subset, subset_indices_unlabeled, subset_indices_labeled, y, n=n
        )

        self.subsampled_indices_ = indices

        return np.array([subsampled_indices[i] for i in indices])

    @property
    def scores_(self):
        if hasattr(self.base_query_strategy, "scores_"):
            return self.base_query_strategy.scores_[: self.subsample_size]
        return None

    def __str__(self):
        return (
            f"SubsamplingQueryStrategy(base_query_strategy={self.base_query_strategy}, "
            f"subsample_size={self.subsample_size})"
        )


class EmbeddingBasedQueryStrategy(QueryStrategy):
    """A base class for embedding-based query strategies.

    To use this class, create a subclass and implement `sample()`.
    """

    def query(
        self,
        clf,
        dataset,
        indices_unlabeled,
        indices_labeled,
        y,
        n=10,
        pbar="tqdm",
        embeddings=None,
        embed_kwargs=dict(),
        save_scores=False,
    ):
        self._validate_query_input(indices_unlabeled, n)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        if embeddings is not None:
            sampled_indices = self.sample(
                clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings
            )
        else:
            try:
                embeddings, proba = (
                    clf.embed(dataset, return_proba=True, pbar=pbar, **embed_kwargs)
                    if embeddings is None
                    else embeddings
                )
                sampled_indices = self.sample(
                    clf,
                    dataset,
                    indices_unlabeled,
                    indices_labeled,
                    y,
                    n,
                    embeddings,
                    embeddings_proba=proba,
                )
            except TypeError as e:
                if "got an unexpected keyword argument 'return_proba'" in e.args[0]:
                    embeddings = (
                        clf.embed(dataset, pbar=pbar, **embed_kwargs)
                        if embeddings is None
                        else embeddings
                    )
                    sampled_indices = self.sample(
                        clf,
                        dataset,
                        indices_unlabeled,
                        indices_labeled,
                        y,
                        n,
                        embeddings,
                    )
                else:
                    raise e

        return indices_unlabeled[sampled_indices]

    @abstractmethod
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
        """Samples from the given embeddings.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : Dataset
            A text dataset.
        indices_unlabeled : ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : ndarray[int]
            List of labels where each label maps by index position to `indices_labeled`.
        dataset : ndarray
            Instances for which the score should be computed.
        n : int
            Number of instances to sample.
        embeddings_proba : ndarray, default=None
            Class probabilities for each embedding in embeddings.

        Returns
        -------
        indices : ndarray[int]
            A numpy array of selected indices (relative to `indices_unlabeled`).
        """
        pass

    def __str__(self):
        return "EmbeddingBasedQueryStrategy()"


class EmbeddingKMeans(EmbeddingBasedQueryStrategy):
    """This is a generalized version of BERT-K-Means [YLB20]_, which is applicable to any kind
    of dense embedding, regardless of the classifier.
    """

    def __init__(self, normalize=True):
        """
        Parameters
        ----------
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        """
        self.normalize = normalize

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
        """Samples from the given embeddings.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A classifier.
        dataset : Dataset
            A dataset.
        indices_unlabeled : ndarray
            Indices (relative to `x`) for the unlabeled data.
        indices_labeled : ndarray
            Indices (relative to `x`) for the labeled data.
        y : ndarray or list of int
            List of labels where each label maps by index position to `indices_labeled`.
        dataset : ndarray
            Instances for which the score should be computed.
        embeddings : ndarray
            Embeddings for each sample in x.

        Returns
        -------
        indices : ndarray
            A numpy array of selected indices (relative to `indices_unlabeled`).
        """
        from sklearn.cluster import KMeans

        if self.normalize:
            from sklearn.preprocessing import normalize

            embeddings = normalize(embeddings, axis=1)

        km = KMeans(n_clusters=n)
        km.fit(embeddings[indices_unlabeled])

        indices = self._get_nearest_to_centers(
            km.cluster_centers_,
            embeddings[indices_unlabeled],
            normalized=self.normalize,
        )

        # fall back to an iterative version if one or more vectors are most similar
        # to multiple cluster centers
        if np.unique(indices).shape[0] < n:
            indices = self._get_nearest_to_centers_iterative(
                km.cluster_centers_,
                embeddings[indices_unlabeled],
                normalized=self.normalize,
            )

        return indices

    @staticmethod
    def _get_nearest_to_centers(centers, vectors, normalized=True):
        sim = EmbeddingKMeans._similarity(centers, vectors, normalized)
        return sim.argmax(axis=1)

    @staticmethod
    def _similarity(centers, vectors, normalized):
        sim = np.matmul(centers, vectors.T)

        if not normalized:
            sim = sim / np.dot(
                np.linalg.norm(centers, axis=1)[:, np.newaxis],
                np.linalg.norm(vectors, axis=1)[np.newaxis, :],
            )
        return sim

    @staticmethod
    def _get_nearest_to_centers_iterative(cluster_centers, vectors, normalized=True):
        indices = np.empty(cluster_centers.shape[0], dtype=int)

        for i in range(cluster_centers.shape[0]):
            sim = EmbeddingKMeans._similarity(
                cluster_centers[None, i], vectors, normalized
            )
            sim[0, indices[0:i]] = -np.inf
            indices[i] = sim.argmax()

        return indices

    def __str__(self):
        return f"EmbeddingKMeans(normalize={self.normalize})"


class ContrastiveActiveLearning(EmbeddingBasedQueryStrategy):
    """Contrastive Active Learning [MVB+21]_ selects instances whose k-nearest neighbours
    exhibit the largest mean Kullback-Leibler divergence."""

    def __init__(
        self, k=10, embed_kwargs=dict(), normalize=True, batch_size=100, pbar="tqdm"
    ):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbours whose KL divergence is considered.
        embed_kwargs : dict
            Embedding keyword args which are passed to `clf.embed()`.
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        batch_size : int, default=100
            Batch size which is used to process the embeddings.
        """
        self.embed_kwargs = embed_kwargs
        self.normalize = normalize
        self.k = k
        self.batch_size = batch_size
        self.pbar = pbar

    def query(
        self,
        clf,
        dataset,
        x_indices_unlabeled,
        x_indices_labeled,
        y,
        n=10,
        pbar="tqdm",
        embeddings=None,
        embed_kwargs=dict(),
        save_scores=False,
    ):

        return super().query(
            clf,
            dataset,
            x_indices_unlabeled,
            x_indices_labeled,
            y,
            n=n,
            embed_kwargs=self.embed_kwargs,
            pbar=self.pbar,
        )

    def sample(
        self,
        _clf,
        dataset,
        indices_unlabeled,
        _indices_labeled,
        _y,
        n,
        embeddings,
        embeddings_proba=None,
    ):
        from sklearn.neighbors import NearestNeighbors

        if embeddings_proba is None:
            raise ValueError(
                "Error: embeddings_proba is None. "
                "This strategy requires a classifier whose embed() method "
                "supports the return_proba kwarg."
            )

        if self.normalize:
            embeddings = normalize(embeddings, axis=1)

        nn = NearestNeighbors(n_neighbors=n)
        nn.fit(embeddings)

        return self._contrastive_active_learning(
            dataset, embeddings, embeddings_proba, indices_unlabeled, nn, n
        )

    def _contrastive_active_learning(
        self, dataset, embeddings, embeddings_proba, indices_unlabeled, nn, n
    ):
        from scipy.special import rel_entr

        scores = []

        embeddings_unlabelled_proba = embeddings_proba[indices_unlabeled]
        embeddings_unlabeled = embeddings[indices_unlabeled]

        num_batches = int(np.ceil(len(dataset) / self.batch_size))
        offset = 0
        for batch_idx in np.array_split(
            np.arange(indices_unlabeled.shape[0]), num_batches, axis=0
        ):

            nn_indices = nn.kneighbors(
                embeddings_unlabeled[batch_idx],
                n_neighbors=self.k,
                return_distance=False,
            )

            kl_divs = np.apply_along_axis(
                lambda v: np.mean(
                    [
                        rel_entr(embeddings_proba[i], embeddings_unlabelled_proba[v])
                        for i in nn_indices[v - offset]
                    ]
                ),
                0,
                batch_idx[None, :],
            )

            scores.extend(kl_divs.tolist())
            offset += batch_idx.shape[0]

        scores = np.array(scores)
        indices = np.argpartition(-scores, n)[:n]

        return indices

    def __str__(self):
        return (
            f"ContrastiveActiveLearning(k={self.k}, "
            f"embed_kwargs={str(self.embed_kwargs)}, "
            f"normalize={self.normalize})"
        )
