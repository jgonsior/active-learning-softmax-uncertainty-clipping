from abc import ABC, abstractmethod
import abc
import collections
import os
import random
from matplotlib.pyplot import sca

import numpy as np
from scipy.stats import entropy
from sklearn import ensemble
from sklearn.preprocessing import normalize
import torch

from torch import nn, optim
from torch.nn import functional as F

from small_text.integrations.transformers.classifiers.trust_score import TrustScore
from small_text.integrations.transformers.datasets import TransformersDataset

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
        if save_scores:
            self.last_scores = [1 for _ in range(n)]
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


        clipping_threshold = np.percentile(
            confidence, (1 - self.uncertainty_clipping) * 100
        )

        print(f"original confidence: {confidence}")

        confidence[
            confidence < clipping_threshold
        ] = 1  # as lower_is_better = True is default, we set it to 1 as this is then the worst possible value

        self.scores_ = confidence
        if not self.lower_is_better:
            confidence = -confidence

        if self.save_scores:
            self.last_scores = confidence
        print(f"clipped confidence: {confidence}")
        print(f"{clipping_threshold}")
        print()
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
            lower_is_better=True, uncertainty_clipping=uncertainty_clipping
        )

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        if self.predict_proba_with_labeled_data:
            clf.tell_me_so_far_labeled_data(X=dataset[_indices_labeled].x, Y=_y)
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
            lower_is_better=True, uncertainty_clipping=uncertainty_clipping
        )

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        if self.predict_proba_with_labeled_data:
            clf.tell_me_so_far_labeled_data(X=dataset[_indices_labeled].x, Y=_y)
        proba = clf.predict_proba(dataset)

        return np.amax(proba, axis=1)

    def __str__(self):
        return "LeastConfidence()"


class PredictionEntropy(ConfidenceBasedQueryStrategy):
    """Selects instances with the largest prediction entropy [HOL08]_."""

    def __init__(self, lower_is_better=False, uncertainty_clipping=1.0):
        super().__init__(
            lower_is_better=False, uncertainty_clipping=uncertainty_clipping
        )

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        if self.predict_proba_with_labeled_data:
            clf.tell_me_so_far_labeled_data(X=dataset[_indices_labeled].x, Y=_y)
        proba = clf.predict_proba(dataset)
        return np.apply_along_axis(lambda x: entropy(x), 1, proba)

    def __str__(self):
        return "PredictionEntropy()"


# UNLABELED DATEN HABEN NIEDRIGEREN TRUSTSCORE,LABELD DATEN HÖHREN TRUSTSCORE, UMSO HÖHER TRUSTSCORE UMSO SICHERER
class Trustscore2(
    ConfidenceBasedQueryStrategy
):  # das richtige weil hier density filter
    """Selects instances with the least prediction confidence (regarding the most likely class)
    [LG94]_."""

    # Btw K=10 if possible hängt damit zusammen das man mit 25 Datenpunkten startet,
    # wenn man 2 Klassen hat -> 12/13 Datenpunkte , d.h K=12 und K=13
    # Bei Trec6 ist am Anfang 4 in jeden Datenpunkt, d.h müssen ersten 1, 2 iterationen unter 10 gehen, danach sind genug dafür da

    def __init__(self, uncertainty_clipping=1.0):
        super().__init__(
            lower_is_better=True, uncertainty_clipping=uncertainty_clipping
        )  # Eigentlich sollten lower Trust score gewählt weden, vor 6.7 waren alle experimente false
        #                                   TRUE IST DEFAULT am besten
        self._clsUNLABELED = None
        self.alle_listen = []  # für distro

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        def clsCreator():
            clsListe = []
            for abc in _indices_labeled:
                test = dataset.data[abc]
                clsListe.append((test[0], test[1], test[2]))
            transformerData = TransformersDataset(clsListe)
            _clsLabeled = clf.embed(transformerData, embedding_method="cls")
            return _clsLabeled

        _clsLabeled = clsCreator()

        # Doppelt weil beim ersten mal CLS Vektor Erstellen es manchmal nicht klappt LOL?
        print(
            "Labeled CLS Vektor Erstellung Efolgreich:",
            float(_clsLabeled[0][0]) is not float(0),
            _clsLabeled[0][0],
        )
        if _clsLabeled[0][0] == float(0):
            _clsLabeled = clsCreator()
            print(
                "Retry CLS Vektor Erstellung Efolgreich:",
                float(_clsLabeled[0][0]) is not float(0),
                _clsLabeled[0][0],
            )

        # _clsLabeled = torch.from_numpy(_clsLabeled)

        # ----------------

        y_pred = clf.predict(
            dataset
        )  # default predictions mit einer Klasse direkt, also kein Onehot von allen Daten

        print("probagroß", y_pred, y_pred.shape)
        y_pred_train = clf.predict(dataset[_indices_labeled])  # vom train Set
        print("probatrained", y_pred_train[:10], y_pred_train.shape)
        # a = clf.predict_proba(dataset,logit=True) # statt Softmax die Logits

        # Initialize trust score.
        trust_model = TrustScore(
            k=10, alpha=0.1, filtering="density"
        )  # default hier nix drin
        # denke wir nehmen cls vektor(statt nur dataset[0]) und y einfach
        # brauchen aber cls vektor von unlabeld daten

        # print("clsOne",self._clsUNLABELED[0])
        trust_model.fit(_clsLabeled, _y)

        # Compute trusts score, given (unlabeled) testing examples and (hard) model predictions.
        trust_score = trust_model.get_score(self._clsUNLABELED, y_pred)
        print("unlabeld Trust Score", trust_score[:8], trust_score.shape)

        return trust_score  # np.amax(proba, axis=1)

    def __str__(self):
        return "TrustScore2()"


class QBC_Base(ConfidenceBasedQueryStrategy):
    def __init__(
        self,
        clf_factory,
        lower_is_better=True,
        uncertainty_clipping=1.0,
        amount_of_ensembles=5,
    ):
        super().__init__(
            lower_is_better=True, uncertainty_clipping=uncertainty_clipping
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

            new_classifier, _ = self.factory.new()

            # TODO: sollte factory später mal mit was anderem als nur softmax umgehen -> hier die ganzen anderen active classifier verwenden!
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


class TemperatureScalingStrat(ConfidenceBasedQueryStrategy):
    """TemperatureScaling nach Liste, Gibt calibrierte Logits und Softmax
    Default LeastConfidence nur mit TemperatureScaling + Softmax"""

    def __init__(self, uncertainty_clipping=1.0, clf_factory=None):
        super().__init__(
            lower_is_better=True, uncertainty_clipping=uncertainty_clipping
        )  # True ist default beste
        # für distro
        self._clf_factory = clf_factory
        self.alle_listen = []

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        orig_model = clf.model
        former = []
        for abc in _indices_labeled:
            test = dataset.data[abc]
            former.append((test[0], test[1], test[2]))

        valid_loader = torch.utils.data.DataLoader(
            former, pin_memory=True, batch_size=16
        )

        scaled_model = ModelWithTemperature(
            orig_model, device=self._clf_factory.kwargs["device"]
        ).to(device=self._clf_factory.kwargs["device"])
        # ACHTUNG TEMPERATURE SCALING HAT ANDERE HYPERPARAMTER, Initial Lr und max_iter
        scaled_model.set_temperature(valid_loader)

        with torch.no_grad():
            proba_scaled = scaled_model(dataset)
            # Softmax umwandlung der Scaled Werte = predictions
            predictions = []
            multi_label = clf.multi_label
            logits_transform = (
                torch.sigmoid if multi_label else partial(F.softmax, dim=1)
            )
            data_loader = torch.utils.data.DataLoader(proba_scaled, batch_size=128)
            for data in data_loader:
                predictions += torch.unsqueeze(
                    logits_transform(data), dim=0
                )  # .to('cpu')

        # predictions = torch.unsqueeze(predictions,dim=0)
        predictions = torch.cat(predictions)

        predictions = predictions.cpu().detach().numpy()

        # proba = clf.predict_proba(dataset) vergleich zu predictions = ohne calibration softmaxwerte

        """
        #print("Schauen ob LeastConfidence und LeastConfidence mit Temp Sclaing andere Activelearning Werte auswählen")
        a = np.amax(predictions, axis=1)
        proba = clf.predict_proba(dataset)
        proba = np.amax(proba, axis=1)
        ind = np.argpartition(a, -13)[-13:]
        ind2 = np.argpartition(proba, -13)[-13:]
        print(ind)
        print(ind2)
        """

        proba = np.amax(predictions, axis=1)
        """
        wert = np.percentile(proba, 5) # schwelle ab den 10% - 90% ist
        for x in range(len(proba)):
            if proba[x] < wert:
                proba[x] = 1 #lower score bekommen hohen wert
        """

        # print("uncertainty",uncertainty)
        """
        uncertaintysort = proba
        uncertaintysort.sort()
        #print("sort",uncertaintysort)
        self.alle_listen.append(uncertaintysort)
        df = pd.DataFrame(self.alle_listen)
        print(df)
        df.to_csv("evidentbasicdistroBAD.csv",index=False)
        """

        return proba  # np.amax(proba, axis=1)

        # return np.amax(predictions, axis=1)

    def __str__(self):
        return "TemperatureScaling()"


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


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model, device):
        super(ModelWithTemperature, self).__init__()
        self.device = device
        self.model = model.to(device=self.device)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5).to(device=self.device)

        # self.cuda()
        # self.model = self.model.cuda()

    def forward(self, input):  # stark bearbeitet für bert nutzung
        # input ist datensatz vom modell mit text und maske, brauchen für das auch für das modell
        input_clean = []
        mask_clean = []
        loader_list = []
        for abc in range(len(input.data)):
            test = input.data[abc]
            # input_clean.append(test[0])
            # mask_clean.append(test[1])
            loader_list.append((test[0][0], test[1][0]))

        # input_clean = torch.cat(input_clean)
        # mask_clean = torch.cat(mask_clean)

        # loader_list = torch.cat(loader_list)

        # print(input_clean)
        # print("-----")
        # print(mask_clean)

        # dataloader weil GPU IS HOLDING ME BACK

        data_loader = torch.utils.data.DataLoader(
            loader_list, pin_memory=True, batch_size=128
        )
        # self.cuda()

        logits_list = []
        with torch.no_grad():
            # self.cuda()

            # self.model = self.model.cuda()
            for data in data_loader:
                input_clean, mask_clean = data
                # print("input",input_clean)
                # print(mask_clean)

                logits = self.model(input_clean, attention_mask=mask_clean).to(
                    device=self.device
                )
                logits_list.append(logits.logits)

        logits_list = torch.cat(logits_list).to(device=self.device)
        # print("logits vor temp",logits_list)
        # print(logits_list.shape)
        # self.model.predict_proba(input,logit=True)
        return self.temperature_scale(logits_list)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # Logits vom Bert Model

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, masks, label in valid_loader:
                # input = input.cuda()
                # logits = self.model(input)
                # print("input",input)
                # logits= self.model.predict_proba((input,mask),logit=True)
                self.model = self.model.cuda()
                input = torch.squeeze(input, 1).cuda()
                masks = torch.squeeze(masks, 1).cuda()
                # print(input.shape,masks.shape) #vllt input shape problematisch?
                logits = self.model(input, attention_mask=masks)
                logits = logits.logits  # .tolist()#to('cpu').tolist()
                # print(logits)
                # for x in range(len(logits)): #optimal coding practise
                #    logits_list.append(logits[x])
                #    labels_list.append(label[x])
                logits_list.append(logits)
                labels_list.append(label)
            # print(logits_list)
            logits = torch.cat(logits_list).cuda()
            # print(labels_list)
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        # print(logits, labels)
        print(
            "Before temperature - NLL: %.3f, ECE: %.3f"
            % (before_temperature_nll, before_temperature_ece)
        )
        print(self.temperature)
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS(
            [self.temperature], lr=0.02, max_iter=500
        )  # learningrate bisll verändert

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            # print("Z 88 temp sclaing",loss)
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        print("Optimal temperature: %.3f" % self.temperature.item())
        print(
            "After temperature - NLL: %.3f, ECE: %.3f"
            % (after_temperature_nll, after_temperature_ece)
        )

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
