import abc
from typing_extensions import Self

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.integrations.transformers.classifiers.classification import (
    TransformerBasedClassification,
)
from small_text.utils.classification import empty_result

import numpy as np

from functools import partial


try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from small_text.integrations.pytorch.utils.data import dataloader
    from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss

except ImportError:
    raise PytorchNotFoundError("Could not import pytorch")


class UncertaintyBaseClass(TransformerBasedClassification):
    @abc.abstractmethod
    def predict_proba(self, test_set):
        raise NotImplementedError


# to be fixed: temp_scaling, label_smoothing, inhibited

# works
class SoftmaxUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        if len(test_set) == 0:
            return empty_result(
                self.multi_label,
                self.num_classes,
                return_prediction=False,
                return_proba=True,
            )

        self.model.eval()
        test_iter = dataloader(
            test_set.data, self.mini_batch_size, self._create_collate_fn(), train=False
        )

        predictions = []
        logits_transform = (
            torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)
        )

        with torch.no_grad():
            for text, masks, *_ in test_iter:
                text, masks = text.to(self.device), masks.to(self.device)
                outputs = self.model(text, attention_mask=masks)

                predictions += logits_transform(outputs.logits).to("cpu").tolist()
                del text, masks

        return np.array(predictions)


# .local/lib/python3.8/site-packages/small_text/integrations/transformers/classifiers/classification.py
#  Implementation for Calibration - Temperature Scaling: https://github.com/shreydesai/calibration/blob/master/calibrate.py
class TemperatureScalingUncertaintyClassifier(UncertaintyBaseClass):
    def __init__(
        self,
        transformer_model,
        num_classes,
        multi_label=False,
        num_epochs=10,
        lr=0.00002,
        mini_batch_size=12,
        validation_set_size=0.1,
        validations_per_epoch=1,
        no_validation_set_action="sample",
        early_stopping_no_improvement=5,
        early_stopping_acc=-1,
        model_selection=True,
        fine_tuning_arguments=None,
        device=None,
        memory_fix=1,
        class_weight=None,
        verbosity=...,
        cache_dir=".active_learning_lib_cache/",
    ):
        super().__init__(
            transformer_model,
            num_classes,
            multi_label,
            num_epochs,
            lr,
            mini_batch_size,
            validation_set_size,
            validations_per_epoch,
            no_validation_set_action,
            early_stopping_no_improvement,
            early_stopping_acc,
            model_selection,
            fine_tuning_arguments,
            device,
            memory_fix,
            class_weight,
            verbosity,
            cache_dir,
        )

        self.temperature = torch.nn.ParameterDict(torch.ones(1) * 1.5)

    def _compute_loss(self, cls, outputs, epoch, validate=False):
        if self.num_classes == 2:
            logits = outputs.logits
            target = F.one_hot(cls, 2).float()
        else:
            logits = outputs.logits.view(-1, self.num_classes)
            target = cls

        loss = self.criterion(self._temperature_scale(logits), target)

        return logits, loss

    # Validate:

    def validate(self, validation_set, epoch):

        valid_loss = 0.0
        acc = 0.0

        self.model.eval()
        valid_iter = dataloader(
            validation_set.data,
            self.mini_batch_size,
            self._create_collate_fn(),
            train=False,
        )

        logits_list = []
        labels_list = []

        for x, masks, cls in valid_iter:
            x, masks, cls = (
                x.to(self.device),
                masks.to(self.device),
                cls.to(self.device),
            )

            with torch.no_grad():
                outputs = self.model(x, attention_mask=masks)
                _, loss = self._compute_loss(cls, outputs, epoch)

                logits_list.append(outputs.logits)
                # labels_list.append(F.one_hot(cls, 2).float())
                labels_list.append(F.one_hot(cls, self.num_classes))

                valid_loss += loss.item()
                acc += self.sum_up_accuracy_(outputs.logits, cls)
                del outputs, x, masks, cls

        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()

        logits_labels = []

        for i in range(len(logits)):
            logits_labels.append([logits[i], labels[i]])

        # logging.info(logits)
        # logging.info(labels)

        best_nll = float("inf")
        best_temp = -1

        temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))

        for temp in temp_values:
            nll = np.mean(
                [
                    self.cross_entropy(F.log_softmax(elem[0] / temp, 0), elem[1])
                    for elem in logits_labels
                ]
            )

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        if best_temp < 1:
            best_temp = 1.8495

        self.temperature = best_temp

        return valid_loss / len(validation_set), acc / len(validation_set)

    def cross_entropy(self, output, target):
        """
        Computes cross-entropy with KL divergence from predicted distribution
        and true distribution, specifically, the predicted log probability
        vector and the true one-hot label vector.
        """

        return F.kl_div(output, target, reduction="sum").item()

    def predict_proba(self, test_set):
        if len(test_set) == 0:
            return empty_result(
                self.multi_label,
                self.num_classes,
                return_prediction=False,
                return_proba=True,
            )
        self.model.eval()
        test_iter = dataloader(
            test_set.data, self.mini_batch_size, self._create_collate_fn(), train=False
        )
        predictions = []
        logits_transform = (
            torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)
        )

        with torch.no_grad():
            for text, masks, label in test_iter:
                text, masks, label = (
                    text.to(self.device),
                    masks.to(self.device),
                    label.to(self.device),
                )
                outputs = self.model(text, attention_mask=masks)

                predictions += logits_transform(outputs.logits).to("cpu").tolist()
                del text, masks

        return np.array(predictions)


class LabelSmoothingUncertaintyClassifier(SoftmaxUncertaintyClassifier):
    def get_default_criterion(self):
        if self.multi_label or self.num_classes == 2:
            print("ERROR-" * 200)
            return BCEWithLogitsLoss(pos_weight=self.class_weights_)
        else:
            return CrossEntropyLoss(weight=self.class_weights_, label_smoothing=0.2)


# works
class MonteCarloDropoutUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        if len(test_set) == 0:
            return empty_result(
                self.multi_label,
                self.num_classes,
                return_prediction=False,
                return_proba=True,
            )

        self.model.eval()
        test_iter = dataloader(
            test_set.data, self.mini_batch_size, self._create_collate_fn(), train=False
        )

        predictions = []
        logits_transform = (
            torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)
        )

        with torch.no_grad():
            for text, masks, _ in test_iter:
                text, masks = text.to(self.device), masks.to(self.device)
                outputs = self.model(text, attention_mask=masks)

                dropout = torch.nn.Dropout(p=0.1)

                softmaxListAfterDropout = []

                # at the beginng we try 50 iterations of dropout and then maybe test 100 and compare the results + time (Is it useful?)
                # Gal in implementation uses 10000 iterations I think

                # what we do: We use dropout on our logits and then input the logits to Softmax and do this n times, then we use the mean of the
                # softmax outputs for the prediction

                # Question: Does the mean equal 1 over all classes?

                for iteration in range(50):
                    logitsPostDropout = dropout(outputs.logits)
                    # print(logits_transform(logitsPostDropout).to('cpu').tolist())
                    softmaxListAfterDropout += [
                        logits_transform(logitsPostDropout).to("cpu").tolist()
                    ]
                # print(softmaxListAfterDropout)

                meanDropout = np.mean(softmaxListAfterDropout, axis=0).tolist()
                predictions += meanDropout

                del text, masks

        return np.array(predictions)


class InhibitedSoftmaxUncertaintyClassifier(UncertaintyBaseClass):

    # we also changed the activation function of BERT from GELU to pdf in 11th layer (second to last) and use these outputs for our loss (see _compute_loss)

    # The implementation of the alternative softmax:

    def inhibitedSoftmax(self, logits):
        logits.cuda()
        # exponentials = np.exp(logits)
        # sum_exponentials = torch.sum(exponentials, 1)

        inhibition = torch.nn.Parameter(torch.ones(1).cuda() * 1.5)
        inhibition = (
            inhibition.unsqueeze(1)
            .expand(logits.size(0), logits.size(1))
            .to(self.device)
        )

        result = torch.exp(logits) / (
            torch.exp(logits).sum(dim=1).view(-1, 1) + inhibition
        )
        result.cuda()

        return result

    # We not only use this softmax for the loss but also for the prediction

    def predict_proba(self, test_set):
        if len(test_set) == 0:
            return empty_result(
                self.multi_label,
                self.num_classes,
                return_prediction=False,
                return_proba=True,
            )

        self.model.eval()
        test_iter = dataloader(
            test_set.data, self.mini_batch_size, self._create_collate_fn(), train=False
        )

        predictions = []
        logits_transform = (
            torch.sigmoid if self.multi_label else partial(self.inhibitedSoftmax)
        )  # partial(F.softmax, dim=1)

        with torch.no_grad():
            for text, masks, _ in test_iter:
                text, masks = text.to(self.device), masks.to(self.device)
                outputs = self.model(text, attention_mask=masks)

                predictions += logits_transform(outputs.logits).to("cpu").tolist()
                del text, masks

        return np.array(predictions)

    # we compute the loss like it is shown in the paper

    def _compute_loss(self, cls, outputs):

        if self.num_classes == 2:
            logits = outputs.logits
            target = F.one_hot(cls, 2).float()
        else:
            logits = outputs.logits.view(-1, self.num_classes)
            target = cls

        #  we take the 11th layer output
        hiddenOutput = outputs.hidden_states[11][1]

        minimization = torch.nn.Linear(in_features=768, out_features=6, bias=True)

        layerOutput = minimization(hiddenOutput)

        loss = (
            F.nll_loss(torch.log(self.inhibitedSoftmax(logits)), target)
            + 0.000001 * layerOutput.sum()
        )

        return logits, loss


#     Effective Deep Learning Implementation (Survoy (?) et al. - Kaplan)
class EvidentialDeepLearning1UncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        if len(test_set) == 0:
            return empty_result(
                self.multi_label,
                self.num_classes,
                return_prediction=False,
                return_proba=True,
            )
        self.model.eval()
        test_iter = dataloader(
            test_set.data, self.mini_batch_size, self._create_collate_fn(), train=False
        )
        predictions = []
        logits_transform = (
            torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)
        )

        with torch.no_grad():
            for text, masks, label in test_iter:
                text, masks, label = (
                    text.to(self.device),
                    masks.to(self.device),
                    label.to(self.device),
                )
                outputs = self.model(text, attention_mask=masks)

                predictions += logits_transform(outputs.logits).to("cpu").tolist()
                del text, masks

        return np.array(predictions)

    def _compute_loss(self, cls, outputs, epoch):

        if self.num_classes == 2:
            logits = outputs.logits
            target = F.one_hot(cls, 2).float()
        else:
            logits = outputs.logits.view(-1, self.num_classes)
            target = cls

        evidence = F.relu(logits)
        alpha = evidence + 1

        # Ich glaube, dass annealing_step = die num_classes ist -> das nehmen wir jetzt mal an

        y = F.one_hot(cls, self.num_classes).float()  # = target

        # KL Divergence:
        def kl_divergence(alpha, num_classes):
            ones = torch.ones([1, num_classes], dtype=torch.float32, device=self.device)
            sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
            first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
            )
            second_term = (
                (alpha - ones)
                .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
                .sum(dim=1, keepdim=True)
            )
            kl = first_term + second_term

            return kl

        # Loglikelihood loss

        def loglikelihood_loss(y, alpha):

            y = y.to(self.device)
            alpha = alpha.to(self.device)
            S = torch.sum(alpha, dim=1, keepdim=True)
            loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
            loglikelihood_var = torch.sum(
                alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
            )
            loglikelihood = loglikelihood_err + loglikelihood_var

            return loglikelihood

        def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):

            y = y.to(self.device)
            alpha = alpha.to(self.device)
            loglikelihood = loglikelihood_loss(y, alpha)

            annealing_coef = torch.min(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
            )

            kl_alpha = (alpha - 1) * (1 - y) + 1
            kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)

            return loglikelihood + kl_div

        loss = torch.mean(
            mse_loss(
                y=y,
                alpha=alpha,
                epoch_num=epoch,
                num_classes=self.num_classes,
                annealing_step=10,
            )
        )

        # loss = self.criterion(logits, target)

        return logits, loss


class EvidentialDeepLearning2UncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class BayesianUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class EnsemblesUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class TrustScoreUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class ModelCalibrationUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError
