from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.transformers.classifiers.classification import (
    TransformerBasedClassification,
)

from small_text.integrations.transformers.classifiers.uncertainty_base_class import (
    BayesianUncertaintyClassifier,
    EnsemblesUncertaintyClassifier,
    EvidentialDeepLearning1UncertaintyClassifier,
    EvidentialDeepLearning2UncertaintyClassifier,
    InhibitedSoftmaxUncertaintyClassifier,
    LabelSmoothingUncertaintyClassifier,
    ModelCalibrationUncertaintyClassifier,
    MonteCarloDropoutUncertaintyClassifier,
    SoftmaxUncertaintyClassifier,
    TemperatureScalingUncertaintyClassifier,
    TemperatureScaling2UncertaintyClassifier,
    TrustScoreUncertaintyClassifier,
)


class TransformerBasedClassificationFactory(AbstractClassifierFactory):
    def __init__(self, transformer_model, num_classes, kwargs={}):
        self.transformer_model = transformer_model
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):
        return TransformerBasedClassification(
            self.transformer_model, self.num_classes, **self.kwargs
        )


class UncertaintyBasedClassificationFactory(TransformerBasedClassificationFactory):
    def __init__(
        self, transformer_model, num_classes, uncertainty_method="softmax", kwargs={}
    ):
        self.uncertainty_method = uncertainty_method
        super().__init__(transformer_model, num_classes, kwargs)

    def new(self):
        if self.uncertainty_method == "softmax":
            return SoftmaxUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "temp_scaling":
            return TemperatureScalingUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "label_smoothing":
            return LabelSmoothingUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "MonteCarlo":
            return MonteCarloDropoutUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "inhibited":
            return InhibitedSoftmaxUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "evidential1":
            return EvidentialDeepLearning1UncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "evidential2":
            return EvidentialDeepLearning2UncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "bayesian":
            return BayesianUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "ensembles":
            return EnsemblesUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "trustscore":
            return TrustScoreUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "model_calibration":
            return ModelCalibrationUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "temp_scaling2":
            return TemperatureScaling2UncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        else:
            return super().new()
