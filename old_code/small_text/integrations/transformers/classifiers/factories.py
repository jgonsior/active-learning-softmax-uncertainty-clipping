from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.transformers.classifiers.classification import (
    TransformerBasedClassification,
)

from small_text.integrations.transformers.classifiers.uncertainty_base_class import (
    EvidentialDeepLearning1UncertaintyClassifier,
    InhibitedSoftmaxUncertaintyClassifier,
    LabelSmoothingUncertaintyClassifier,
    MonteCarloDropoutUncertaintyClassifier,
    SoftmaxUncertaintyClassifier,
    TemperatureScalingUncertaintyClassifier,
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
        original_transformer = TransformerBasedClassification(
            self.transformer_model, self.num_classes, **self.kwargs
        )
        if self.uncertainty_method == "softmax":
            return original_transformer, SoftmaxUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "temp_scaling":
            return original_transformer, TemperatureScalingUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "label_smoothing":
            return original_transformer, LabelSmoothingUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "MonteCarlo":
            return original_transformer, MonteCarloDropoutUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "inhibited":
            return original_transformer, InhibitedSoftmaxUncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        elif self.uncertainty_method == "evidential1":
            return original_transformer, EvidentialDeepLearning1UncertaintyClassifier(
                self.transformer_model, self.num_classes, **self.kwargs
            )
        else:
            return super().new()
