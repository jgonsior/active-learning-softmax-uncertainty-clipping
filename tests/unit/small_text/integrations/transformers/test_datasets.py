import unittest
import pytest

import numpy as np

from scipy.sparse import csr_matrix

from numpy.testing import assert_array_equal
from parameterized import parameterized_class

from small_text.integrations.pytorch.datasets import PytorchDatasetView
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.testing import (
    assert_array_not_equal,
    assert_csr_matrix_equal,
    assert_csr_matrix_not_equal,
    assert_list_of_tensors_equal,
)

try:
    import torch

    from small_text.integrations.pytorch.query_strategies import (
        ExpectedGradientLength,
        ExpectedGradientLengthMaxWord,
    )
    from small_text.integrations.transformers.datasets import TransformersDataset
    from tests.utils.datasets import random_transformer_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
@parameterized_class(
    [
        {"target_labels": "explicit", "multi_label": True},
        {"target_labels": "explicit", "multi_label": False},
        {"target_labels": "inferred", "multi_label": True},
        {"target_labels": "inferred", "multi_label": False},
    ]
)
class TransformersDatasetTest(unittest.TestCase):
    NUM_SAMPLES = 100
    NUM_LABELS = 3

    def _random_data(self, num_samples=NUM_SAMPLES):
        if self.target_labels not in ["explicit", "inferred"]:
            raise ValueError(
                "Invalid test parameter value for target_labels:" + self.target_labels
            )

        return random_transformer_dataset(
            num_samples=num_samples,
            multi_label=self.multi_label,
            num_classes=self.NUM_LABELS,
            target_labels=self.target_labels,
        )

    def test_init(self):
        ds = self._random_data()
        self.assertIsNotNone(ds._data)

    def test_get_features(self):
        ds = self._random_data()
        self.assertEqual(self.NUM_SAMPLES, len(ds.x))

    def test_set_features(self):
        ds = self._random_data()
        self.assertEqual(self.NUM_SAMPLES, len(ds.x))
        ds_new = self._random_data()
        ds.x = ds_new.x
        assert_list_of_tensors_equal(self, ds.x, ds_new.x)

    def test_get_labels(self):
        ds = self._random_data()
        if self.multi_label:
            self.assertTrue(isinstance(ds.y, csr_matrix))
            y_expected = np.zeros((self.NUM_SAMPLES, self.NUM_LABELS))
            for i, d in enumerate(ds.data):
                labels = d[ds.INDEX_LABEL]
                if labels is not None:
                    y_expected[i, labels] = 1
            y_expected = csr_matrix(y_expected)
            assert_csr_matrix_equal(y_expected, ds.y)
        else:
            self.assertTrue(isinstance(ds.y, np.ndarray))
            self.assertEqual(self.NUM_SAMPLES, ds.y.shape[0])

    def test_set_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        ds_new = self._random_data(num_samples=self.NUM_SAMPLES)

        if self.multi_label:
            assert_csr_matrix_not_equal(ds.y, ds_new.y)
        else:
            assert_array_not_equal(ds.y, ds_new.y)

        ds.y = ds_new.y

        if self.multi_label:
            assert_csr_matrix_equal(ds.y, ds_new.y)
        else:
            assert_array_equal(ds.y, ds_new.y)
            self.assertTrue(isinstance(ds.y, np.ndarray))
            self.assertEqual(self.NUM_SAMPLES, ds.y.shape[0])

    def test_set_labels_with_mismatching_data_length(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        ds_new = self._random_data(num_samples=self.NUM_SAMPLES + 1)

        with self.assertRaisesRegex(ValueError, "Size mismatch: "):
            ds.y = ds_new.y

    def test_is_multi_label(self):
        dataset = self._random_data(num_samples=self.NUM_SAMPLES)
        if self.multi_label:
            self.assertTrue(dataset.is_multi_label)
        else:
            self.assertFalse(dataset.is_multi_label)

    def test_get_target_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        expected_target_labels = np.arange(self.NUM_LABELS)
        assert_array_equal(expected_target_labels, ds.target_labels)

    def test_set_target_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        new_target_labels = np.arange(self.NUM_LABELS + 1)
        ds.target_labels = new_target_labels
        assert_array_equal(new_target_labels, ds.target_labels)

    def test_set_target_labels_where_existing_labels_are_outside_of_given_set(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        new_target_labels = np.array([0])
        with self.assertRaisesRegex(ValueError, "Cannot remove existing labels"):
            ds.target_labels = new_target_labels

    def test_get_data(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        assert_array_equal(len(ds), len(ds.data))

    def test_indexing_single_index(self, index=42):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result, PytorchDatasetView))

        self.assertTrue(torch.equal(ds.x[index], result.x[0]))

    def test_indexing_list_index(self):
        index = [1, 42, 56, 99]
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(4, len(result))
        self.assertTrue(isinstance(result, PytorchDatasetView))

        expected = [ds.x[i] for i in index]
        assert_list_of_tensors_equal(self, expected, result.x)

    def test_indexing_slicing(self):
        index = np.s_[10:20]
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, PytorchDatasetView))

        expected = [ds.x[i] for i in np.arange(self.NUM_SAMPLES)[index]]
        assert_list_of_tensors_equal(self, expected, result.x)
