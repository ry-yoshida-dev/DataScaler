from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from data_scaler import FeatureScaling, Scaler, TransformerLike
from data_scaler.types import FloatArray

X: NDArray[np.float64] = np.array(
    [[1.0, -10.0], [2.0, 0.0], [3.0, 10.0], [4.0, 20.0], [100.0, 30.0]]
)


class TestFeatureScaling(unittest.TestCase):
    def test_none_keeps_values_in_a_new_float_array(self) -> None:
        scaled: FloatArray = FeatureScaling.NONE.build_scaler().fit_transform(X)
        np.testing.assert_array_equal(scaled, X)
        self.assertFalse(np.shares_memory(scaled, X))
        integers: NDArray[np.int64] = np.array([[1], [2]], dtype=np.int64)
        converted: FloatArray = FeatureScaling.NONE.build_scaler().fit_transform(
            integers
        )
        self.assertEqual(converted.dtype, np.float64)

    def test_standard_has_zero_mean_and_unit_variance(self) -> None:
        scaled: FloatArray = FeatureScaling.STANDARD.build_scaler().fit_transform(X)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0)

    def test_min_max_spans_the_unit_range(self) -> None:
        scaled: FloatArray = FeatureScaling.MIN_MAX.build_scaler().fit_transform(X)
        np.testing.assert_allclose(scaled.min(axis=0), 0.0)
        np.testing.assert_allclose(scaled.max(axis=0), 1.0)

    def test_max_abs_keeps_zeros(self) -> None:
        scaled: FloatArray = FeatureScaling.MAX_ABS.build_scaler().fit_transform(X)
        np.testing.assert_allclose(np.abs(scaled).max(axis=0), 1.0)
        self.assertEqual(scaled.item(1, 1), 0.0)

    def test_robust_uses_the_median_and_interquartile_range(self) -> None:
        scaled: FloatArray = FeatureScaling.ROBUST.build_scaler().fit_transform(X)
        self.assertEqual(scaled.item(2, 0), 0.0)
        self.assertEqual(scaled.item(1, 0), -0.5)

    def test_unit_norm_works_per_sample(self) -> None:
        scaled: FloatArray = FeatureScaling.UNIT_NORM.build_scaler().fit_transform(X)
        lengths: FloatArray = np.sqrt(np.square(scaled).sum(axis=1))
        np.testing.assert_allclose(lengths, 1.0)

    def test_each_build_returns_an_independent_scaler(self) -> None:
        for scaling in FeatureScaling:
            with self.subTest(scaling=scaling):
                first: Scaler[TransformerLike] = scaling.build_scaler()
                self.assertIsNot(first, scaling.build_scaler())
