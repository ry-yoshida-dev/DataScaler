from __future__ import annotations

import unittest
from dataclasses import fields
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import PowerTransformer

from data_scaler import (
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    Norm,
    NoScalingParameters,
    OutputDistribution,
    PowerMethod,
    PowerScalingParameters,
    QuantileScalingParameters,
    RobustScalingParameters,
    ScalingParameters,
    StandardScalingParameters,
    UnitNormScalingParameters,
)
from data_scaler.types import FloatArray

RANDOM: np.random.Generator = np.random.default_rng(0)
SKEWED: NDArray[np.float64] = RANDOM.lognormal(size=(300, 3))
ALL_METHODS: tuple[ScalingParameters, ...] = (
    NoScalingParameters(),
    StandardScalingParameters(),
    StandardScalingParameters(is_centered=False),
    MinMaxScalingParameters(feature_range=(-1.0, 1.0)),
    MaxAbsScalingParameters(),
    RobustScalingParameters(quantile_range=(10.0, 90.0)),
    UnitNormScalingParameters(norm=Norm.L1),
    QuantileScalingParameters(
        n_quantiles=50, output_distribution=OutputDistribution.NORMAL
    ),
    PowerScalingParameters(),
    PowerScalingParameters(method=PowerMethod.BOX_COX, is_standardized=False),
)


class TestEffects(unittest.TestCase):
    def test_min_max_range_and_clipping(self) -> None:
        scaled: FloatArray = (
            MinMaxScalingParameters(feature_range=(-1.0, 1.0))
            .build_scaler()
            .fit_transform(SKEWED)
        )
        np.testing.assert_allclose(scaled.min(axis=0), -1.0)
        np.testing.assert_allclose(scaled.max(axis=0), 1.0)
        clipped = MinMaxScalingParameters(is_clipped=True).build_scaler()
        clipped.fit(np.array([[0.0], [1.0]]))
        np.testing.assert_array_equal(clipped.transform(np.array([[5.0]])), [[1.0]])

    def test_standard_options(self) -> None:
        not_centered: FloatArray = (
            StandardScalingParameters(is_centered=False)
            .build_scaler()
            .fit_transform(SKEWED)
        )
        np.testing.assert_allclose(not_centered.std(axis=0), 1.0)
        self.assertGreater(float(not_centered.mean()), 0.1)

    def test_quantile_output_distributions(self) -> None:
        uniform: FloatArray = (
            QuantileScalingParameters(n_quantiles=100)
            .build_scaler()
            .fit_transform(SKEWED)
        )
        np.testing.assert_allclose(uniform.mean(axis=0), 0.5, atol=0.01)
        normal: FloatArray = (
            QuantileScalingParameters(
                n_quantiles=100, output_distribution=OutputDistribution.NORMAL
            )
            .build_scaler()
            .fit_transform(SKEWED)
        )
        np.testing.assert_allclose(normal.mean(axis=0), 0.0, atol=0.05)

    def test_power_matches_the_standardized_estimator(self) -> None:
        scaled: FloatArray = (
            PowerScalingParameters().build_scaler().fit_transform(SKEWED)
        )
        np.testing.assert_allclose(
            scaled, PowerTransformer().fit(SKEWED).transform(SKEWED)
        )

    def test_unit_norm_measures(self) -> None:
        scaled: FloatArray = (
            UnitNormScalingParameters(norm=Norm.MAX)
            .build_scaler()
            .fit_transform(SKEWED)
        )
        np.testing.assert_allclose(np.abs(scaled).max(axis=1), 1.0)


class TestPipeline(unittest.TestCase):
    def test_estimators_are_usable_as_raw_transformers(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(parameters=parameters):
                estimator = parameters.build_transformer()
                _ = estimator.fit(SKEWED)
                self.assertEqual(estimator.transform(SKEWED).shape, SKEWED.shape)


class TestParameterValidation(unittest.TestCase):
    def test_every_field_is_keyword_only(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(parameters=parameters):
                self.assertTrue(all(field.kw_only for field in fields(parameters)))

    def test_rejects_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            _ = MinMaxScalingParameters(feature_range=(1.0, 0.0))
        with self.assertRaises(ValueError):
            _ = MinMaxScalingParameters(feature_range=(float("nan"), 1.0))
        with self.assertRaises(ValueError):
            _ = MinMaxScalingParameters(feature_range=(0.0, float("inf")))
        with self.assertRaises(ValueError):
            _ = RobustScalingParameters(quantile_range=(75.0, 25.0))
        with self.assertRaises(ValueError):
            _ = RobustScalingParameters(quantile_range=(-1.0, 50.0))

    def test_rejects_invalid_quantile_counts(self) -> None:
        with self.assertRaises(ValueError):
            _ = QuantileScalingParameters(n_quantiles=1)
        with self.assertRaises(ValueError):
            _ = QuantileScalingParameters(n_quantiles=100, subsample=10)
        with self.assertRaises(TypeError):
            _ = QuantileScalingParameters(n_quantiles=cast(int, 1.5))
        with self.assertRaises(TypeError):
            _ = QuantileScalingParameters(
                n_quantiles=cast(int, cast(object, np.int64(50)))
            )
        with self.assertRaises(ValueError):
            _ = QuantileScalingParameters(random_state=-1)

    def test_unit_norm_rejects_missing_values(self) -> None:
        with self.assertRaises(ValueError):
            _ = UnitNormScalingParameters(is_nan_allowed=True)
