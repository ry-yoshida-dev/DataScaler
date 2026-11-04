from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from data_scaler import (
    FeatureScaling,
    MinMaxScalingParameters,
    Norm,
    OutputDistribution,
    PowerMethod,
    PowerScalingParameters,
    QuantileScalingParameters,
    RobustScalingParameters,
    Scaler,
    ScalingKind,
    ScalingParameters,
    StandardScalingParameters,
    TransformerLike,
    UnitNormScalingParameters,
)
from data_scaler.types import FloatArray

N_FEATURES: int = 3
_generator: np.random.Generator = np.random.default_rng(0)
TRAINING: FloatArray = _generator.lognormal(size=(300, N_FEATURES)) + 0.1
# Values outside the training range expose clipping that training values would hide.
PROBE: FloatArray = np.vstack([TRAINING, TRAINING * 3.0, TRAINING * 0.01])

CASES: list[tuple[FeatureScaling | ScalingParameters, ScalingKind]] = [
    (FeatureScaling.NONE, ScalingKind.IDENTITY),
    (FeatureScaling.MAX_ABS, ScalingKind.PROPORTIONAL),
    (FeatureScaling.STANDARD, ScalingKind.AFFINE),
    (FeatureScaling.MIN_MAX, ScalingKind.AFFINE),
    (FeatureScaling.ROBUST, ScalingKind.AFFINE),
    (FeatureScaling.UNIT_NORM, ScalingKind.SAMPLE_WISE),
    (UnitNormScalingParameters(norm=Norm.L1), ScalingKind.SAMPLE_WISE),
    (UnitNormScalingParameters(norm=Norm.MAX), ScalingKind.SAMPLE_WISE),
    (MinMaxScalingParameters(feature_range=(-1.0, 1.0)), ScalingKind.AFFINE),
    (MinMaxScalingParameters(is_clipped=True), ScalingKind.MONOTONIC),
    (QuantileScalingParameters(n_quantiles=100), ScalingKind.MONOTONIC),
    (
        QuantileScalingParameters(
            n_quantiles=100, output_distribution=OutputDistribution.NORMAL
        ),
        ScalingKind.MONOTONIC,
    ),
    (PowerScalingParameters(), ScalingKind.MONOTONIC),
    (PowerScalingParameters(is_standardized=False), ScalingKind.MONOTONIC),
    (PowerScalingParameters(method=PowerMethod.BOX_COX), ScalingKind.MONOTONIC),
    (StandardScalingParameters(is_centered=False), ScalingKind.PROPORTIONAL),
    (StandardScalingParameters(is_scaled=False), ScalingKind.AFFINE),
    (
        StandardScalingParameters(is_centered=False, is_scaled=False),
        ScalingKind.IDENTITY,
    ),
    (RobustScalingParameters(is_centered=False), ScalingKind.PROPORTIONAL),
    (RobustScalingParameters(is_scaled=False), ScalingKind.AFFINE),
    (
        RobustScalingParameters(is_centered=False, is_scaled=False),
        ScalingKind.IDENTITY,
    ),
]


class TestScalingKind(unittest.TestCase):
    def test_kind_follows_the_configuration(self) -> None:
        for strategy, kind in CASES:
            with self.subTest(strategy=strategy):
                self.assertIs(strategy.kind, kind)

    def test_kind_is_the_strictest_description_of_the_transformation(self) -> None:
        for strategy, _ in CASES:
            with self.subTest(strategy=strategy):
                scaler: Scaler[TransformerLike] = strategy.build_scaler()
                scaler.fit(TRAINING)
                scaled: FloatArray = scaler.transform(PROBE)
                for feature in range(N_FEATURES):
                    self.assertIs(
                        self._observe_kind(PROBE[:, feature], scaled[:, feature]),
                        strategy.kind,
                    )

    @staticmethod
    def _observe_kind(original: FloatArray, scaled: FloatArray) -> ScalingKind:
        design: FloatArray = np.column_stack([original, np.ones_like(original)])
        coefficients: FloatArray = np.linalg.lstsq(design, scaled, rcond=None)[0]
        slope: float = coefficients.item(0)
        intercept: float = coefficients.item(1)
        fitted: FloatArray = slope * original + intercept
        is_affine: bool = bool(np.allclose(fitted, scaled))
        if not is_affine:
            order: NDArray[np.intp] = np.argsort(original)
            is_monotonic: bool = bool((np.diff(scaled[order]) >= -1e-9).all())
            return ScalingKind.MONOTONIC if is_monotonic else ScalingKind.SAMPLE_WISE
        if not np.isclose(intercept, 0.0, atol=1e-8):
            return ScalingKind.AFFINE
        return (
            ScalingKind.IDENTITY if np.isclose(slope, 1.0) else ScalingKind.PROPORTIONAL
        )
