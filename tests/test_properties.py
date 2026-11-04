from __future__ import annotations

import contextlib
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypothesis_numpy
from numpy.typing import NDArray

from data_scaler import (
    IncrementalScaler,
    InvertibleScaler,
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    Norm,
    NoScalingParameters,
    PowerMethod,
    PowerScalingParameters,
    QuantileScalingParameters,
    RobustScalingParameters,
    Scaler,
    ScalerLoader,
    ScalingParameters,
    StandardScalingParameters,
    UnitNormScalingParameters,
)

CONFIGURATIONS: tuple[ScalingParameters, ...] = (
    NoScalingParameters(),
    StandardScalingParameters(),
    StandardScalingParameters(is_centered=False),
    StandardScalingParameters(is_scaled=False),
    MinMaxScalingParameters(),
    MinMaxScalingParameters(feature_range=(-3.0, 7.5), is_clipped=True),
    MaxAbsScalingParameters(),
    RobustScalingParameters(),
    RobustScalingParameters(quantile_range=(1.0, 99.0), is_unit_variance=True),
    UnitNormScalingParameters(),
    UnitNormScalingParameters(norm=Norm.L1),
    UnitNormScalingParameters(norm=Norm.MAX),
    QuantileScalingParameters(n_quantiles=10),
    PowerScalingParameters(),
    PowerScalingParameters(method=PowerMethod.BOX_COX, is_standardized=False),
)
PROFILE = settings(max_examples=150, deadline=None, derandomize=True, database=None)


def feature_matrices() -> st.SearchStrategy[NDArray[np.float64]]:
    """
    Generate finite matrices that include constant features and extreme magnitudes.

    Returns
    -------
    st.SearchStrategy[NDArray[np.float64]]
        The strategy.
    """
    return hypothesis_numpy.arrays(
        np.float64,
        st.tuples(st.integers(1, 12), st.integers(1, 3)),
        elements=st.floats(allow_nan=False, allow_infinity=False, width=64),
    )


def fit_or_none(scaler: Scaler, X: NDArray[np.float64]) -> bool:
    """
    Fit a scaler, for properties that hold whenever it agreed to learn.

    Parameters
    ----------
    scaler : Scaler
        The scaler.
    X : NDArray[np.float64]
        The features.

    Returns
    -------
    bool
        Whether the scaler learned from the features.
    """
    try:
        scaler.fit(X)
    except ValueError:
        return False
    return True


class TestProperties(unittest.TestCase):
    @PROFILE
    @given(X=feature_matrices(), parameters=st.sampled_from(CONFIGURATIONS))
    def test_whatever_was_learned_can_be_saved_and_loaded(
        self, X: NDArray[np.float64], parameters: ScalingParameters
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler: Scaler = parameters.build_scaler()
            if not fit_or_none(scaler, X):
                return
            with tempfile.TemporaryDirectory() as directory:
                path: Path = Path(directory) / "scaler.npz"
                scaler.save(path)
                restored: Scaler = ScalerLoader.load(path)
            self.assertIs(type(restored), type(scaler))
            self.assertEqual(restored.n_features, scaler.n_features)

    @PROFILE
    @given(
        X=feature_matrices(),
        unseen=feature_matrices(),
        parameters=st.sampled_from(CONFIGURATIONS),
    )
    def test_finite_inputs_give_finite_outputs_or_an_error(
        self,
        X: NDArray[np.float64],
        unseen: NDArray[np.float64],
        parameters: ScalingParameters,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler: Scaler = parameters.build_scaler()
            if not fit_or_none(scaler, X):
                return
            probe: NDArray[np.float64] = np.resize(
                unseen, (unseen.shape[0], X.shape[1])
            )
            with contextlib.suppress(ValueError):
                self.assertTrue(np.isfinite(scaler.transform(probe)).all())
            if isinstance(scaler, InvertibleScaler):
                with contextlib.suppress(ValueError):
                    self.assertTrue(np.isfinite(scaler.inverse_transform(probe)).all())

    @PROFILE
    @given(
        first=feature_matrices(),
        second=feature_matrices(),
        parameters=st.sampled_from(
            (
                StandardScalingParameters(),
                MinMaxScalingParameters(),
                MaxAbsScalingParameters(),
            )
        ),
    )
    def test_batches_keep_the_state_loadable(
        self,
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        parameters: StandardScalingParameters
        | MinMaxScalingParameters
        | MaxAbsScalingParameters,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler: IncrementalScaler = parameters.build_scaler()
            batch: NDArray[np.float64] = np.resize(
                second, (second.shape[0], first.shape[1])
            )
            try:
                scaler.partial_fit(first)
                scaler.partial_fit(batch)
            except ValueError:
                return
            restored: Scaler = ScalerLoader.from_state(scaler.to_state())
            self.assertEqual(restored.n_features, first.shape[1])
