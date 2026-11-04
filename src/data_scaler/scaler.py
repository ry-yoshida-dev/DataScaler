from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from .feature_validator import FeatureValidator
from .scaler_state import ScalerState
from .scaling_configuration import ScalingConfigurationLike
from .statistic_shape import StatisticShape
from .transformer import TransformerLike
from .types import FloatArray, NumericArray


class Scaler:
    """
    Applies a scaling strategy with strict input checks.

    Inputs are never modified. Unlike the wrapped transformer, a scaler rejects a
    changed number of features with a clear error and can be saved without pickle.

    Parameters
    ----------
    parameters : ScalingConfigurationLike
        The configuration this scaler was built from.
    transformer : TransformerLike
        The wrapped transformer.
    """

    def __init__(
        self, parameters: ScalingConfigurationLike, transformer: TransformerLike
    ) -> None:
        self._parameters: ScalingConfigurationLike = parameters
        self._transformer: TransformerLike = transformer
        self._n_features: int | None = None

    @property
    def parameters(self) -> ScalingConfigurationLike:
        """
        Get the configuration this scaler was built from.

        Returns
        -------
        ScalingConfigurationLike
            The parameters.
        """
        return self._parameters

    @property
    def n_features(self) -> int:
        """
        Get the number of features seen during fitting.

        Returns
        -------
        int
            The number of features.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        if self._n_features is None:
            raise NotFittedError("Scaler is not fitted yet. Call fit() first.")
        return self._n_features

    def fit(self, X: NumericArray) -> None:
        """
        Learn the statistics of the samples, discarding earlier ones.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).

        Raises
        ------
        TypeError
            If the features are not a numeric NumPy array.
        ValueError
            If the features are not a valid matrix for this strategy, a feature has
            no observed value, or a statistic overflows.
        """
        self._n_features = None
        self._validate(X)
        self._validate_observed(X)
        _ = self._transformer.fit(X)
        self._record_fitted(X)

    def transform(self, X: NumericArray) -> FloatArray:
        """
        Scale features with the fitted statistics.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The scaled features in a new array.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        TypeError
            If the features are not a numeric NumPy array.
        ValueError
            If the features are invalid or their number differs from fitting.
        """
        self._validate_fitted_shape(X)
        return self._to_new_float_array(self._transformer.transform(X), X)

    def fit_transform(self, X: NumericArray) -> FloatArray:
        """
        Learn the statistics of the samples and scale them.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The scaled features in a new array.

        Raises
        ------
        TypeError
            If the features are not a numeric NumPy array.
        ValueError
            If the features are not a valid matrix for this strategy, a feature has
            no observed value, or a statistic overflows.
        """
        self.fit(X)
        return self.transform(X)

    def to_state(self) -> ScalerState:
        """
        Export the configuration and fitted statistics.

        Returns
        -------
        ScalerState
            The state, a read-only copy which ScalerLoader can restore.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        n_features: int = self.n_features
        arrays: dict[str, NDArray[np.generic]] = {}
        empty_attributes: list[str] = []
        for path in self._parameters.state_attributes:
            value: object = self._read_statistic(path)
            if value is None:
                empty_attributes.append(path)
            else:
                array: NDArray[np.generic] = np.array(value)
                array.setflags(write=False)
                arrays[path] = array
        return ScalerState(
            strategy=self._parameters.strategy,
            parameters=self._parameters.encode(),
            n_features=n_features,
            arrays=arrays,
            scikit_learn_version=sklearn.__version__,
            empty_attributes=tuple(empty_attributes),
        )

    def restore_state(self, state: ScalerState) -> None:
        """
        Load fitted statistics into this scaler.

        Parameters
        ----------
        state : ScalerState
            The state, exported from a scaler with the same parameters.

        Raises
        ------
        ValueError
            If the state belongs to another configuration, its attributes differ
            from what this strategy learns, or a statistic has an impossible shape
            or value.
        """
        self._validate_state(state)
        if state.scikit_learn_version != sklearn.__version__:
            warnings.warn(
                f"State saved with scikit-learn {state.scikit_learn_version} is loaded with {sklearn.__version__}, whose fitted attributes may differ",
                UserWarning,
                stacklevel=2,
            )
        for path, array in state.arrays.items():
            # A copy keeps this scaler apart from the state. Indexing keeps NumPy scalars, which scikit-learn needs to continue learning.
            value: object = cast(object, np.array(array)[()])
            setattr(self._resolve_owner(path), path.rsplit(".", 1)[-1], value)
        # scikit-learn tells "fitted without this statistic" from "not fitted" by presence.
        for path in state.empty_attributes:
            setattr(self._resolve_owner(path), path.rsplit(".", 1)[-1], None)
        self._n_features = state.n_features

    def save(self, path: Path) -> None:
        """
        Write the configuration and fitted statistics to a file, without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ScalerLoader reads it back.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        self.to_state().save(path)

    def _record_fitted(self, X: NumericArray) -> None:
        for path in self._parameters.state_attributes:
            value: object = self._read_statistic(path)
            if value is not None and not np.isfinite(np.array(value)).all():
                # scikit-learn has already overwritten what it knew, so only a new fit recovers.
                self._n_features = None
                raise ValueError(
                    f"Statistic {path} overflowed, because the features are too large to learn from"
                )
        self._n_features = X.shape[1]

    def _read_statistic(self, path: str) -> object:
        return cast(
            object,
            getattr(self._resolve_owner(path), path.rsplit(".", 1)[-1], None),
        )

    def _validate(self, X: NumericArray) -> None:
        FeatureValidator.require_matrix(
            X, is_nan_allowed=self._parameters.is_nan_allowed
        )

    def _validate_observed(self, X: NumericArray) -> None:
        if self._parameters.is_nan_allowed:
            FeatureValidator.require_observed_features(X)

    def _validate_fitted_shape(self, X: NumericArray) -> None:
        n_features: int = self.n_features
        self._validate(X)
        if X.shape[1] != n_features:
            raise ValueError(
                f"X must have {n_features} features as during fit, got {X.shape[1]}"
            )

    def _validate_state(self, state: ScalerState) -> None:
        if state.strategy is not self._parameters.strategy or (
            dict(state.parameters) != self._parameters.encode()
        ):
            raise ValueError(
                f"State of {state.strategy.name} does not match the parameters {self._parameters}"
            )
        saved: list[str] = [*state.arrays, *state.empty_attributes]
        if sorted(saved) != sorted(self._parameters.state_attributes):
            raise ValueError(
                f"State attributes {sorted(saved)} differ from {sorted(self._parameters.state_attributes)}"
            )
        empty: frozenset[str] = self._parameters.empty_state_attributes
        if frozenset(state.empty_attributes) != empty:
            raise ValueError(
                f"State leaves {sorted(state.empty_attributes)} empty, but this configuration leaves {sorted(empty)} empty"
            )
        shapes: Mapping[str, StatisticShape] = self._parameters.state_attributes
        for path, array in state.arrays.items():
            if not shapes[path].matches(array.shape, state.n_features):
                raise ValueError(
                    f"State {path} has shape {array.shape}, which does not fit {shapes[path].name} with {state.n_features} features"
                )
            if path.endswith("n_features_in_") and not np.array_equal(
                array, state.n_features
            ):
                raise ValueError(
                    f"State {path} disagrees with its {state.n_features} features"
                )
            # scikit-learn replaces zero scales with one, so anything else is damage.
            if path.endswith("scale_") and not np.greater(array, 0).all():
                raise ValueError(f"State {path} must be positive")
            # Only observed features are learned from, so every count is at least one.
            if path.endswith("n_samples_seen_") and not np.greater(array, 0).all():
                raise ValueError(f"State {path} must be positive")
            if path.endswith("var_") and np.less(array, 0).any():
                raise ValueError(f"State {path} must not be negative")
        self._parameters.validate_statistics(state.arrays)

    def _resolve_owner(self, path: str) -> object:
        if "." not in path or not isinstance(self._transformer, Pipeline):
            return self._transformer
        return self._transformer.named_steps[path.split(".", 1)[0]]

    @staticmethod
    def _to_new_float_array(result: FloatArray, X: NumericArray) -> FloatArray:
        is_floating: bool = np.issubdtype(result.dtype, np.floating)
        if is_floating and result is not X:
            return result
        return np.array(result, dtype=result.dtype if is_floating else np.float64)
