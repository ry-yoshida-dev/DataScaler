from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final, cast

import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from .feature_validator import FeatureValidator
from .scaler_state import ScalerState
from .scaling_configuration import ScalingConfigurationLike
from .transformer import TransformerLike
from .types import FloatArray, NumericArray

PACKAGE_DIRECTORY: Final[str] = str(Path(__file__).parent)


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
        # The global configuration of scikit-learn could otherwise return DataFrames.
        _ = self._transformer.set_output(transform="default")
        self._n_features: int | None = None
        # A loaded scaler keeps the version that its statistics come from.
        self._scikit_learn_version: str = sklearn.__version__

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

        Features rejected before learning leave earlier statistics in place. Once
        learning fails, the scaler is unfitted.

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
            no observed value, or a statistic comes out impossible, such as by
            overflowing.
        """
        self._validate(X)
        self._validate_observed(X)
        self._learn_from_scratch(X)

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
            If the features are invalid, their number differs from fitting, or the
            result cannot be represented.
        """
        self._validate_fitted_shape(X)
        output: FloatArray = self._to_output(self._transformer.transform(X), X)
        self._parameters.validate_output(X, output)
        return output

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
            no observed value, a statistic comes out impossible, or the result cannot
            be represented.
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
        arrays: dict[str, NDArray[np.generic]] = self._read_statistics()
        empty_attributes: list[str] = [
            path for path in self._parameters.state_attributes if path not in arrays
        ]
        return ScalerState(
            strategy=self._parameters.strategy,
            parameters=self._parameters.encode(),
            n_features=n_features,
            arrays=arrays,
            scikit_learn_version=self._scikit_learn_version,
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
                skip_file_prefixes=(PACKAGE_DIRECTORY,),
            )
        for path, array in state.arrays.items():
            # A copy keeps this scaler apart from the state. Indexing keeps NumPy scalars, which scikit-learn needs to continue learning.
            self._write_statistic(path, cast(object, np.array(array)[()]))
        # scikit-learn tells "fitted without this statistic" from "not fitted" by presence.
        for path in state.empty_attributes:
            self._write_statistic(path, None)
        self._n_features = state.n_features
        self._scikit_learn_version = state.scikit_learn_version

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

    def _learn_from_scratch(self, X: NumericArray) -> None:
        self._n_features = None
        self._scikit_learn_version = sklearn.__version__
        try:
            _ = self._transformer.fit(X)
        except ValueError as error:
            raise ValueError(
                f"Strategy {self._parameters.strategy.name} cannot learn from these features, which may be too large or degenerate: {error}"
            ) from error
        self._record_fitted(X)

    def _record_fitted(self, X: NumericArray) -> None:
        # scikit-learn has already overwritten what it knew, so only a new fit recovers.
        self._n_features = None
        n_features: int = cast(int, X.shape[1])
        self._validate_statistics(self._read_statistics(), n_features)
        self._n_features = n_features

    def _read_statistics(self) -> dict[str, NDArray[np.generic]]:
        statistics: dict[str, NDArray[np.generic]] = {}
        for path in self._parameters.state_attributes:
            value: object = cast(
                object,
                getattr(self._resolve_owner(path), path.rsplit(".", 1)[-1], None),
            )
            if value is not None:
                statistics[path] = np.array(value)
        return statistics

    def _write_statistic(self, path: str, value: object) -> None:
        setattr(self._resolve_owner(path), path.rsplit(".", 1)[-1], value)

    def _validate_statistics(
        self, statistics: dict[str, NDArray[np.generic]], n_features: int
    ) -> None:
        for path, array in statistics.items():
            self._parameters.state_attributes[path].validate(path, array, n_features)
        self._parameters.validate_statistics(statistics)

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
        self._validate_statistics(dict(state.arrays), state.n_features)

    def _resolve_owner(self, path: str) -> object:
        if "." not in path or not isinstance(self._transformer, Pipeline):
            return self._transformer
        return self._transformer.named_steps[path.split(".", 1)[0]]

    @staticmethod
    def _to_output(result: FloatArray, X: NumericArray) -> FloatArray:
        is_floating: bool = np.issubdtype(result.dtype, np.floating)
        output: FloatArray = (
            result
            if is_floating and result is not X
            else np.array(result, dtype=result.dtype if is_floating else np.float64)
        )
        if np.isfinite(output).all():
            return output
        is_new: NDArray[np.bool_] = ~np.isfinite(output) & ~np.isnan(X)
        if is_new.any():
            raise ValueError(
                f"The result is not finite in samples {np.flatnonzero(is_new.any(axis=1))[:5].tolist()}, which the input does not explain"
            )
        return output
