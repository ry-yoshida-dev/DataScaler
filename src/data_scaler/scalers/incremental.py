from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from ..feature_validator import FeatureValidator
from ..scaling_configuration import ScalingConfigurationLike
from ..statistic_shape import StatisticShape
from ..transformers import IncrementalTransformerLike
from ..types import NumericArray
from .invertible import InvertibleScaler


class IncrementalScaler(InvertibleScaler):
    """
    Scaler that can learn from data too large to hold in memory at once.

    Parameters
    ----------
    parameters : ScalingConfigurationLike
        The configuration this scaler was built from.
    transformer : IncrementalTransformerLike
        The wrapped transformer.
    """

    def __init__(
        self,
        parameters: ScalingConfigurationLike,
        transformer: IncrementalTransformerLike,
    ) -> None:
        super().__init__(parameters, transformer)
        self._incremental_transformer: IncrementalTransformerLike = transformer

    def partial_fit(self, X: NumericArray) -> None:
        """
        Update the statistics with a batch of samples.

        A feature with only NaN in a later batch keeps its statistics, so samples can
        arrive one at a time. The first batch has to observe every feature.

        Parameters
        ----------
        X : NumericArray
            The feature matrix of the batch with shape (n_samples, n_features).

        Raises
        ------
        TypeError
            If the features are not a numeric NumPy array.
        ValueError
            If the features are invalid, their number differs from earlier batches,
            a feature of the first batch has no observed value, or a statistic comes
            out impossible, such as by overflowing, which also discards what was
            learned.
        """
        if self._n_features is None:
            self._validate(X)
            self._validate_observed(X)
            # A failed fit may have left statistics behind, which fit discards.
            self._learn_from_scratch(X)
            return
        self._validate_fitted_shape(X)
        unobserved: NDArray[np.intp] = (
            FeatureValidator.find_unobserved_features(X)
            if self._parameters.is_nan_allowed
            else np.empty(0, dtype=np.intp)
        )
        if unobserved.size == 0:
            _ = self._incremental_transformer.partial_fit(X)
        else:
            self._update_observed_features(X, unobserved)
        self._record_fitted(X)

    def _update_observed_features(
        self, X: NumericArray, unobserved: NDArray[np.intp]
    ) -> None:
        # scikit-learn turns the statistics of a feature without new values into NaN.
        before: dict[str, NDArray[np.generic]] = self._read_statistics()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _ = self._incremental_transformer.partial_fit(X)
        after: dict[str, NDArray[np.generic]] = self._read_statistics()
        for path, array in after.items():
            shape: StatisticShape = self._parameters.state_attributes[path].shape
            if shape is StatisticShape.PER_FEATURE:
                array[unobserved] = before[path][unobserved]
                self._write_statistic(path, array)
