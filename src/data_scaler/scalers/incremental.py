from __future__ import annotations

from ..scaling_configuration import ScalingConfigurationLike
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
            a feature of the batch has no observed value, or a statistic overflows,
            which also discards what was learned.
        """
        if self._n_features is None:
            self._validate(X)
            self._validate_observed(X)
            # A failed fit may have left statistics behind, which fit discards.
            _ = self._transformer.fit(X)
        else:
            self._validate_fitted_shape(X)
            self._validate_observed(X)
            _ = self._incremental_transformer.partial_fit(X)
        self._record_fitted(X)
