from __future__ import annotations

from .invertible_scaler import InvertibleScaler
from .transformers import IncrementalTransformerLike
from .types import NumericArray


class IncrementalScaler(InvertibleScaler[IncrementalTransformerLike]):
    """
    Scaler that can learn from data too large to hold in memory at once.
    """

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
            If the features are not numeric.
        ValueError
            If the features are invalid or their number differs from earlier batches.
        """
        if self._n_features is None:
            self._validate(X)
            # A failed fit may have left statistics behind, which fit discards.
            _ = self._estimator.fit(X)
        else:
            self._validate_fitted_shape(X)
            _ = self._estimator.partial_fit(X)
        self._record_fitted(X)
