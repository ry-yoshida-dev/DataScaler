from __future__ import annotations

from ..scaler import Scaler
from ..scaling_configuration import ScalingConfigurationLike
from ..transformers import InvertibleTransformerLike
from ..types import FloatArray, NumericArray


class InvertibleScaler(Scaler):
    """
    Scaler whose transformation can be undone.

    Strategies that discard information offer no inverse, which the type checker
    rejects instead of a failure at runtime.

    Parameters
    ----------
    parameters : ScalingConfigurationLike
        The configuration this scaler was built from.
    transformer : InvertibleTransformerLike
        The wrapped transformer.
    """

    def __init__(
        self,
        parameters: ScalingConfigurationLike,
        transformer: InvertibleTransformerLike,
    ) -> None:
        super().__init__(parameters, transformer)
        self._invertible_transformer: InvertibleTransformerLike = transformer

    def inverse_transform(self, X: NumericArray) -> FloatArray:
        """
        Return scaled features to their original scale.

        Parameters
        ----------
        X : NumericArray
            The scaled features with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The features on their original scale, in a new array.

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
        return self._to_new_float_array(
            self._invertible_transformer.inverse_transform(X), X
        )
