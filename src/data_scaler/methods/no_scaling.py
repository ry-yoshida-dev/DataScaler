from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import FunctionTransformer

from ..invertible_scaler import InvertibleScaler
from ..parameter import ScalingParameters
from ..transformers import InvertibleTransformerLike


@dataclass(frozen=True)
class NoScalingParameters(ScalingParameters):
    """
    Features are used as they are.
    """

    @property
    @override
    def state_attributes(self) -> tuple[str, ...]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        tuple[str, ...]
            No attributes, because nothing is learned.
        """
        return ()

    @override
    def build_estimator(self) -> FunctionTransformer:
        """
        Build an identity transformer.

        Returns
        -------
        FunctionTransformer
            The transformer.
        """
        return FunctionTransformer()

    @override
    def build_scaler(self) -> InvertibleScaler[InvertibleTransformerLike]:
        """
        Build an unfitted scaler.

        Returns
        -------
        InvertibleScaler[InvertibleTransformerLike]
            The scaler.
        """
        return InvertibleScaler(self, self.build_estimator())
