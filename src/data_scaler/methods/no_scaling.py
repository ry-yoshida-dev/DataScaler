from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import FunctionTransformer

from ..invertible_scaler import InvertibleScaler
from ..parameter import ScalingParameters
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy


@dataclass(frozen=True, kw_only=True)
class NoScalingParameters(ScalingParameters):
    """
    Features are used as they are.
    """

    @property
    @override
    def strategy(self) -> ScalingStrategy:
        """
        Get the identifier of this strategy.

        Returns
        -------
        ScalingStrategy
            The strategy.
        """
        return ScalingStrategy.NONE

    @property
    @override
    def kind(self) -> ScalingKind:
        """
        Get what this strategy does to the values.

        Returns
        -------
        ScalingKind
            The kind.
        """
        return ScalingKind.IDENTITY

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
    def build_transformer(self) -> FunctionTransformer:
        """
        Build an identity transformer.

        Returns
        -------
        FunctionTransformer
            The transformer.
        """
        return FunctionTransformer()

    @override
    def build_scaler(self) -> InvertibleScaler:
        """
        Build an unfitted scaler.

        Returns
        -------
        InvertibleScaler
            The scaler.
        """
        return InvertibleScaler(self, self.build_transformer())
