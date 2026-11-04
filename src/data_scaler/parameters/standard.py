from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import StandardScaler

from ..parameter import ScalingParameters
from ..scalers import IncrementalScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy


@dataclass(frozen=True, kw_only=True)
class StandardScalingParameters(ScalingParameters):
    """
    Each feature has zero mean and unit variance.

    Attributes
    ----------
    is_centered : bool
        Whether to subtract the mean.
    is_scaled : bool
        Whether to divide by the standard deviation.
    """

    is_centered: bool = True
    is_scaled: bool = True

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
        return ScalingStrategy.STANDARD

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
        if self.is_centered:
            return ScalingKind.AFFINE
        return ScalingKind.PROPORTIONAL if self.is_scaled else ScalingKind.IDENTITY

    @property
    @override
    def state_attributes(self) -> tuple[str, ...]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        tuple[str, ...]
            The attribute names.
        """
        return ("mean_", "var_", "scale_", "n_samples_seen_", "n_features_in_")

    @override
    def build_transformer(self) -> StandardScaler:
        """
        Build an unfitted transformer.

        Returns
        -------
        StandardScaler
            The transformer.
        """
        return StandardScaler(with_mean=self.is_centered, with_std=self.is_scaled)

    @override
    def build_scaler(self) -> IncrementalScaler:
        """
        Build an unfitted scaler.

        Returns
        -------
        IncrementalScaler
            The scaler.
        """
        return IncrementalScaler(self, self.build_transformer())
