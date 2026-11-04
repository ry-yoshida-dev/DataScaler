from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import StandardScaler

from ..parameter import ScalingParameters
from ..scalers import IncrementalScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_shape import StatisticShape


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
    def state_attributes(self) -> Mapping[str, StatisticShape]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        Mapping[str, StatisticShape]
            The shape of each attribute by name.
        """
        return {
            "mean_": StatisticShape.PER_FEATURE,
            "var_": StatisticShape.PER_FEATURE,
            "scale_": StatisticShape.PER_FEATURE,
            "n_samples_seen_": StatisticShape.SCALAR_OR_PER_FEATURE,
            "n_features_in_": StatisticShape.SCALAR,
        }

    @property
    @override
    def empty_state_attributes(self) -> frozenset[str]:
        """
        Get the state attributes that this configuration leaves empty after fitting.

        Returns
        -------
        frozenset[str]
            The attribute names.
        """
        if self.is_scaled:
            return frozenset()
        if self.is_centered:
            return frozenset({"var_", "scale_"})
        return frozenset({"mean_", "var_", "scale_"})

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
