from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import MinMaxScaler

from ..parameter import ScalingParameters
from ..scalers import IncrementalScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_shape import StatisticShape


@dataclass(frozen=True, kw_only=True)
class MinMaxScalingParameters(ScalingParameters):
    """
    Each feature is rescaled to a fixed range.

    Attributes
    ----------
    feature_range : tuple[float, float]
        The lower and upper bounds of the range.
    is_clipped : bool
        Whether unseen values are clipped to the range.

    Raises
    ------
    ValueError
        If the bounds are not finite and increasing.
    """

    feature_range: tuple[float, float] = (0.0, 1.0)
    is_clipped: bool = False

    def __post_init__(self) -> None:
        lower, upper = self.feature_range
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(
                f"feature_range must be finite and increasing, got {self.feature_range}"
            )

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
        return ScalingStrategy.MIN_MAX

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
        return ScalingKind.MONOTONIC if self.is_clipped else ScalingKind.AFFINE

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
            "min_": StatisticShape.PER_FEATURE,
            "scale_": StatisticShape.PER_FEATURE,
            "data_min_": StatisticShape.PER_FEATURE,
            "data_max_": StatisticShape.PER_FEATURE,
            "data_range_": StatisticShape.PER_FEATURE,
            "n_samples_seen_": StatisticShape.SCALAR,
            "n_features_in_": StatisticShape.SCALAR,
        }

    @override
    def build_transformer(self) -> MinMaxScaler:
        """
        Build an unfitted transformer.

        Returns
        -------
        MinMaxScaler
            The transformer.
        """
        return MinMaxScaler(feature_range=self.feature_range, clip=self.is_clipped)

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
