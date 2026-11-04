from __future__ import annotations

import math
from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import MinMaxScaler

from ..incremental_scaler import IncrementalScaler
from ..parameter import ScalingParameters
from ..scaling_kind import ScalingKind


@dataclass(frozen=True)
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
    def state_attributes(self) -> tuple[str, ...]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        tuple[str, ...]
            The attribute names.
        """
        return (
            "min_",
            "scale_",
            "data_min_",
            "data_max_",
            "data_range_",
            "n_samples_seen_",
            "n_features_in_",
        )

    @override
    def build_estimator(self) -> MinMaxScaler:
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
        return IncrementalScaler(self, self.build_estimator())
