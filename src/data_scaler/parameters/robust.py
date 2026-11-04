from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import RobustScaler

from ..parameter import ScalingParameters
from ..scalers import InvertibleScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_constraint import StatisticConstraint
from ..statistic_shape import StatisticShape
from ..statistic_spec import StatisticSpec


@dataclass(frozen=True, kw_only=True)
class RobustScalingParameters(ScalingParameters):
    """
    Each feature is centered on its median and scaled by a quantile range.

    Outliers barely influence these statistics, but they remain outliers in the
    output.

    Attributes
    ----------
    is_centered : bool
        Whether to subtract the median.
    is_scaled : bool
        Whether to divide by the quantile range.
    quantile_range : tuple[float, float]
        The lower and upper percentiles of the range.
    is_unit_variance : bool
        Whether normally distributed features end up with unit variance.

    Raises
    ------
    ValueError
        If the percentiles are not increasing within [0, 100], or unit variance is
        requested with a percentile of 0 or 100, whose normal quantile is infinite.
    """

    is_centered: bool = True
    is_scaled: bool = True
    quantile_range: tuple[float, float] = (25.0, 75.0)
    is_unit_variance: bool = False

    @override
    def _validate_values(self) -> None:
        lower: float
        upper: float
        lower, upper = self.quantile_range
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError(
                f"quantile_range must be increasing within [0, 100], got {self.quantile_range}"
            )
        if (
            self.is_scaled
            and self.is_unit_variance
            and (lower == 0.0 or upper == 100.0)
        ):
            raise ValueError(
                f"is_unit_variance needs a quantile_range within (0, 100), got {self.quantile_range}"
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
        return ScalingStrategy.ROBUST

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
    def state_attributes(self) -> Mapping[str, StatisticSpec]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        Mapping[str, StatisticSpec]
            The shape of each attribute by name.
        """
        return {
            "center_": StatisticSpec(StatisticShape.PER_FEATURE),
            "scale_": StatisticSpec(
                StatisticShape.PER_FEATURE, StatisticConstraint.POSITIVE
            ),
            "n_features_in_": StatisticSpec(
                StatisticShape.SCALAR, StatisticConstraint.FEATURE_COUNT
            ),
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
        centering: frozenset[str] = (
            frozenset() if self.is_centered else frozenset({"center_"})
        )
        scaling: frozenset[str] = (
            frozenset() if self.is_scaled else frozenset({"scale_"})
        )
        return centering | scaling

    @override
    def build_transformer(self) -> RobustScaler:
        """
        Build an unfitted transformer.

        Returns
        -------
        RobustScaler
            The transformer.
        """
        return RobustScaler(
            with_centering=self.is_centered,
            with_scaling=self.is_scaled,
            quantile_range=self.quantile_range,
            unit_variance=self.is_unit_variance,
        )

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
