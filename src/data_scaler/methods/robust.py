from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import RobustScaler

from ..invertible_scaler import InvertibleScaler
from ..parameter import ScalingParameters
from ..transformers import InvertibleTransformerLike


@dataclass(frozen=True)
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
        If the percentiles are not increasing within [0, 100].
    """

    is_centered: bool = True
    is_scaled: bool = True
    quantile_range: tuple[float, float] = (25.0, 75.0)
    is_unit_variance: bool = False

    def __post_init__(self) -> None:
        lower, upper = self.quantile_range
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError(
                f"quantile_range must be increasing within [0, 100], got {self.quantile_range}"
            )

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
        return ("center_", "scale_", "n_features_in_")

    @override
    def build_estimator(self) -> RobustScaler:
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
    def build_scaler(self) -> InvertibleScaler[InvertibleTransformerLike]:
        """
        Build an unfitted scaler.

        Returns
        -------
        InvertibleScaler[InvertibleTransformerLike]
            The scaler.
        """
        return InvertibleScaler(self, self.build_estimator())
