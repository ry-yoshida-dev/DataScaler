from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import QuantileTransformer

from ..options import OutputDistribution
from ..parameter import ScalingParameters
from ..scalers import InvertibleScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy


@dataclass(frozen=True, kw_only=True)
class QuantileScalingParameters(ScalingParameters):
    """
    Each feature is mapped through its empirical distribution.

    This is the strongest protection against outliers and skew, at the cost of
    distorting linear relationships between features.

    Attributes
    ----------
    n_quantiles : int
        Number of quantiles estimated; at most the number of training samples.
    output_distribution : OutputDistribution
        The distribution of the output.
    subsample : int | None
        Samples used to estimate the quantiles; all samples when None.
    random_state : int | None
        Seed for subsampling.

    Raises
    ------
    TypeError
        If a count or the seed is not an integer.
    ValueError
        If a count or the seed is out of range.
    """

    n_quantiles: int = 1000
    output_distribution: OutputDistribution = OutputDistribution.UNIFORM
    subsample: int | None = 10000
    random_state: int | None = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("n_quantiles", self.n_quantiles),
            ("subsample", self.subsample),
            ("random_state", self.random_state),
        ):
            if value is not None and type(value) is not int:
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2")
        if self.subsample is not None and self.subsample < self.n_quantiles:
            raise ValueError("subsample must not be below n_quantiles")
        if self.random_state is not None and self.random_state < 0:
            raise ValueError("random_state must not be negative")

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
        return ScalingStrategy.QUANTILE

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
        return ScalingKind.MONOTONIC

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
        return ("quantiles_", "references_", "n_quantiles_", "n_features_in_")

    @override
    def build_transformer(self) -> QuantileTransformer:
        """
        Build an unfitted transformer.

        Returns
        -------
        QuantileTransformer
            The transformer.
        """
        return QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution.value,
            subsample=self.subsample,
            random_state=self.random_state,
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
