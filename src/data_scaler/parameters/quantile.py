from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, cast, override

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import QuantileTransformer

from ..options import OutputDistribution
from ..parameter import ScalingParameters
from ..scalers import InvertibleScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_constraint import StatisticConstraint
from ..statistic_shape import StatisticShape
from ..statistic_spec import StatisticSpec
from ..types import FloatArray

MAX_RANDOM_STATE: Final[int] = 2**32 - 1


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
    ValueError
        If a count or the seed is out of range.
    """

    n_quantiles: int = 1000
    output_distribution: OutputDistribution = OutputDistribution.UNIFORM
    subsample: int | None = 10000
    random_state: int | None = 0

    @override
    def _validate_values(self) -> None:
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2")
        if self.subsample is not None and self.subsample < self.n_quantiles:
            raise ValueError("subsample must not be below n_quantiles")
        if self.random_state is not None and not (
            0 <= self.random_state <= MAX_RANDOM_STATE
        ):
            raise ValueError(
                f"random_state must be within [0, {MAX_RANDOM_STATE}], got {self.random_state}"
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
    def state_attributes(self) -> Mapping[str, StatisticSpec]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        Mapping[str, StatisticSpec]
            The shape of each attribute by name.
        """
        return {
            "quantiles_": StatisticSpec(StatisticShape.PER_QUANTILE_AND_FEATURE),
            "references_": StatisticSpec(StatisticShape.PER_QUANTILE),
            "n_quantiles_": StatisticSpec(
                StatisticShape.SCALAR, StatisticConstraint.COUNT
            ),
            "n_features_in_": StatisticSpec(
                StatisticShape.SCALAR, StatisticConstraint.FEATURE_COUNT
            ),
        }

    @override
    def validate_statistics(
        self, statistics: Mapping[str, NDArray[np.generic]]
    ) -> None:
        """
        Check that the statistics describe one increasing mapping per feature.

        Parameters
        ----------
        statistics : Mapping[str, NDArray[np.generic]]
            The statistics by attribute name, whose shapes are already checked.

        Raises
        ------
        ValueError
            If the statistics disagree on the number of quantiles, there are more
            quantiles than configured, or the quantiles or references are out of
            order or range.
        """
        quantiles: FloatArray = cast(FloatArray, statistics["quantiles_"])
        references: FloatArray = cast(FloatArray, statistics["references_"])
        reference_shape: tuple[int, ...] = references.shape
        n_quantiles: int = reference_shape[0]
        if quantiles.shape[0] != n_quantiles or not np.array_equal(
            statistics["n_quantiles_"], n_quantiles
        ):
            raise ValueError(
                f"Statistics disagree on the number of quantiles: {quantiles.shape[0]}, {n_quantiles}, {statistics['n_quantiles_']}"
            )
        if n_quantiles > self.n_quantiles:
            raise ValueError(
                f"Statistics have {n_quantiles} quantiles, above the configured {self.n_quantiles}"
            )
        if np.less(np.diff(quantiles, axis=0), 0).any():
            raise ValueError("quantiles_ must not decrease")
        if np.less_equal(np.diff(references), 0).any():
            raise ValueError("references_ must increase")
        if np.less(references, 0).any() or np.greater(references, 1).any():
            raise ValueError("references_ must be within [0, 1]")

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
