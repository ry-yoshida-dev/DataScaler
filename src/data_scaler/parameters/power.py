from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, override

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from ..options import PowerMethod
from ..parameter import ScalingParameters
from ..scalers import InvertibleScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_constraint import StatisticConstraint
from ..statistic_shape import StatisticShape
from ..statistic_spec import StatisticSpec
from ..transformers import InvertibleTransformerLike
from .standard import StandardScalingParameters

STANDARD_STEP: Final[str] = "standard."


@dataclass(frozen=True, kw_only=True)
class PowerScalingParameters(ScalingParameters):
    """
    Each feature is made more Gaussian by a learned power transformation.

    Attributes
    ----------
    method : PowerMethod
        The family of transformations.
    is_standardized : bool
        Whether the output has zero mean and unit variance.
    """

    method: PowerMethod = PowerMethod.YEO_JOHNSON
    is_standardized: bool = True

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
        return ScalingStrategy.POWER

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
            The shape of each attribute by name, prefixed by its pipeline step.
        """
        power: dict[str, StatisticSpec] = {
            "power.lambdas_": StatisticSpec(StatisticShape.PER_FEATURE),
            "power.n_features_in_": StatisticSpec(
                StatisticShape.SCALAR, StatisticConstraint.FEATURE_COUNT
            ),
        }
        if not self.is_standardized:
            return power
        return {
            **power,
            **{
                f"{STANDARD_STEP}{name}": spec
                for name, spec in StandardScalingParameters().state_attributes.items()
            },
        }

    @override
    def validate_statistics(
        self, statistics: Mapping[str, NDArray[np.generic]]
    ) -> None:
        """
        Check the statistics of the standardization step.

        Parameters
        ----------
        statistics : Mapping[str, NDArray[np.generic]]
            The statistics by attribute name, whose shapes are already checked.

        Raises
        ------
        ValueError
            If the statistics disagree.
        """
        if self.is_standardized:
            StandardScalingParameters().validate_statistics(
                {
                    path.removeprefix(STANDARD_STEP): array
                    for path, array in statistics.items()
                    if path.startswith(STANDARD_STEP)
                }
            )

    @override
    def build_transformer(self) -> InvertibleTransformerLike:
        """
        Build an unfitted transformer.

        The standardization is a separate step, so that all fitted statistics are
        public and can be saved.

        Returns
        -------
        InvertibleTransformerLike
            The transformer, a pipeline of the two steps.
        """
        pipeline: Pipeline[np.float64] = Pipeline(
            [
                (
                    "power",
                    PowerTransformer(method=self.method.value, standardize=False),
                ),
                (
                    "standard",
                    StandardScaler() if self.is_standardized else FunctionTransformer(),
                ),
            ]
        )
        return pipeline

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
