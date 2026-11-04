from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from ..options import PowerMethod
from ..parameter import ScalingParameters
from ..scalers import InvertibleScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_shape import StatisticShape
from ..transformers import InvertibleTransformerLike


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
    def state_attributes(self) -> Mapping[str, StatisticShape]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        Mapping[str, StatisticShape]
            The shape of each attribute by name, prefixed by its pipeline step.
        """
        power: dict[str, StatisticShape] = {
            "power.lambdas_": StatisticShape.PER_FEATURE,
            "power.n_features_in_": StatisticShape.SCALAR,
        }
        if not self.is_standardized:
            return power
        return {
            **power,
            "standard.mean_": StatisticShape.PER_FEATURE,
            "standard.var_": StatisticShape.PER_FEATURE,
            "standard.scale_": StatisticShape.PER_FEATURE,
            "standard.n_samples_seen_": StatisticShape.SCALAR_OR_PER_FEATURE,
            "standard.n_features_in_": StatisticShape.SCALAR,
        }

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
