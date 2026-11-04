from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import override

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from ..invertible_scaler import InvertibleScaler
from ..parameter import ScalingParameters
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy


class PowerMethod(Enum):
    """
    Family of power transformations.

    Attributes
    ----------
    YEO_JOHNSON
        Works with any values.
    BOX_COX
        Requires strictly positive values.
    """

    YEO_JOHNSON = "yeo-johnson"
    BOX_COX = "box-cox"


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
    def state_attributes(self) -> tuple[str, ...]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        tuple[str, ...]
            The attribute names, prefixed by their pipeline step.
        """
        power: tuple[str, ...] = ("power.lambdas_", "power.n_features_in_")
        if not self.is_standardized:
            return power
        return (
            *power,
            "standard.mean_",
            "standard.var_",
            "standard.scale_",
            "standard.n_samples_seen_",
            "standard.n_features_in_",
        )

    @override
    def build_transformer(self) -> Pipeline[np.float64]:
        """
        Build an unfitted transformer.

        The standardization is a separate step, so that all fitted statistics are
        public and can be saved.

        Returns
        -------
        Pipeline[np.float64]
            The transformer.
        """
        return Pipeline(
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
