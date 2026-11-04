from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import Normalizer

from ..options import Norm
from ..parameter import ScalingParameters
from ..scaler import Scaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_shape import StatisticShape


@dataclass(frozen=True, kw_only=True)
class UnitNormScalingParameters(ScalingParameters):
    """
    Each sample, not each feature, is divided by its length.

    Nothing is learned, and the original lengths are lost, so there is no inverse.
    All-zero samples stay unchanged.

    Attributes
    ----------
    norm : Norm
        How the length of a sample is measured.

    Raises
    ------
    ValueError
        If NaN is allowed, because a missing value leaves the length undefined.
    """

    norm: Norm = Norm.L2

    def __post_init__(self) -> None:
        if self.is_nan_allowed:
            raise ValueError("is_nan_allowed is not supported by unit norm scaling")

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
        return ScalingStrategy.UNIT_NORM

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
        return ScalingKind.SAMPLE_WISE

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
        return {"n_features_in_": StatisticShape.SCALAR}

    @override
    def build_transformer(self) -> Normalizer:
        """
        Build an unfitted transformer.

        Returns
        -------
        Normalizer
            The transformer.
        """
        return Normalizer(norm=self.norm.value)

    @override
    def build_scaler(self) -> Scaler:
        """
        Build an unfitted scaler.

        Returns
        -------
        Scaler
            The scaler.
        """
        return Scaler(self, self.build_transformer())
