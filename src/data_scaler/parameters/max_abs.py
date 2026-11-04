from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast, override

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MaxAbsScaler

from ..parameter import ScalingParameters
from ..scalers import IncrementalScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_constraint import StatisticConstraint
from ..statistic_shape import StatisticShape
from ..statistic_spec import StatisticSpec
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class MaxAbsScalingParameters(ScalingParameters):
    """
    Each feature is divided by its largest absolute value, keeping zeros.
    """

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
        return ScalingStrategy.MAX_ABS

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
        return ScalingKind.PROPORTIONAL

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
            "max_abs_": StatisticSpec(
                StatisticShape.PER_FEATURE, StatisticConstraint.NON_NEGATIVE
            ),
            "scale_": StatisticSpec(
                StatisticShape.PER_FEATURE, StatisticConstraint.POSITIVE
            ),
            "n_samples_seen_": StatisticSpec(
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
        Check that the scale follows from the largest absolute value.

        Parameters
        ----------
        statistics : Mapping[str, NDArray[np.generic]]
            The statistics by attribute name, whose shapes are already checked.

        Raises
        ------
        ValueError
            If the statistics disagree.
        """
        self._require_scale(
            "scale_",
            cast(FloatArray, statistics["scale_"]),
            cast(FloatArray, statistics["max_abs_"]),
            fallback=1.0,
        )

    @override
    def build_transformer(self) -> MaxAbsScaler:
        """
        Build an unfitted transformer.

        Returns
        -------
        MaxAbsScaler
            The transformer.
        """
        return MaxAbsScaler()

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
