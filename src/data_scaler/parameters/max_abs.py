from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import MaxAbsScaler

from ..parameter import ScalingParameters
from ..scalers import IncrementalScaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy


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
    def state_attributes(self) -> tuple[str, ...]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        tuple[str, ...]
            The attribute names.
        """
        return ("max_abs_", "scale_", "n_samples_seen_", "n_features_in_")

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
