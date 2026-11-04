from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import Normalizer

from ..options import Norm
from ..parameter import ScalingParameters
from ..scaler import Scaler
from ..scaling_kind import ScalingKind
from ..scaling_strategy import ScalingStrategy
from ..statistic_constraint import StatisticConstraint
from ..statistic_shape import StatisticShape
from ..statistic_spec import StatisticSpec
from ..types import FloatArray, NumericArray


@dataclass(frozen=True, kw_only=True)
class UnitNormScalingParameters(ScalingParameters):
    """
    Each sample, not each feature, is divided by its length.

    Nothing is learned, and the original lengths are lost, so there is no inverse.
    Samples of nearly zero length, relative to the precision of their floating point
    type, stay unchanged, as scikit-learn takes them for all-zero samples.

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

    @override
    def _validate_values(self) -> None:
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
    def state_attributes(self) -> Mapping[str, StatisticSpec]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        Mapping[str, StatisticSpec]
            The shape of each attribute by name.
        """
        return {
            "n_features_in_": StatisticSpec(
                StatisticShape.SCALAR, StatisticConstraint.FEATURE_COUNT
            )
        }

    @override
    def validate_output(self, X: NumericArray, output: FloatArray) -> None:
        """
        Check that every sample ended up with unit length or stayed unchanged.

        scikit-learn computes lengths without guarding against overflow, which
        silently zeroes samples of large magnitude.

        Parameters
        ----------
        X : NumericArray
            The features before scaling.
        output : FloatArray
            The scaled features.

        Raises
        ------
        ValueError
            If the length of a sample overflows its floating point type.
        """
        lengths: FloatArray
        match self.norm:
            case Norm.L1:
                lengths = np.abs(output).sum(axis=1)
            case Norm.L2:
                lengths = np.linalg.norm(output, axis=1)
            case Norm.MAX:
                lengths = np.abs(output).max(axis=1)
        is_unit: NDArray[np.bool_] = self._is_close(lengths, np.ones_like(lengths))
        if not (is_unit | np.equal(output, X).all(axis=1)).all():
            raise ValueError(
                "X must have samples whose length fits the floating point type"
            )

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
