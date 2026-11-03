from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import override

from sklearn.preprocessing import Normalizer

from ..parameter import ScalingParameters
from ..scaler import Scaler
from ..transformers import TransformerLike


class Norm(Enum):
    """
    Length measure of a sample.

    Attributes
    ----------
    L1
        Sum of absolute values.
    L2
        Euclidean length, as for cosine similarity.
    MAX
        Largest absolute value.
    """

    L1 = "l1"
    L2 = "l2"
    MAX = "max"


@dataclass(frozen=True)
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
    def state_attributes(self) -> tuple[str, ...]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        tuple[str, ...]
            The attribute names.
        """
        return ("n_features_in_",)

    @override
    def build_estimator(self) -> Normalizer:
        """
        Build an unfitted transformer.

        Returns
        -------
        Normalizer
            The transformer.
        """
        return Normalizer(norm=self.norm.value)

    @override
    def build_scaler(self) -> Scaler[TransformerLike]:
        """
        Build an unfitted scaler.

        Returns
        -------
        Scaler[TransformerLike]
            The scaler.
        """
        return Scaler(self, self.build_estimator())
