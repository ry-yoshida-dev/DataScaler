from __future__ import annotations

from enum import Enum


class ScalingKind(Enum):
    """
    What a scaling strategy does to the values.

    Attributes
    ----------
    IDENTITY
        Values stay as they are.
    PROPORTIONAL
        Each feature is multiplied by a constant, keeping zeros and ratios.
    AFFINE
        Each feature is shifted and multiplied by a constant, keeping its shape.
    MONOTONIC
        Each feature is reshaped nonlinearly, keeping the order of its values,
        though distinct values may become equal.
    SAMPLE_WISE
        Each sample is rescaled on its own, which mixes the features of a sample
        and does not keep the order of a feature.
    """

    IDENTITY = "identity"
    PROPORTIONAL = "proportional"
    AFFINE = "affine"
    MONOTONIC = "monotonic"
    SAMPLE_WISE = "sample_wise"

    @classmethod
    def of_linear(cls, *, is_centered: bool, is_scaled: bool) -> ScalingKind:
        """
        Get the kind of a per-feature strategy that can center and divide.

        Parameters
        ----------
        is_centered : bool
            Whether a location is subtracted from each feature.
        is_scaled : bool
            Whether each feature is divided by a spread.

        Returns
        -------
        ScalingKind
            The kind.
        """
        if is_centered:
            return cls.AFFINE
        return cls.PROPORTIONAL if is_scaled else cls.IDENTITY
