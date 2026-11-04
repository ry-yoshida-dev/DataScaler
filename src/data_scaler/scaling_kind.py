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
