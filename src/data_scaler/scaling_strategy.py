from __future__ import annotations

from enum import Enum


class ScalingStrategy(Enum):
    """
    Identifier of a scaling strategy, as written to saved scalers.

    The values are part of the file format, so they stay the same when classes
    are renamed.

    Attributes
    ----------
    NONE
        Features stay as they are.
    STANDARD
        Mean and standard deviation.
    MIN_MAX
        A fixed range.
    MAX_ABS
        Division by the largest absolute value.
    ROBUST
        Median and quantile range.
    UNIT_NORM
        Per-sample length.
    QUANTILE
        The empirical distribution.
    POWER
        A learned power transformation.
    """

    NONE = "none"
    STANDARD = "standard"
    MIN_MAX = "min_max"
    MAX_ABS = "max_abs"
    ROBUST = "robust"
    UNIT_NORM = "unit_norm"
    QUANTILE = "quantile"
    POWER = "power"
