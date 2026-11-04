from __future__ import annotations

from enum import Enum


class OutputDistribution(Enum):
    """
    Distribution that a quantile transformation maps features to.

    Attributes
    ----------
    UNIFORM
        Uniform on [0, 1].
    NORMAL
        Standard normal.
    """

    UNIFORM = "uniform"
    NORMAL = "normal"
