from __future__ import annotations

from enum import Enum


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
