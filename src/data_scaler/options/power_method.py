from __future__ import annotations

from enum import Enum


class PowerMethod(Enum):
    """
    Family of power transformations.

    Attributes
    ----------
    YEO_JOHNSON
        Works with any values.
    BOX_COX
        Requires strictly positive values.
    """

    YEO_JOHNSON = "yeo-johnson"
    BOX_COX = "box-cox"
