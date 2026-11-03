"""Type aliases for numeric data and serialized parameters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.floating]
type JsonValue = (
    bool | int | float | str | None | list[JsonValue] | dict[str, JsonValue]
)
