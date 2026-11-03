"""Structural type for what a scaler needs from its configuration."""

from __future__ import annotations

from typing import Protocol

from .types import JsonValue


class ScalingConfigurationLike(Protocol):
    """
    Structural type for the configuration of a scaler.

    Attributes
    ----------
    is_nan_allowed : bool
        Whether NaN marks missing values.
    """

    @property
    def is_nan_allowed(self) -> bool:
        """
        Get whether NaN marks missing values.

        Returns
        -------
        bool
            True when NaN is allowed.
        """
        ...

    @property
    def state_attributes(self) -> tuple[str, ...]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        tuple[str, ...]
            The attribute names, prefixed by ``step.`` inside a pipeline.
        """
        ...

    def encode(self) -> dict[str, JsonValue]:
        """
        Convert the configuration to JSON-compatible values.

        Returns
        -------
        dict[str, JsonValue]
            The values by field name.
        """
        ...
