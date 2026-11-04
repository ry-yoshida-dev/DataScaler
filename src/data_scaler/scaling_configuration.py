"""Structural type for what a scaler needs from its configuration."""

from __future__ import annotations

from typing import Protocol

from .scaling_kind import ScalingKind
from .scaling_strategy import ScalingStrategy
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
    def strategy(self) -> ScalingStrategy:
        """
        Get the identifier of this strategy.

        Returns
        -------
        ScalingStrategy
            The strategy.
        """
        ...

    @property
    def kind(self) -> ScalingKind:
        """
        Get what this strategy does to the values.

        Returns
        -------
        ScalingKind
            The kind.
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
