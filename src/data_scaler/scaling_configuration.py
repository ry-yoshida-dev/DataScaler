"""Structural type for what a scaler needs from its configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .scaling_kind import ScalingKind
from .scaling_strategy import ScalingStrategy
from .statistic_spec import StatisticSpec
from .types import FloatArray, JsonValue, NumericArray


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
    def state_attributes(self) -> Mapping[str, StatisticSpec]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        Mapping[str, StatisticSpec]
            The shape of each attribute by name, prefixed by ``step.`` inside a
            pipeline.
        """
        ...

    @property
    def empty_state_attributes(self) -> frozenset[str]:
        """
        Get the state attributes that this configuration leaves empty after fitting.

        Returns
        -------
        frozenset[str]
            The attribute names.
        """
        ...

    def validate_statistics(
        self, statistics: Mapping[str, NDArray[np.generic]]
    ) -> None:
        """
        Check what this strategy requires of its fitted statistics together.

        Parameters
        ----------
        statistics : Mapping[str, NDArray[np.generic]]
            The statistics by attribute name, whose shapes are already checked.

        Raises
        ------
        ValueError
            If the statistics cannot come from fitting this strategy.
        """
        ...

    def validate_output(self, X: NumericArray, output: FloatArray) -> None:
        """
        Check what this strategy requires of its scaled features.

        Parameters
        ----------
        X : NumericArray
            The features before scaling.
        output : FloatArray
            The scaled features.

        Raises
        ------
        ValueError
            If the output is wrong without being non-finite.
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
