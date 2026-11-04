from __future__ import annotations

from pathlib import Path
from typing import Final

from .parameter import ScalingParameters
from .parameter_codec import ParameterCodec
from .parameters import (
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    NoScalingParameters,
    PowerScalingParameters,
    QuantileScalingParameters,
    RobustScalingParameters,
    StandardScalingParameters,
    UnitNormScalingParameters,
)
from .scaler import Scaler
from .scaler_state import ScalerState
from .scaling_strategy import ScalingStrategy

PARAMETER_TYPES: Final[dict[ScalingStrategy, type[ScalingParameters]]] = {
    ScalingStrategy.NONE: NoScalingParameters,
    ScalingStrategy.STANDARD: StandardScalingParameters,
    ScalingStrategy.MIN_MAX: MinMaxScalingParameters,
    ScalingStrategy.MAX_ABS: MaxAbsScalingParameters,
    ScalingStrategy.ROBUST: RobustScalingParameters,
    ScalingStrategy.UNIT_NORM: UnitNormScalingParameters,
    ScalingStrategy.QUANTILE: QuantileScalingParameters,
    ScalingStrategy.POWER: PowerScalingParameters,
}


class ScalerLoader:
    """
    Restores fitted scalers from their saved state.
    """

    @staticmethod
    def from_state(state: ScalerState) -> Scaler:
        """
        Rebuild a fitted scaler.

        Parameters
        ----------
        state : ScalerState
            The state.

        Returns
        -------
        Scaler
            The fitted scaler of the original concrete type.

        Raises
        ------
        ValueError
            If the parameters or attributes do not fit the strategy.
        """
        parameters: ScalingParameters = ParameterCodec.decode(
            PARAMETER_TYPES[state.strategy], state.parameters
        )
        scaler: Scaler = parameters.build_scaler()
        scaler.restore_state(state)
        return scaler

    @staticmethod
    def load(path: Path) -> Scaler:
        """
        Read a fitted scaler written by ``Scaler.save``.

        Pickle is not involved, so loading an untrusted file cannot run code.

        Parameters
        ----------
        path : Path
            The file to read.

        Returns
        -------
        Scaler
            The fitted scaler of the original concrete type.

        Raises
        ------
        ValueError
            If the file is not a valid scaler state.
        """
        return ScalerLoader.from_state(ScalerState.load(path))
