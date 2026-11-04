from __future__ import annotations

from pathlib import Path
from typing import Final, overload

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

    @overload
    @staticmethod
    def from_state(state: ScalerState) -> Scaler: ...

    @overload
    @staticmethod
    def from_state[ScalerT: Scaler](
        state: ScalerState, scaler_type: type[ScalerT]
    ) -> ScalerT: ...

    @staticmethod
    def from_state(state: ScalerState, scaler_type: type[Scaler] = Scaler) -> Scaler:
        """
        Rebuild a fitted scaler.

        Parameters
        ----------
        state : ScalerState
            The state.
        scaler_type : type[Scaler]
            The capabilities the caller relies on, which the type checker then sees.

        Returns
        -------
        Scaler
            The fitted scaler of the original concrete type.

        Raises
        ------
        TypeError
            If the strategy of the state does not offer ``scaler_type``.
        ValueError
            If the parameters or attributes do not fit the strategy.
        """
        parameters: ScalingParameters = ParameterCodec.decode(
            PARAMETER_TYPES[state.strategy], state.parameters
        )
        scaler: Scaler = parameters.build_scaler()
        if not isinstance(scaler, scaler_type):
            raise TypeError(
                f"Strategy {state.strategy.name} builds {type(scaler).__name__}, which does not offer {scaler_type.__name__}"
            )
        scaler.restore_state(state)
        return scaler

    @overload
    @staticmethod
    def load(path: Path) -> Scaler: ...

    @overload
    @staticmethod
    def load[ScalerT: Scaler](path: Path, scaler_type: type[ScalerT]) -> ScalerT: ...

    @staticmethod
    def load(path: Path, scaler_type: type[Scaler] = Scaler) -> Scaler:
        """
        Read a fitted scaler written by ``Scaler.save``.

        Pickle is not involved, so loading an untrusted file cannot run code.

        Parameters
        ----------
        path : Path
            The file to read.
        scaler_type : type[Scaler]
            The capabilities the caller relies on, which the type checker then sees.

        Returns
        -------
        Scaler
            The fitted scaler of the original concrete type.

        Raises
        ------
        TypeError
            If the saved strategy does not offer ``scaler_type``.
        ValueError
            If the file is not a valid scaler state.
        """
        return ScalerLoader.from_state(ScalerState.load(path), scaler_type)
