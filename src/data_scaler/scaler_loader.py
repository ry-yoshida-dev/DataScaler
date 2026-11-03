from __future__ import annotations

from pathlib import Path
from typing import Final

from .methods import (
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    NoScalingParameters,
    PowerScalingParameters,
    QuantileScalingParameters,
    RobustScalingParameters,
    StandardScalingParameters,
    UnitNormScalingParameters,
)
from .parameter import ScalingParameters
from .parameter_codec import ParameterCodec
from .scaler import Scaler
from .scaler_state import ScalerState
from .transformers import TransformerLike

PARAMETER_TYPES: Final[dict[str, type[ScalingParameters]]] = {
    parameter_type.__name__: parameter_type
    for parameter_type in (
        MaxAbsScalingParameters,
        MinMaxScalingParameters,
        NoScalingParameters,
        PowerScalingParameters,
        QuantileScalingParameters,
        RobustScalingParameters,
        StandardScalingParameters,
        UnitNormScalingParameters,
    )
}


class ScalerLoader:
    """
    Restores fitted scalers from their saved state.
    """

    @staticmethod
    def from_state(state: ScalerState) -> Scaler[TransformerLike]:
        """
        Rebuild a fitted scaler.

        Parameters
        ----------
        state : ScalerState
            The state.

        Returns
        -------
        Scaler[TransformerLike]
            The fitted scaler of the original concrete type.

        Raises
        ------
        ValueError
            If the state names an unknown strategy.
        """
        if state.method not in PARAMETER_TYPES:
            raise ValueError(f"Unknown scaling method: {state.method}")
        parameters: ScalingParameters = ParameterCodec.decode(
            PARAMETER_TYPES[state.method], state.parameters
        )
        scaler: Scaler[TransformerLike] = parameters.build_scaler()
        scaler.restore_state(state)
        return scaler

    @staticmethod
    def load(path: Path) -> Scaler[TransformerLike]:
        """
        Read a fitted scaler written by ``Scaler.save``.

        Pickle is not involved, so loading an untrusted file cannot run code.

        Parameters
        ----------
        path : Path
            The file to read.

        Returns
        -------
        Scaler[TransformerLike]
            The fitted scaler of the original concrete type.

        Raises
        ------
        ValueError
            If the file is not a valid scaler state.
        """
        return ScalerLoader.from_state(ScalerState.load(path))
