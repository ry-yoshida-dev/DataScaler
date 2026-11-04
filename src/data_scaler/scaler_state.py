from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

from .scaling_strategy import ScalingStrategy
from .types import JsonValue

METADATA_KEY: Final[str] = "__metadata__"
FORMAT_VERSION: Final[int] = 2


@dataclass(frozen=True, eq=False)
class ScalerState:
    """
    Everything a fitted scaler needs, in a form that needs no pickle.

    Attributes
    ----------
    strategy : ScalingStrategy
        The identifier of the strategy.
    parameters : Mapping[str, JsonValue]
        The parameter values.
    n_features : int
        The number of features seen during fitting.
    arrays : Mapping[str, NDArray[np.generic]]
        The fitted statistics by attribute path.
    scikit_learn_version : str
        The version of scikit-learn that produced the statistics.
    empty_attributes : tuple[str, ...]
        The attribute paths that this configuration leaves empty after fitting.

    Raises
    ------
    ValueError
        If the number of features is not positive or a statistic is not numeric.
    """

    strategy: ScalingStrategy
    parameters: Mapping[str, JsonValue]
    n_features: int
    arrays: Mapping[str, NDArray[np.generic]]
    scikit_learn_version: str
    empty_attributes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be positive, got {self.n_features}")
        for name, array in self.arrays.items():
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")

    def save(self, path: Path) -> None:
        """
        Write the state to a NumPy archive.

        Parameters
        ----------
        path : Path
            The file to write.
        """
        metadata: str = json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "strategy": self.strategy.value,
                "parameters": dict(self.parameters),
                "n_features": self.n_features,
                "scikit_learn_version": self.scikit_learn_version,
                "empty_attributes": list(self.empty_attributes),
            }
        )
        with path.open("wb") as file:
            np.savez(
                file,
                allow_pickle=False,
                **{METADATA_KEY: np.array(metadata)},
                **self.arrays,
            )

    @staticmethod
    def load(path: Path) -> ScalerState:
        """
        Read a state written by save.

        Parameters
        ----------
        path : Path
            The file to read.

        Returns
        -------
        ScalerState
            The state.

        Raises
        ------
        ValueError
            If the file is not a scaler state of a supported format version.
        """
        loaded: NpzFile[np.generic] | NDArray[np.generic] = cast(
            NpzFile[np.generic] | NDArray[np.generic],
            np.load(path, allow_pickle=False),
        )
        if not isinstance(loaded, NpzFile):
            raise ValueError(f"{path} is not a scaler state")
        with loaded as archive:
            if METADATA_KEY not in archive.files:
                raise ValueError(f"{path} is not a scaler state")
            metadata: JsonValue = cast(
                JsonValue, json.loads(str(archive[METADATA_KEY]))
            )
            arrays: dict[str, NDArray[np.generic]] = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != METADATA_KEY
            }
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} is not a scaler state")
        version: JsonValue = metadata.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported scaler state format version: {version}")
        strategy: JsonValue = metadata.get("strategy")
        parameters: JsonValue = metadata.get("parameters")
        n_features: JsonValue = metadata.get("n_features")
        scikit_learn_version: JsonValue = metadata.get("scikit_learn_version")
        empty_attributes: JsonValue = metadata.get("empty_attributes")
        if (
            not isinstance(strategy, str)
            or not isinstance(parameters, dict)
            or not isinstance(n_features, int)
            or isinstance(n_features, bool)
            or not isinstance(scikit_learn_version, str)
            or not isinstance(empty_attributes, list)
        ):
            raise ValueError(f"{path} has invalid scaler state metadata")
        empty_names: tuple[str, ...] = tuple(
            name for name in empty_attributes if isinstance(name, str)
        )
        if len(empty_names) != len(empty_attributes):
            raise ValueError(f"{path} has invalid scaler state metadata")
        return ScalerState(
            strategy=ScalingStrategy(strategy),
            parameters=parameters,
            n_features=n_features,
            arrays=arrays,
            scikit_learn_version=scikit_learn_version,
            empty_attributes=empty_names,
        )
