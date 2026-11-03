from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

from .types import JsonValue

METADATA_KEY: Final[str] = "__metadata__"
FORMAT_VERSION: Final[int] = 1


@dataclass(frozen=True, eq=False)
class ScalerState:
    """
    Everything a fitted scaler needs, in a form that needs no pickle.

    Attributes
    ----------
    method : str
        The name of the parameter class.
    parameters : Mapping[str, JsonValue]
        The parameter values.
    n_features : int
        The number of features seen during fitting.
    arrays : Mapping[str, NDArray[np.generic]]
        The fitted statistics by attribute path.
    """

    method: str
    parameters: Mapping[str, JsonValue]
    n_features: int
    arrays: Mapping[str, NDArray[np.generic]]

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
                "method": self.method,
                "parameters": dict(self.parameters),
                "n_features": self.n_features,
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
        loaded: NpzFile[np.generic] = cast(
            NpzFile[np.generic], np.load(path, allow_pickle=False)
        )
        with loaded as archive:
            if METADATA_KEY not in archive.files:
                raise ValueError(f"{path} is not a scaler state")
            metadata: dict[str, JsonValue] = cast(
                dict[str, JsonValue], json.loads(str(archive[METADATA_KEY]))
            )
            arrays: dict[str, NDArray[np.generic]] = {
                name: archive[name] for name in archive.files if name != METADATA_KEY
            }
        version: JsonValue = metadata["format_version"]
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported scaler state format version: {version}")
        return ScalerState(
            method=str(metadata["method"]),
            parameters=cast(dict[str, JsonValue], metadata["parameters"]),
            n_features=cast(int, metadata["n_features"]),
            arrays=arrays,
        )
