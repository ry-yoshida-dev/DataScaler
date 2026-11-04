from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from .parameter_codec import ParameterCodec
from .parameter_validator import ParameterValidator
from .pipeline import PipelineLike
from .predictor import PredictorLike
from .scaling_kind import ScalingKind
from .scaling_strategy import ScalingStrategy
from .statistic_spec import StatisticSpec
from .transformer import TransformerLike
from .types import FloatArray, JsonValue, NumericArray

if TYPE_CHECKING:
    from .scaler import Scaler


@dataclass(frozen=True, kw_only=True)
class ScalingParameters(ABC):
    """
    Base class for the configuration of a scaling strategy.

    Attributes
    ----------
    is_nan_allowed : bool
        Whether NaN marks missing values, which are ignored while learning and
        stay NaN in the output. When False, NaN raises ValueError.

    Raises
    ------
    TypeError
        If a field does not have its annotated type.
    ValueError
        If a value is out of range for this strategy.
    """

    is_nan_allowed: bool = False

    def __post_init__(self) -> None:
        ParameterValidator.require_valid_fields(self)
        self._validate_values()

    def encode(self) -> dict[str, JsonValue]:
        """
        Convert the configuration to JSON-compatible values.

        Returns
        -------
        dict[str, JsonValue]
            The values by field name.
        """
        return ParameterCodec.encode(self)

    @property
    @abstractmethod
    def strategy(self) -> ScalingStrategy:
        """
        Get the identifier of this strategy.

        Returns
        -------
        ScalingStrategy
            The strategy.
        """

    @property
    @abstractmethod
    def kind(self) -> ScalingKind:
        """
        Get what this strategy does to the values.

        Returns
        -------
        ScalingKind
            The kind.
        """

    @property
    @abstractmethod
    def state_attributes(self) -> Mapping[str, StatisticSpec]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        Mapping[str, StatisticSpec]
            The shape of each attribute by name, prefixed by ``step.`` inside a
            pipeline.
        """

    @property
    def empty_state_attributes(self) -> frozenset[str]:
        """
        Get the state attributes that this configuration leaves empty after fitting.

        Returns
        -------
        frozenset[str]
            The attribute names, none by default.
        """
        return frozenset()

    def validate_statistics(
        self, statistics: Mapping[str, NDArray[np.generic]]
    ) -> None:
        """
        Check what this strategy requires of its fitted statistics together.

        Nothing is required by default.

        Parameters
        ----------
        statistics : Mapping[str, NDArray[np.generic]]
            The statistics by attribute name, whose shapes are already checked.

        Raises
        ------
        ValueError
            If the statistics cannot come from fitting this strategy.
        """
        _ = statistics

    def validate_output(self, X: NumericArray, output: FloatArray) -> None:
        """
        Check what this strategy requires of its scaled features.

        Nothing is required by default.

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
        _ = X, output

    @abstractmethod
    def build_transformer(self) -> TransformerLike:
        """
        Build an unfitted scikit-learn transformer, without input checks.

        Returns
        -------
        TransformerLike
            The transformer, usable as a pipeline step.
        """

    @abstractmethod
    def build_scaler(self) -> Scaler:
        """
        Build an unfitted scaler.

        Returns
        -------
        Scaler
            The scaler.
        """

    def build_pipeline[PredictionT: np.generic](
        self, estimator: PredictorLike[PredictionT]
    ) -> PipelineLike[PredictionT]:
        """
        Build a pipeline that applies this scaling before an estimator.

        Parameters
        ----------
        estimator : PredictorLike[PredictionT]
            The scikit-learn estimator to run on the scaled features.

        Returns
        -------
        PipelineLike[PredictionT]
            A pipeline with a ``scaler`` step and an ``estimator`` step.
        """
        return Pipeline(
            [("scaler", self.build_transformer()), ("estimator", estimator)]
        )

    def _validate_values(self) -> None:
        """Check field values, whose types are already checked."""
        return None

    @staticmethod
    def _is_close(actual: FloatArray, expected: FloatArray) -> NDArray[np.bool_]:
        # Half the digits of the type leaves room for rounding in other formulas.
        precision: float = float(
            cast(float, np.finfo(np.result_type(actual.dtype, np.float16)).eps)
        )
        tolerance: float = math.sqrt(precision)
        return np.isclose(actual, expected, rtol=tolerance, atol=0.0)

    @staticmethod
    def _require_close(name: str, actual: FloatArray, expected: FloatArray) -> None:
        if not ScalingParameters._is_close(actual, expected).all():
            raise ValueError(f"Statistic {name} disagrees with the other statistics")

    @staticmethod
    def _require_scale(
        name: str, scale: FloatArray, expected: FloatArray, *, fallback: float
    ) -> None:
        # scikit-learn replaces a near-zero divisor with one, which gives the fallback.
        is_expected: NDArray[np.bool_] = ScalingParameters._is_close(scale, expected)
        is_fallback: NDArray[np.bool_] = ScalingParameters._is_close(
            scale, np.full_like(scale, fallback)
        )
        if not (is_expected | is_fallback).all():
            raise ValueError(f"Statistic {name} disagrees with the other statistics")
