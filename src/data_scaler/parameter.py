from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from .parameter_codec import ParameterCodec
from .pipeline import PipelineLike
from .predictor import PredictorLike
from .scaling_kind import ScalingKind
from .scaling_strategy import ScalingStrategy
from .statistic_shape import StatisticShape
from .transformer import TransformerLike
from .types import JsonValue

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
    """

    is_nan_allowed: bool = False

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
    def state_attributes(self) -> Mapping[str, StatisticShape]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        Mapping[str, StatisticShape]
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
