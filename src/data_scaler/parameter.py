from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.pipeline import Pipeline

from .parameter_codec import ParameterCodec
from .scaling_kind import ScalingKind
from .scaling_strategy import ScalingStrategy
from .transformers import PredictorLike, TransformerLike
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
    def state_attributes(self) -> tuple[str, ...]:
        """
        Get the fitted attributes that define a fitted scaler.

        Returns
        -------
        tuple[str, ...]
            The attribute names, prefixed by ``step.`` inside a pipeline.
        """

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
    ) -> Pipeline[PredictionT]:
        """
        Build a pipeline that applies this scaling before an estimator.

        Parameters
        ----------
        estimator : PredictorLike[PredictionT]
            The scikit-learn estimator to run on the scaled features.

        Returns
        -------
        Pipeline[PredictionT]
            A pipeline with a ``scaler`` step and an ``estimator`` step.
        """
        return Pipeline(
            [("scaler", self.build_transformer()), ("estimator", estimator)]
        )
