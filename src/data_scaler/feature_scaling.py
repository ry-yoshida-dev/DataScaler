from __future__ import annotations

from enum import Enum
from typing import overload

import numpy as np

from .parameter import ScalingParameters
from .parameters import (
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    NoScalingParameters,
    RobustScalingParameters,
    StandardScalingParameters,
    UnitNormScalingParameters,
)
from .pipeline import PipelineLike
from .pipelines import ProbabilisticPipelineLike, ScoringPipelineLike
from .predictor import PredictorLike
from .predictors import ProbabilisticPredictorLike, ScoringPredictorLike
from .scaler import Scaler
from .scaling_kind import ScalingKind


class FeatureScaling(Enum):
    """
    Shorthand for the common scaling strategies with their default parameters.

    Use a parameter class from ``data_scaler`` directly to configure a strategy.

    Attributes
    ----------
    NONE
        Features are used as they are.
    STANDARD
        Zero mean and unit variance per feature.
    MIN_MAX
        The [0, 1] range per feature.
    MAX_ABS
        Division by the largest absolute value per feature, keeping zeros.
    ROBUST
        Centered on the median and scaled by the interquartile range per feature.
    UNIT_NORM
        Unit Euclidean length per sample.
    """

    NONE = "none"
    STANDARD = "standard"
    MIN_MAX = "min_max"
    MAX_ABS = "max_abs"
    ROBUST = "robust"
    UNIT_NORM = "unit_norm"

    @property
    def parameters(self) -> ScalingParameters:
        """
        Get the default parameters of this strategy.

        Returns
        -------
        ScalingParameters
            The parameters.
        """
        match self:
            case FeatureScaling.NONE:
                return NoScalingParameters()
            case FeatureScaling.STANDARD:
                return StandardScalingParameters()
            case FeatureScaling.MIN_MAX:
                return MinMaxScalingParameters()
            case FeatureScaling.MAX_ABS:
                return MaxAbsScalingParameters()
            case FeatureScaling.ROBUST:
                return RobustScalingParameters()
            case FeatureScaling.UNIT_NORM:
                return UnitNormScalingParameters()

    @property
    def kind(self) -> ScalingKind:
        """
        Get what this strategy does to the values.

        Returns
        -------
        ScalingKind
            The kind.
        """
        return self.parameters.kind

    def build_scaler(self) -> Scaler:
        """
        Build an unfitted scaler.

        Returns
        -------
        Scaler
            The scaler.
        """
        return self.parameters.build_scaler()

    @overload
    def build_pipeline[PredictionT: np.generic](
        self, estimator: ProbabilisticPredictorLike[PredictionT]
    ) -> ProbabilisticPipelineLike[PredictionT]: ...

    @overload
    def build_pipeline[PredictionT: np.generic](
        self, estimator: ScoringPredictorLike[PredictionT]
    ) -> ScoringPipelineLike[PredictionT]: ...

    @overload
    def build_pipeline[PredictionT: np.generic](
        self, estimator: PredictorLike[PredictionT]
    ) -> PipelineLike[PredictionT]: ...

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
            A pipeline with a ``scaler`` step and an ``estimator`` step, which also
            offers ``score`` and ``predict_proba`` when the estimator does.
        """
        return self.parameters.build_pipeline(estimator)
