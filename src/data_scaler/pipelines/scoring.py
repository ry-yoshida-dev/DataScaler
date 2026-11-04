"""Structural type for pipelines that end with a scoring estimator."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..pipeline import PipelineLike
from ..types import NumericArray


class ScoringPipelineLike[PredictionT: np.generic](PipelineLike[PredictionT], Protocol):
    """
    Structural type for a scaling pipeline whose estimator has a quality measure.
    """

    def score(self, X: NumericArray, y: NumericArray) -> float:
        """
        Scale features and measure the quality of the predictions.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : NumericArray
            The targets.

        Returns
        -------
        float
            The score, where higher is better.
        """
        ...
