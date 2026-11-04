"""Structural type for estimators that score themselves."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..predictor import PredictorLike
from ..types import NumericArray


class ScoringPredictorLike[PredictionT: np.generic](
    PredictorLike[PredictionT], Protocol
):
    """
    Structural type for estimators with a default quality measure.
    """

    def score(self, X: NumericArray, y: NumericArray, /) -> float:
        """
        Measure the quality of the predictions.

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
