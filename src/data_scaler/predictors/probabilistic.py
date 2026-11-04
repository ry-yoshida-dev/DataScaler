"""Structural type for estimators that predict class probabilities."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..types import FloatArray, NumericArray
from .scoring import ScoringPredictorLike


class ProbabilisticPredictorLike[PredictionT: np.generic](
    ScoringPredictorLike[PredictionT], Protocol
):
    """
    Structural type for classifiers that predict class probabilities.
    """

    def predict_proba(self, X: NumericArray, /) -> FloatArray:
        """
        Predict the probability of each class.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        FloatArray
            The probabilities with shape (n_samples, n_classes).
        """
        ...
