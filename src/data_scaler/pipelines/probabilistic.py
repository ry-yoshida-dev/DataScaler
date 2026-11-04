"""Structural type for pipelines that end with a probabilistic classifier."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..types import FloatArray, NumericArray
from .scoring import ScoringPipelineLike


class ProbabilisticPipelineLike[PredictionT: np.generic](
    ScoringPipelineLike[PredictionT], Protocol
):
    """
    Structural type for a scaling pipeline whose classifier predicts probabilities.
    """

    def predict_proba(self, X: NumericArray) -> FloatArray:
        """
        Scale features and predict the probability of each class.

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
