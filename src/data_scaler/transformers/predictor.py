"""Structural type for estimators at the end of a scaling pipeline."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..types import NumericArray


class PredictorLike[PredictionT: np.generic](Protocol):
    """
    Structural type for estimators, parameterized by their prediction type.

    The parameter carries through a pipeline, so estimators keep their distinct
    prediction types after wrapping.
    """

    def predict(self, X: NumericArray) -> NDArray[PredictionT]:
        """
        Predict targets.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        NDArray[PredictionT]
            The predictions with shape (n_samples,).
        """
        ...
