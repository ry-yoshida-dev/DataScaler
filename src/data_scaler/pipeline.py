"""Structural type for pipelines that end with an estimator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray

from .types import NumericArray


class PipelineLike[PredictionT: np.generic](Protocol):
    """
    Structural type for a scaling pipeline, parameterized by its prediction type.

    scikit-learn ships no type information, so this keeps the prediction type for
    callers. At runtime the object is a ``sklearn.pipeline.Pipeline``.
    """

    @property
    def named_steps(self) -> Mapping[str, object]:
        """
        Get the steps of the pipeline.

        Returns
        -------
        Mapping[str, object]
            The steps by name.
        """
        ...

    def fit(self, X: NumericArray, y: NumericArray | None = None) -> Self:
        """
        Fit the scaling and the estimator.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : NumericArray | None
            The targets, if the estimator uses them.

        Returns
        -------
        Self
            The fitted pipeline.
        """
        ...

    def predict(self, X: NumericArray) -> NDArray[PredictionT]:
        """
        Scale features and predict targets.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        NDArray[PredictionT]
            The predictions.
        """
        ...
