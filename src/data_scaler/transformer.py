"""Structural type for fitted-then-applied transformers."""

from __future__ import annotations

from typing import Protocol

from .types import FloatArray, NumericArray


class TransformerLike(Protocol):
    """
    Structural type for scikit-learn style transformers.
    """

    def set_output(self, *, transform: str) -> object:
        """
        Choose the container of the output, regardless of the global configuration.

        Parameters
        ----------
        transform : str
            The name of the container.

        Returns
        -------
        object
            The configured transformer.
        """
        ...

    def fit(self, X: NumericArray) -> object:
        """
        Learn the statistics of the samples.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        object
            The fitted transformer.
        """
        ...

    def transform(self, X: NumericArray) -> FloatArray:
        """
        Apply the transformation.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        FloatArray
            The transformed features.
        """
        ...
