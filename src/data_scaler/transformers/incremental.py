"""Structural type for transformers that learn in batches."""

from __future__ import annotations

from typing import Protocol

from ..types import NumericArray
from .invertible import InvertibleTransformerLike


class IncrementalTransformerLike(InvertibleTransformerLike, Protocol):
    """
    Structural type for invertible transformers that learn batch by batch.
    """

    def partial_fit(self, X: NumericArray) -> object:
        """
        Update the statistics with a batch of samples.

        Parameters
        ----------
        X : NumericArray
            The feature matrix of the batch.

        Returns
        -------
        object
            The updated transformer.
        """
        ...
