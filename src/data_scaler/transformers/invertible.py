"""Structural type for transformers that can be undone."""

from __future__ import annotations

from typing import Protocol

from ..types import FloatArray, NumericArray
from .transformer import TransformerLike


class InvertibleTransformerLike(TransformerLike, Protocol):
    """
    Structural type for transformers with an inverse.
    """

    def inverse_transform(self, X: NumericArray) -> FloatArray:
        """
        Undo the transformation.

        Parameters
        ----------
        X : NumericArray
            The transformed features.

        Returns
        -------
        FloatArray
            The features on their original scale.
        """
        ...
