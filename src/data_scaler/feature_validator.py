from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import NumericArray


class FeatureValidator:
    """
    Runtime checks of feature matrices.
    """

    @staticmethod
    def require_matrix(X: NumericArray, *, is_nan_allowed: bool) -> None:
        """
        Check that features form a non-empty numeric matrix without infinity.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        is_nan_allowed : bool
            Whether NaN marks missing values rather than an error.

        Raises
        ------
        TypeError
            If the features are not numeric.
        ValueError
            If the features are not 2-D, are empty, contain infinity, or contain
            NaN when it is not allowed.
        """
        is_numeric: bool = np.issubdtype(X.dtype, np.integer) or np.issubdtype(
            X.dtype, np.floating
        )
        if not is_numeric:
            raise TypeError(f"X must be numeric, got dtype {X.dtype}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim} dimensions")
        if X.size == 0:
            raise ValueError(f"X must not be empty, got shape {X.shape}")
        if np.isinf(X).any():
            raise ValueError("X must not contain infinity")
        if not is_nan_allowed and np.isnan(X).any():
            raise ValueError("X must not contain NaN unless is_nan_allowed is set")

    @staticmethod
    def require_observed_features(X: NumericArray) -> None:
        """
        Check that every feature has a value to learn from.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).

        Raises
        ------
        ValueError
            If a feature contains only NaN.
        """
        unobserved: NDArray[np.intp] = np.flatnonzero(np.isnan(X).all(axis=0))
        if unobserved.size > 0:
            raise ValueError(
                f"Features must have an observed value, got only NaN in {unobserved.tolist()}"
            )
