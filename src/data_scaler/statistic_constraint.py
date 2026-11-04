from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray


class StatisticConstraint(Enum):
    """
    What the values of a fitted statistic must satisfy, beyond being finite.

    The values describe the requirement in error messages.

    Attributes
    ----------
    FINITE
        Nothing else is required.
    NON_NEGATIVE
        No value is below zero.
    POSITIVE
        Every value is above zero.
    COUNT
        Every value is a positive integer.
    FEATURE_COUNT
        The value is the number of features seen during fitting.
    """

    FINITE = "finite"
    NON_NEGATIVE = "non-negative"
    POSITIVE = "positive"
    COUNT = "a positive integer"
    FEATURE_COUNT = "the number of features"

    def is_satisfied_by(self, array: NDArray[np.generic], n_features: int) -> bool:
        """
        Check the values of a finite real array against this constraint.

        Parameters
        ----------
        array : NDArray[np.generic]
            The values.
        n_features : int
            The number of features seen during fitting.

        Returns
        -------
        bool
            True when all values satisfy the constraint.
        """
        match self:
            case StatisticConstraint.FINITE:
                return True
            case StatisticConstraint.NON_NEGATIVE:
                return bool(np.greater_equal(array, 0).all())
            case StatisticConstraint.POSITIVE:
                return bool(np.greater(array, 0).all())
            case StatisticConstraint.COUNT:
                # scikit-learn keeps some counts as floats.
                return bool(
                    np.greater(array, 0).all()
                    and np.equal(array, np.floor(array)).all()
                )
            case StatisticConstraint.FEATURE_COUNT:
                return np.issubdtype(array.dtype, np.integer) and np.array_equal(
                    array, n_features
                )
