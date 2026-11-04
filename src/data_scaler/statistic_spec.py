from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .statistic_constraint import StatisticConstraint
from .statistic_shape import StatisticShape


@dataclass(frozen=True)
class StatisticSpec:
    """
    What a single fitted statistic must look like, after fitting and after loading.

    Attributes
    ----------
    shape : StatisticShape
        The array shape.
    constraint : StatisticConstraint
        What the values must satisfy.
    """

    shape: StatisticShape
    constraint: StatisticConstraint = StatisticConstraint.FINITE

    def validate(self, path: str, array: NDArray[np.generic], n_features: int) -> None:
        """
        Check a statistic against this description.

        Parameters
        ----------
        path : str
            The attribute path, for error messages.
        array : NDArray[np.generic]
            The values.
        n_features : int
            The number of features seen during fitting.

        Raises
        ------
        ValueError
            If the shape or the values do not fit.
        """
        if not self.shape.matches(array.shape, n_features):
            raise ValueError(
                f"Statistic {path} has shape {array.shape}, which does not fit {self.shape.name} with {n_features} features"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"Statistic {path} is not finite: it overflowed because the features are too large, or a subsample saw only NaN in a feature"
            )
        if not self.constraint.is_satisfied_by(array, n_features):
            raise ValueError(f"Statistic {path} must be {self.constraint.value}")
