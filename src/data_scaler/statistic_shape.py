from __future__ import annotations

from enum import Enum


class StatisticShape(Enum):
    """
    Array shape of a fitted statistic, relative to what the scaler has seen.

    Attributes
    ----------
    SCALAR
        A single value.
    PER_FEATURE
        One value per feature.
    SCALAR_OR_PER_FEATURE
        A single value, or one per feature when they differ.
    PER_QUANTILE
        One value per quantile.
    PER_QUANTILE_AND_FEATURE
        One row per quantile and one column per feature.
    """

    SCALAR = "scalar"
    PER_FEATURE = "per_feature"
    SCALAR_OR_PER_FEATURE = "scalar_or_per_feature"
    PER_QUANTILE = "per_quantile"
    PER_QUANTILE_AND_FEATURE = "per_quantile_and_feature"

    def matches(self, shape: tuple[int, ...], n_features: int) -> bool:
        """
        Check an array shape against this description.

        Parameters
        ----------
        shape : tuple[int, ...]
            The array shape.
        n_features : int
            The number of features seen during fitting.

        Returns
        -------
        bool
            True when the array shape fits.
        """
        match self:
            case StatisticShape.SCALAR:
                return shape == ()
            case StatisticShape.PER_FEATURE:
                return shape == (n_features,)
            case StatisticShape.SCALAR_OR_PER_FEATURE:
                return shape in ((), (n_features,))
            case StatisticShape.PER_QUANTILE:
                return len(shape) == 1
            case StatisticShape.PER_QUANTILE_AND_FEATURE:
                return len(shape) == 2 and shape[1] == n_features
