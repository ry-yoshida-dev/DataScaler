# methods

## Overview

Parameter classes for each scaling strategy. Every parameter class is frozen, validates its fields on construction, and builds a scaler via `build_scaler()`, a raw transformer via `build_estimator()`, or a pipeline via `build_pipeline()`.

## Components

| Component | Description |
|-----------|-------------|
| [no_scaling.py](./no_scaling.py) | `NoScalingParameters`, the identity. |
| [standard.py](./standard.py) | `StandardScalingParameters`, zero mean and unit variance. |
| [min_max.py](./min_max.py) | `MinMaxScalingParameters`, a fixed range. |
| [max_abs.py](./max_abs.py) | `MaxAbsScalingParameters`, division by the largest absolute value. |
| [robust.py](./robust.py) | `RobustScalingParameters`, median and quantile range. |
| [unit_norm.py](./unit_norm.py) | `UnitNormScalingParameters` and `Norm`, per-sample length. |
| [quantile.py](./quantile.py) | `QuantileScalingParameters` and `OutputDistribution`, the empirical distribution. |
| [power.py](./power.py) | `PowerScalingParameters` and `PowerMethod`, a learned power transformation. |
