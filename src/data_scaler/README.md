# data_scaler

## Overview

Scaling and normalization strategies. Parameter classes configure a strategy and build a `Scaler`, whose capabilities grow with what the strategy can do: every scaler transforms and saves, `InvertibleScaler` adds an inverse, and `IncrementalScaler` adds learning in batches.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases for numeric data and serialized parameters. |
| [feature_validator.py](./feature_validator.py) | `FeatureValidator`, runtime checks of feature matrices. |
| [parameter.py](./parameter.py) | `ScalingParameters`, the abstract base class of configurations. |
| [parameter_codec.py](./parameter_codec.py) | `ParameterCodec`, which converts parameters to and from JSON values. |
| [scaling_configuration.py](./scaling_configuration.py) | `ScalingConfigurationLike`, what a scaler needs from its configuration. |
| [scaler.py](./scaler.py) | `Scaler`, with strict `fit`, `transform`, and `save`. |
| [invertible_scaler.py](./invertible_scaler.py) | `InvertibleScaler`, which adds `inverse_transform`. |
| [incremental_scaler.py](./incremental_scaler.py) | `IncrementalScaler`, which adds `partial_fit`. |
| [scaler_state.py](./scaler_state.py) | `ScalerState`, the fitted statistics in a pickle-free archive. |
| [scaler_loader.py](./scaler_loader.py) | `ScalerLoader`, which restores fitted scalers. |
| [feature_scaling.py](./feature_scaling.py) | `FeatureScaling` enum, a shorthand for strategies with default parameters. |
| [transformers/](./transformers/README.md) | Structural types of wrapped estimators. |
| [methods/](./methods/README.md) | Per-strategy parameter classes. |
