# data_scaler

## Overview

Scaling and normalization strategies. Parameter classes configure a strategy and build a `Scaler`, whose capabilities grow with what the strategy can do: every scaler transforms and saves, `InvertibleScaler` adds an inverse, and `IncrementalScaler` adds learning in batches.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases for numeric data and serialized parameters. |
| [feature_validator.py](./feature_validator.py) | `FeatureValidator`, runtime checks of feature matrices. |
| [scaling_kind.py](./scaling_kind.py) | `ScalingKind`, what a strategy does to the values. |
| [scaling_strategy.py](./scaling_strategy.py) | `ScalingStrategy`, the stable identifier of a strategy in saved scalers. |
| [statistic_shape.py](./statistic_shape.py) | `StatisticShape`, the array shape of a fitted statistic. |
| [parameter.py](./parameter.py) | `ScalingParameters`, the abstract base class of configurations. |
| [parameters/](./parameters/README.md) | Per-strategy parameter classes. |
| [options/](./options/README.md) | Enums for the fields of parameter classes. |
| [parameter_codec.py](./parameter_codec.py) | `ParameterCodec`, which converts parameters to and from JSON values. |
| [scaling_configuration.py](./scaling_configuration.py) | `ScalingConfigurationLike`, what a scaler needs from its configuration. |
| [scaler.py](./scaler.py) | `Scaler`, with strict `fit`, `transform`, and `save`. |
| [scalers/](./scalers/README.md) | Scalers with more capabilities. |
| [scaler_state.py](./scaler_state.py) | `ScalerState`, the fitted statistics in a pickle-free archive. |
| [scaler_loader.py](./scaler_loader.py) | `ScalerLoader`, which restores fitted scalers. |
| [feature_scaling.py](./feature_scaling.py) | `FeatureScaling` enum, a shorthand for strategies with default parameters. |
| [transformer.py](./transformer.py) | `TransformerLike`, the structural type of wrapped transformers. |
| [transformers/](./transformers/README.md) | Structural types of transformers with more capabilities. |
| [pipeline.py](./pipeline.py) | `PipelineLike`, the structural type of a scaling pipeline. |
| [predictor.py](./predictor.py) | `PredictorLike`, the estimators that a scaling pipeline can end with. |
