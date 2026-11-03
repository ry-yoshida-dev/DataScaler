# transformers

## Overview

Structural types of the estimators that scalers and pipelines wrap, from least to most capable.

## Components

| Component | Description |
|-----------|-------------|
| [transformer.py](./transformer.py) | `TransformerLike`, with `fit` and `transform`. |
| [invertible.py](./invertible.py) | `InvertibleTransformerLike`, which adds `inverse_transform`. |
| [incremental.py](./incremental.py) | `IncrementalTransformerLike`, which adds `partial_fit`. |
| [predictor.py](./predictor.py) | `PredictorLike`, the estimators that a scaling pipeline can end with. |
