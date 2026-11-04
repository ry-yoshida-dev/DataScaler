# scalers

## Overview

Scalers that extend [`Scaler`](../scaler.py) with what a strategy can additionally do, so that the type checker rejects other uses.

## Components

| Component | Description |
|-----------|-------------|
| [invertible.py](./invertible.py) | `InvertibleScaler`, which adds `inverse_transform`. |
| [incremental.py](./incremental.py) | `IncrementalScaler`, which adds `partial_fit`. |
