# transformers

## Overview

Structural types of transformers that extend [`TransformerLike`](../transformer.py), from less to more capable.

## Components

| Component | Description |
|-----------|-------------|
| [invertible.py](./invertible.py) | `InvertibleTransformerLike`, which adds `inverse_transform`. |
| [incremental.py](./incremental.py) | `IncrementalTransformerLike`, which adds `partial_fit`. |
