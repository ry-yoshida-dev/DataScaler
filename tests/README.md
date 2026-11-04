# Tests

## Overview

Unit tests for the `data_scaler` package.

## Components

| Component | Description |
| --- | --- |
| [`test_feature_scaling.py`](test_feature_scaling.py) | Tests for each `FeatureScaling` shorthand. |
| [`test_scaling_kind.py`](test_scaling_kind.py) | Tests that `kind` follows the configuration and matches the actual transformation. |
| [`test_parameters.py`](test_parameters.py) | Tests for the effects and validation of each parameter class. |
| [`test_scaler.py`](test_scaler.py) | Tests for input checks, missing values, inverses, batches, persistence, and pipelines across all strategies. |

## Examples

```bash
uv run pytest
```
