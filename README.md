# DataScaler

## Overview

DataScaler (`data_scaler`) puts features on a comparable scale. It builds on scikit-learn and adds what production use needs: configurations validated at construction, strict input checks, capabilities visible to the type checker, and saved scalers that need no pickle.

| Strategy | Parameters | Effect | Inverse | Batches |
| --- | --- | --- | --- | --- |
| None | `NoScalingParameters` | Features stay as they are. | Yes | |
| Standard | `StandardScalingParameters` | Zero mean and unit variance per feature. | Yes | Yes |
| Min-max | `MinMaxScalingParameters` | A fixed range per feature. | Yes | Yes |
| Max-abs | `MaxAbsScalingParameters` | Division by the largest absolute value per feature, keeping zeros. | Yes | Yes |
| Robust | `RobustScalingParameters` | Median and quantile range per feature, which resist outliers. | Yes | |
| Quantile | `QuantileScalingParameters` | The empirical distribution per feature, mapped to uniform or normal. | Yes | |
| Power | `PowerScalingParameters` | A learned transformation per feature towards Gaussian. | Yes | |
| Unit norm | `UnitNormScalingParameters` | Unit length per sample, as for cosine similarity. | | |

## Installation

```bash
pip install .
```

## Usage

Fit on training samples only, then apply to anything else:

```python
from data_scaler import RobustScalingParameters

scaler = RobustScalingParameters(quantile_range=(10.0, 90.0)).build_scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_restored = scaler.inverse_transform(X_test_scaled)
```

`FeatureScaling` is a shorthand for the common strategies with default parameters:

```python
from data_scaler import FeatureScaling

scaler = FeatureScaling.STANDARD.build_scaler()
```

Learn from data too large for memory:

```python
from data_scaler import StandardScalingParameters

scaler = StandardScalingParameters().build_scaler()
for batch in batches:
    scaler.partial_fit(batch)
```

Save a fitted scaler and load it elsewhere, without pickle:

```python
from pathlib import Path

from data_scaler import IncrementalScaler, ScalerLoader

scaler.save(Path("scaler.npz"))
restored = ScalerLoader.load(Path("scaler.npz"), IncrementalScaler)
restored.partial_fit(batch)
```

Prevent leakage during cross-validation by making scaling part of the estimator:

```python
pipeline = RobustScalingParameters().build_pipeline(estimator)
```
