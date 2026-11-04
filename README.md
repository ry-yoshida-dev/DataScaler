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

For module-level design notes, see [src/data_scaler/README.md](src/data_scaler/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Quick example

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

## Notes

- `Scaler` accepts NumPy arrays only and raises `TypeError` for anything else, such as lists or DataFrames. Convert them with `np.asarray`.
- Inputs are never modified, and outputs never share memory with inputs.
- Parameter fields are checked at construction: `TypeError` for another type than annotated, `ValueError` for a value out of range.
- `transform` and `inverse_transform` raise `ValueError` when the result is not finite where the input is, such as a value outside the range that a power transformation can undo. Unit norm scaling also raises it for a sample whose length overflows the floating point type, while samples of nearly zero length stay unchanged, as scikit-learn takes them for all-zero samples.
- `Scaler` rejects NaN unless `is_nan_allowed` is set; then NaN marks missing values, which are ignored while learning and stay NaN in the output. Infinity is always rejected. `fit` and `partial_fit` reject a feature with only NaN in the given samples, because scikit-learn would lose the statistics of that feature for good; a rejected input leaves what the scaler has learned in place. They also reject statistics that come out impossible, such as by overflowing, by the same checks as loading; the scaler is then unfitted, because what it had learned is lost. Whatever a scaler learned can therefore be saved and loaded.
- Pipelines and `build_transformer` use the scikit-learn transformers directly, without the checks of `Scaler`.
- `partial_fit` and `inverse_transform` exist only on scalers that support them, so the type checker rejects other uses.
- `QuantileScalingParameters.n_quantiles` must not exceed the number of training samples, or scikit-learn warns and reduces it.
- `kind` on `FeatureScaling` and on every parameter class tells what the configured strategy does to the values, as a `ScalingKind`, for callers whose results depend on it.
- Loading rejects damaged files and statistics that a fitted scaler cannot have, such as wrong shapes, non-finite values, non-positive scales or sample counts, negative variances, statistics that disagree with each other, statistics missing for the configuration, or unordered quantiles, with `ValueError`. Passing a scaler type to `ScalerLoader.load` lets the type checker see its capabilities, and raises `TypeError` when the saved strategy does not offer them.
- `build_pipeline` is typed as `PipelineLike`, which keeps the prediction type because scikit-learn ships no type information. At runtime it returns a `sklearn.pipeline.Pipeline`.
- Saved scalers depend on the fitted attributes of scikit-learn, whose meaning could change between versions, so loading warns when the version differs from the one that saved them.
- Parameter fields are keyword-only.
- `save` writes a temporary file next to the target and replaces it at the end, so a failed write keeps the previous file.
- Inverses restore values up to what the strategy keeps: clipping in `MinMaxScalingParameters` and the quantile mapping are lossy outside the training range.
- `FeatureScaling.build_scaler` returns the base `Scaler`. Use a parameter class when the type checker should see `inverse_transform` or `partial_fit`.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on every push to `main` and every pull request, along with the tests on the minimum supported dependency versions.

## License

Released under the [MIT License](LICENSE).
