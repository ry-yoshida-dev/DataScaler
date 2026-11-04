# sklearn

## Overview

Type stubs for the parts of scikit-learn that `data_scaler` and its tests use.

## Components

| Component | Description |
|-----------|-------------|
| [base.pyi](./base.pyi) | Estimator cloning. |
| [ensemble.pyi](./ensemble.pyi) | An estimator without `score`, used in pipeline tests. |
| [exceptions/](./exceptions/README.md) | Errors raised for unfitted estimators. |
| [linear_model.pyi](./linear_model.pyi) | The estimators used in pipeline tests. |
| [pipeline.pyi](./pipeline.pyi) | `Pipeline`, generic over its prediction type. |
| [preprocessing.pyi](./preprocessing.pyi) | The wrapped transformers. |
