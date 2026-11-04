# predictors

## Overview

Structural types of estimators that extend [`PredictorLike`](../predictor.py), from less to more capable. `build_pipeline` selects the matching pipeline type from them.

## Components

| Component | Description |
|-----------|-------------|
| [scoring.py](./scoring.py) | `ScoringPredictorLike`, which adds `score`. |
| [probabilistic.py](./probabilistic.py) | `ProbabilisticPredictorLike`, which adds `predict_proba`. |
