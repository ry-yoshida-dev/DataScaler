# pipelines

## Overview

Structural types of scaling pipelines that extend [`PipelineLike`](../pipeline.py), from less to more capable, so that the type checker sees what the final estimator offers.

## Components

| Component | Description |
|-----------|-------------|
| [scoring.py](./scoring.py) | `ScoringPipelineLike`, which adds `score`. |
| [probabilistic.py](./probabilistic.py) | `ProbabilisticPipelineLike`, which adds `predict_proba`. |
