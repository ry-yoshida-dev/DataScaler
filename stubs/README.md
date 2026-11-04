# stubs

## Overview

Type stubs for dependencies without usable type information. Only the type checkers of this repository read them; they are not part of the distributed package, because a stub package would replace the types of scikit-learn for every user. Public signatures therefore use the structural types of `data_scaler` where callers need precise types.

## Components

| Component | Description |
|-----------|-------------|
| [sklearn/](./sklearn/README.md) | The parts of scikit-learn that `data_scaler` uses. |
