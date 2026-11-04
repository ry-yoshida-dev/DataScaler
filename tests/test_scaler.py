from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from data_scaler import (
    FeatureScaling,
    IncrementalScaler,
    InvertibleScaler,
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    Norm,
    NoScalingParameters,
    OutputDistribution,
    PowerMethod,
    PowerScalingParameters,
    QuantileScalingParameters,
    RobustScalingParameters,
    Scaler,
    ScalerLoader,
    ScalerState,
    ScalingKind,
    ScalingParameters,
    ScalingStrategy,
    StandardScalingParameters,
    UnitNormScalingParameters,
)
from data_scaler.scaler_state import FORMAT_VERSION
from data_scaler.types import FloatArray, JsonValue, NumericArray

RANDOM: np.random.Generator = np.random.default_rng(0)
X: NDArray[np.float64] = RANDOM.lognormal(size=(300, 3))
UNSEEN: NDArray[np.float64] = RANDOM.lognormal(size=(20, 3))
ALL_METHODS: tuple[ScalingParameters, ...] = (
    NoScalingParameters(),
    StandardScalingParameters(),
    StandardScalingParameters(is_centered=False),
    StandardScalingParameters(is_scaled=False),
    MinMaxScalingParameters(feature_range=(-1.0, 1.0), is_clipped=True),
    MaxAbsScalingParameters(),
    RobustScalingParameters(quantile_range=(10.0, 90.0), is_unit_variance=True),
    RobustScalingParameters(is_centered=False),
    UnitNormScalingParameters(norm=Norm.L1),
    QuantileScalingParameters(
        n_quantiles=50, output_distribution=OutputDistribution.NORMAL
    ),
    PowerScalingParameters(),
    PowerScalingParameters(method=PowerMethod.BOX_COX, is_standardized=False),
)


class TestStrictInputs(unittest.TestCase):
    def test_requires_fit(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(parameters=parameters):
                scaler: Scaler = parameters.build_scaler()
                with self.assertRaises(NotFittedError):
                    _ = scaler.transform(X)
                with self.assertRaises(NotFittedError):
                    scaler.save(Path("unused.npz"))

    def test_rejects_invalid_matrices(self) -> None:
        for parameters in ALL_METHODS:
            scaler: Scaler = parameters.build_scaler()
            for invalid in (
                np.zeros(3),
                np.zeros((0, 3)),
                np.array([[np.inf, 1.0]]),
                np.array([[np.nan, 1.0]]),
            ):
                with self.subTest(parameters=parameters, shape=invalid.shape):
                    with self.assertRaises(ValueError):
                        scaler.fit(invalid)
            with (
                self.subTest(parameters=parameters, dtype="bool"),
                self.assertRaises(TypeError),
            ):
                scaler.fit(cast(NumericArray, np.zeros((2, 2), dtype=np.bool_)))

    def test_rejects_a_changed_number_of_features_for_every_strategy(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(parameters=parameters):
                scaler: Scaler = parameters.build_scaler()
                scaler.fit(X)
                with self.assertRaisesRegex(ValueError, "3 features"):
                    _ = scaler.transform(np.ones((2, 4)))

    def test_failed_fit_leaves_the_scaler_unfitted(self) -> None:
        scaler: Scaler = StandardScalingParameters().build_scaler()
        scaler.fit(X)
        with self.assertRaises(ValueError):
            scaler.fit(np.array([[np.nan, 1.0]]))
        with self.assertRaises(NotFittedError):
            _ = scaler.transform(X)

    def test_never_modifies_inputs(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(parameters=parameters):
                data: NDArray[np.float64] = X.copy()
                data.setflags(write=False)
                scaler: Scaler = parameters.build_scaler()
                self.assertFalse(np.shares_memory(scaler.fit_transform(data), data))

    def test_keeps_single_precision(self) -> None:
        single: NDArray[np.float32] = X.astype(np.float32)
        scaled: FloatArray = (
            StandardScalingParameters().build_scaler().fit_transform(single)
        )
        self.assertEqual(scaled.dtype, np.float32)

    def test_accepts_integers(self) -> None:
        integers: NDArray[np.int64] = np.arange(12, dtype=np.int64).reshape(4, 3)
        scaled: FloatArray = (
            StandardScalingParameters().build_scaler().fit_transform(integers)
        )
        self.assertTrue(np.issubdtype(scaled.dtype, np.floating))

    def test_constant_features_do_not_produce_nan(self) -> None:
        constant: NDArray[np.float64] = np.ones((5, 2))
        for scaling in (
            FeatureScaling.STANDARD,
            FeatureScaling.MIN_MAX,
            FeatureScaling.ROBUST,
        ):
            with self.subTest(scaling=scaling):
                scaled: FloatArray = scaling.build_scaler().fit_transform(constant)
                self.assertTrue(np.isfinite(scaled).all())


class TestMissingValues(unittest.TestCase):
    def test_fit_rejects_features_without_observed_values(self) -> None:
        data: NDArray[np.float64] = X.copy()
        data[:, 1] = np.nan
        for parameters in (
            NoScalingParameters(is_nan_allowed=True),
            StandardScalingParameters(is_nan_allowed=True),
            MinMaxScalingParameters(is_nan_allowed=True),
            MaxAbsScalingParameters(is_nan_allowed=True),
            RobustScalingParameters(is_nan_allowed=True),
            QuantileScalingParameters(n_quantiles=50, is_nan_allowed=True),
            PowerScalingParameters(is_nan_allowed=True),
        ):
            with (
                self.subTest(parameters=parameters),
                self.assertRaisesRegex(ValueError, "observed"),
            ):
                parameters.build_scaler().fit(data)

    def test_allowed_nan_is_ignored_and_stays_nan(self) -> None:
        data: NDArray[np.float64] = np.array([[1.0], [np.nan], [3.0]])
        scaled: FloatArray = (
            StandardScalingParameters(is_nan_allowed=True)
            .build_scaler()
            .fit_transform(data)
        )
        self.assertTrue(np.isnan(scaled.item(1, 0)))
        np.testing.assert_allclose(scaled[[0, 2], 0], [-1.0, 1.0])

    def test_all_strategies_that_support_nan(self) -> None:
        data: NDArray[np.float64] = X.copy()
        data[0, 0] = np.nan
        for parameters in (
            NoScalingParameters(is_nan_allowed=True),
            StandardScalingParameters(is_nan_allowed=True),
            MinMaxScalingParameters(is_nan_allowed=True),
            MaxAbsScalingParameters(is_nan_allowed=True),
            RobustScalingParameters(is_nan_allowed=True),
            QuantileScalingParameters(n_quantiles=50, is_nan_allowed=True),
            PowerScalingParameters(is_nan_allowed=True),
        ):
            with self.subTest(parameters=parameters):
                scaled: FloatArray = parameters.build_scaler().fit_transform(data)
                self.assertTrue(np.isnan(scaled.item(0, 0)))
                self.assertEqual(np.count_nonzero(np.isnan(scaled)), 1)


class TestInverse(unittest.TestCase):
    def test_round_trip(self) -> None:
        for parameters in ALL_METHODS:
            scaler: Scaler = parameters.build_scaler()
            if not isinstance(scaler, InvertibleScaler):
                continue
            with self.subTest(parameters=parameters):
                restored: FloatArray = scaler.inverse_transform(scaler.fit_transform(X))
                np.testing.assert_allclose(restored, X, rtol=1e-6)

    def test_unit_norm_offers_no_inverse(self) -> None:
        self.assertNotIsInstance(
            UnitNormScalingParameters().build_scaler(), InvertibleScaler
        )

    def test_inverse_checks_features(self) -> None:
        scaler: IncrementalScaler = StandardScalingParameters().build_scaler()
        with self.assertRaises(NotFittedError):
            _ = scaler.inverse_transform(X)
        scaler.fit(X)
        with self.assertRaises(ValueError):
            _ = scaler.inverse_transform(np.ones((2, 4)))


class TestIncremental(unittest.TestCase):
    def test_batches_match_fitting_at_once(self) -> None:
        for parameters in (
            StandardScalingParameters(),
            MinMaxScalingParameters(),
            MaxAbsScalingParameters(),
        ):
            with self.subTest(parameters=parameters):
                incremental: IncrementalScaler = parameters.build_scaler()
                for batch in np.array_split(X, 7):
                    incremental.partial_fit(batch)
                at_once: IncrementalScaler = parameters.build_scaler()
                at_once.fit(X)
                np.testing.assert_allclose(
                    incremental.transform(UNSEEN), at_once.transform(UNSEEN)
                )
                self.assertEqual(incremental.n_features, 3)

    def test_failed_fit_does_not_leak_into_later_batches(self) -> None:
        scaler: IncrementalScaler = StandardScalingParameters().build_scaler()
        scaler.fit(X)
        with self.assertRaises(ValueError):
            scaler.fit(np.array([[np.nan, 1.0, 1.0]]))
        scaler.partial_fit(X[:10])
        fresh: IncrementalScaler = StandardScalingParameters().build_scaler()
        fresh.partial_fit(X[:10])
        np.testing.assert_array_equal(scaler.transform(UNSEEN), fresh.transform(UNSEEN))

    def test_rejects_batches_with_other_features(self) -> None:
        scaler: IncrementalScaler = StandardScalingParameters().build_scaler()
        scaler.partial_fit(X)
        with self.assertRaises(ValueError):
            scaler.partial_fit(np.ones((2, 4)))

    def test_rejects_invalid_first_batch(self) -> None:
        with self.assertRaises(ValueError):
            StandardScalingParameters().build_scaler().partial_fit(np.array([[np.nan]]))


class TestPersistence(unittest.TestCase):
    def test_saved_scaler_gives_identical_results(self) -> None:
        for parameters in ALL_METHODS:
            with (
                self.subTest(parameters=parameters),
                tempfile.TemporaryDirectory() as directory,
            ):
                path: Path = Path(directory) / "scaler.npz"
                scaler: Scaler = parameters.build_scaler()
                scaler.fit(X)
                scaler.save(path)
                loaded: Scaler = ScalerLoader.load(path)
                self.assertEqual(loaded.parameters, parameters)
                self.assertIs(type(loaded), type(scaler))
                self.assertEqual(loaded.n_features, 3)
                np.testing.assert_array_equal(
                    loaded.transform(UNSEEN), scaler.transform(UNSEEN)
                )
                if isinstance(loaded, InvertibleScaler) and isinstance(
                    scaler, InvertibleScaler
                ):
                    scaled: FloatArray = scaler.transform(UNSEEN)
                    np.testing.assert_array_equal(
                        loaded.inverse_transform(scaled),
                        scaler.inverse_transform(scaled),
                    )

    def test_loaded_incremental_scaler_continues_learning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "scaler.npz"
            first: IncrementalScaler = StandardScalingParameters().build_scaler()
            first.partial_fit(X[:100])
            first.save(path)
            loaded: Scaler = ScalerLoader.load(path)
            assert isinstance(loaded, IncrementalScaler)
            loaded.partial_fit(X[100:])
            at_once: IncrementalScaler = StandardScalingParameters().build_scaler()
            at_once.fit(X)
            np.testing.assert_allclose(
                loaded.transform(UNSEEN), at_once.transform(UNSEEN)
            )

    def test_every_incremental_configuration_continues_learning(self) -> None:
        for parameters in (
            StandardScalingParameters(),
            StandardScalingParameters(is_centered=False),
            StandardScalingParameters(is_scaled=False),
            StandardScalingParameters(is_centered=False, is_scaled=False),
            MinMaxScalingParameters(),
            MaxAbsScalingParameters(),
        ):
            with (
                self.subTest(parameters=parameters),
                tempfile.TemporaryDirectory() as directory,
            ):
                path: Path = Path(directory) / "scaler.npz"
                first: Scaler = parameters.build_scaler()
                assert isinstance(first, IncrementalScaler)
                first.partial_fit(X[:100])
                first.save(path)
                loaded: Scaler = ScalerLoader.load(path)
                assert isinstance(loaded, IncrementalScaler)
                loaded.partial_fit(X[100:])
                at_once: Scaler = parameters.build_scaler()
                at_once.fit(X)
                np.testing.assert_allclose(
                    loaded.transform(UNSEEN), at_once.transform(UNSEEN)
                )

    def test_restore_rejects_the_state_of_another_configuration(self) -> None:
        source: IncrementalScaler = MinMaxScalingParameters().build_scaler()
        source.fit(X)
        for target in (
            StandardScalingParameters().build_scaler(),
            MinMaxScalingParameters(is_clipped=True).build_scaler(),
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                target.restore_state(source.to_state())

    def test_rejects_malformed_states(self) -> None:
        source: IncrementalScaler = StandardScalingParameters().build_scaler()
        source.fit(X)
        state: ScalerState = source.to_state()
        arrays: dict[str, NDArray[np.generic]] = dict(state.arrays)
        valid: dict[str, JsonValue] = {
            "format_version": FORMAT_VERSION,
            "strategy": state.strategy.value,
            "parameters": dict(state.parameters),
            "n_features": state.n_features,
            "scikit_learn_version": state.scikit_learn_version,
            "empty_attributes": [],
        }
        cases: list[tuple[JsonValue, dict[str, NDArray[np.generic]]]] = [
            ([1], arrays),
            ({key: valid[key] for key in valid if key != "format_version"}, arrays),
            ({key: valid[key] for key in valid if key != "strategy"}, arrays),
            ({key: valid[key] for key in valid if key != "n_features"}, arrays),
            ({**valid, "n_features": "3"}, arrays),
            ({**valid, "n_features": 0}, arrays),
            ({**valid, "n_features": 4}, arrays),
            ({**valid, "empty_attributes": [1]}, arrays),
            ({**valid, "parameters": {"unknown": 1}}, arrays),
            ({**valid, "parameters": {"is_centered": "yes"}}, arrays),
            ({**valid, "strategy": "unknown"}, arrays),
            ({**valid, "strategy": ["standard"]}, arrays),
            ({**valid, "scikit_learn_version": 1}, arrays),
            (valid, {}),
            (valid, {name: arrays[name] for name in arrays if name != "scale_"}),
            (valid, {**arrays, "transform": np.zeros(3)}),
            (valid, {**arrays, "with_mean": np.array(0)}),
            (valid, {**arrays, "scale_": np.array(["a", "b", "c"])}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "state.npz"
            for metadata, entries in cases:
                with (
                    self.subTest(metadata=metadata, entries=sorted(entries)),
                    self.assertRaises(ValueError),
                ):
                    with path.open("wb") as file:
                        np.savez(
                            file,
                            allow_pickle=False,
                            **{"__metadata__": np.array(json.dumps(metadata))},
                            **entries,
                        )
                    _ = ScalerLoader.load(path)

    def test_every_strategy_can_be_loaded_with_its_kind(self) -> None:
        strategies: set[ScalingStrategy] = set()
        for parameters in ALL_METHODS:
            with (
                self.subTest(parameters=parameters),
                tempfile.TemporaryDirectory() as directory,
            ):
                path: Path = Path(directory) / "scaler.npz"
                scaler: Scaler = parameters.build_scaler()
                scaler.fit(X)
                scaler.save(path)
                kind: ScalingKind = ScalerLoader.load(path).parameters.kind
                self.assertIs(kind, parameters.kind)
                strategies.add(parameters.strategy)
        self.assertEqual(strategies, set(ScalingStrategy))

    def test_warns_about_another_scikit_learn_version(self) -> None:
        source: IncrementalScaler = StandardScalingParameters().build_scaler()
        source.fit(X)
        state: ScalerState = replace(source.to_state(), scikit_learn_version="0.0")
        with self.assertWarnsRegex(UserWarning, "scikit-learn 0.0"):
            _ = ScalerLoader.from_state(state)

    def test_rejects_a_bare_array_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "array.npy"
            with path.open("wb") as file:
                np.save(file, np.zeros(2))
            with self.assertRaises(ValueError):
                _ = ScalerLoader.load(path)

    def test_rejects_invalid_parameter_values_in_a_state(self) -> None:
        source: IncrementalScaler = MinMaxScalingParameters().build_scaler()
        source.fit(X)
        state: ScalerState = source.to_state()
        with self.assertRaises(ValueError):
            _ = ScalerLoader.from_state(
                replace(state, parameters={**state.parameters, "feature_range": "ab"})
            )

    def test_rejects_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "other.npz"
            with path.open("wb") as file:
                np.savez(file, values=np.zeros(2))
            with self.assertRaises(ValueError):
                _ = ScalerLoader.load(path)

    def test_loading_never_unpickles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "pickled.npz"
            with path.open("wb") as file:
                np.savez(file, __metadata__=np.array([{"code": 1}], dtype=object))
            with self.assertRaises(ValueError):
                _ = ScalerLoader.load(path)


class TestScikitLearnInterop(unittest.TestCase):
    def test_every_strategy_works_in_a_cloneable_pipeline(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(parameters=parameters):
                pipeline: Pipeline[np.integer] = parameters.build_pipeline(
                    IsolationForest(n_estimators=5, random_state=0)
                )
                copied: Pipeline[np.integer] = clone(pipeline)
                _ = copied.fit(X)
                self.assertEqual(copied.predict(X).shape, (300,))

    def test_shorthand_builds_the_same_steps(self) -> None:
        pipeline: Pipeline[np.integer] = FeatureScaling.ROBUST.build_pipeline(
            IsolationForest(n_estimators=5, random_state=0)
        )
        self.assertEqual(list(pipeline.named_steps), ["scaler", "estimator"])
