from __future__ import annotations

import json
import tempfile
import unittest
import warnings
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np
from numpy.typing import NDArray
from sklearn import config_context
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError

from data_scaler import (
    FeatureScaling,
    FloatArray,
    IncrementalScaler,
    InvertibleScaler,
    JsonValue,
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    Norm,
    NoScalingParameters,
    NumericArray,
    OutputDistribution,
    PipelineLike,
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

    def test_rejected_fit_keeps_the_fitted_statistics(self) -> None:
        scaler: Scaler = StandardScalingParameters().build_scaler()
        scaler.fit(X)
        expected: FloatArray = scaler.transform(UNSEEN)
        with self.assertRaises(ValueError):
            scaler.fit(np.array([[np.nan, 1.0]]))
        np.testing.assert_array_equal(scaler.transform(UNSEEN), expected)

    def test_failed_learning_leaves_the_scaler_unfitted(self) -> None:
        scaler: Scaler = PowerScalingParameters(
            method=PowerMethod.BOX_COX
        ).build_scaler()
        scaler.fit(X)
        with self.assertRaises(ValueError):
            scaler.fit(-X)
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

    def test_rejects_what_is_not_an_array(self) -> None:
        for features in ([[1.0, 2.0]], ((1.0, 2.0),), 1.0, None):
            with self.subTest(features=features), self.assertRaises(TypeError):
                StandardScalingParameters().build_scaler().fit(
                    cast(NumericArray, cast(object, features))
                )

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
        with (
            warnings.catch_warnings(),
            self.assertRaises(ValueError),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            scaler.fit(np.array([[1.7e308, 1.0, 1.0], [-1.7e308, 2.0, 2.0]]))
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

    def test_rejects_batches_with_an_unobserved_feature(self) -> None:
        observed: NDArray[np.float64] = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        )
        unobserved: NDArray[np.float64] = np.array([[4.0, np.nan], [5.0, np.nan]])
        for parameters in (
            StandardScalingParameters(is_nan_allowed=True),
            MinMaxScalingParameters(is_nan_allowed=True),
            MaxAbsScalingParameters(is_nan_allowed=True),
        ):
            with self.subTest(parameters=parameters):
                scaler: IncrementalScaler = parameters.build_scaler()
                with self.assertRaisesRegex(ValueError, "observed"):
                    scaler.partial_fit(unobserved)
                scaler.partial_fit(observed)
                expected: FloatArray = scaler.transform(observed)
                with self.assertRaisesRegex(ValueError, "observed"):
                    scaler.partial_fit(unobserved)
                np.testing.assert_array_equal(scaler.transform(observed), expected)

    def test_overflowing_batch_unfits_the_scaler(self) -> None:
        huge: NDArray[np.float64] = np.array([[1.7e308, 1.0], [-1.7e308, 2.0]])
        for parameters in (StandardScalingParameters(), MinMaxScalingParameters()):
            with self.subTest(parameters=parameters), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scaler: IncrementalScaler = parameters.build_scaler()
                scaler.fit(X[:, :2])
                with self.assertRaisesRegex(ValueError, "overflowed"):
                    scaler.partial_fit(huge)
                with self.assertRaises(NotFittedError):
                    _ = scaler.transform(X[:, :2])
                scaler.partial_fit(X[:, :2])
                reference: IncrementalScaler = parameters.build_scaler()
                reference.fit(X[:, :2])
                np.testing.assert_array_equal(
                    scaler.transform(X[:, :2]), reference.transform(X[:, :2])
                )

    def test_fit_rejects_features_that_overflow_statistics(self) -> None:
        huge: NDArray[np.float64] = np.array([[1.7e308, 1.0], [-1.7e308, 2.0]])
        for parameters in (
            StandardScalingParameters(),
            MinMaxScalingParameters(),
            PowerScalingParameters(),
        ):
            with self.subTest(parameters=parameters), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scaler: Scaler = parameters.build_scaler()
                with self.assertRaisesRegex(ValueError, "overflowed"):
                    scaler.fit(huge)
                with self.assertRaises(NotFittedError):
                    _ = scaler.transform(huge)


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
            loaded: IncrementalScaler = ScalerLoader.load(path, IncrementalScaler)
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
                loaded: IncrementalScaler = ScalerLoader.load(path, IncrementalScaler)
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
        values: list[JsonValue] = [
            "ab",
            [0.0],
            [0.0, 1.0, 2.0],
            [[0.0], 1.0],
            [10**400, 10**401],
        ]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _ = ScalerLoader.from_state(
                    replace(
                        state, parameters={**state.parameters, "feature_range": value}
                    )
                )

    def test_rejects_damaged_files(self) -> None:
        source: IncrementalScaler = StandardScalingParameters().build_scaler()
        source.fit(X)
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "scaler.npz"
            source.save(path)
            content: bytes = path.read_bytes()
            for damaged in (b"", content[: len(content) // 2], content[:4]):
                with self.subTest(size=len(damaged)), self.assertRaises(ValueError):
                    _ = path.write_bytes(damaged)
                    _ = ScalerLoader.load(path)

    def test_corrupted_bytes_raise_only_value_error(self) -> None:
        source: IncrementalScaler = StandardScalingParameters().build_scaler()
        source.fit(X)
        generator: np.random.Generator = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "scaler.npz"
            source.save(path)
            content: bytes = path.read_bytes()
            for _ in range(2000):
                damaged: bytearray = bytearray(content)
                for _ in range(4):
                    position: int = int(generator.integers(0, len(content)))
                    damaged[position] = int(generator.integers(0, 256))
                _ = path.write_bytes(bytes(damaged))
                try:
                    _ = ScalerLoader.load(path)
                except ValueError:
                    pass

    def test_missing_file_is_not_reported_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                _ = ScalerLoader.load(Path(directory) / "missing.npz")

    def test_rejects_invalid_statistics(self) -> None:
        source: IncrementalScaler = StandardScalingParameters().build_scaler()
        source.fit(X)
        state: ScalerState = source.to_state()
        cases: dict[str, NDArray[np.generic]] = {
            "mean_": np.array(1.0),
            "var_": np.ones(1),
            "scale_": np.ones(4),
            "n_samples_seen_": np.ones((3, 3)),
        }
        for name, array in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "shape"):
                _ = ScalerLoader.from_state(
                    replace(state, arrays={**state.arrays, name: array})
                )
        for name, value in (
            ("mean_", np.array([np.nan, 0.0, 0.0])),
            ("mean_", np.array([np.inf, 0.0, 0.0])),
            ("mean_", np.array([1j, 0.0, 0.0])),
            ("scale_", np.array([0.0, 1.0, 1.0])),
            ("scale_", np.array([-1.0, 1.0, 1.0])),
            ("var_", np.array([-1.0, 1.0, 1.0])),
            ("n_samples_seen_", np.array(0)),
            ("n_samples_seen_", np.array(-3)),
            ("n_samples_seen_", np.array([300, 0, 300])),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                _ = ScalerLoader.from_state(
                    replace(state, arrays={**state.arrays, name: value})
                )

    def test_rejects_absurdly_sized_arrays_in_files(self) -> None:
        source: IncrementalScaler = StandardScalingParameters().build_scaler()
        source.fit(X)
        header: bytes = (
            b"{'descr': '<f8', 'fortran_order': False, 'shape': (1000000000000000,), }".ljust(
                118
            )
            + b"\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "scaler.npz"
            source.save(path)
            with zipfile.ZipFile(path) as archive:
                members: dict[str, bytes] = {
                    name: archive.read(name) for name in archive.namelist()
                }
            members["mean_.npy"] = (
                b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
            )
            with zipfile.ZipFile(path, "w") as archive:
                for name, content in members.items():
                    archive.writestr(name, content)
            with self.assertRaises(ValueError):
                _ = ScalerLoader.load(path)

    def test_rejects_inconsistent_quantile_statistics(self) -> None:
        source: Scaler = QuantileScalingParameters(n_quantiles=10).build_scaler()
        source.fit(X)
        state: ScalerState = source.to_state()
        for name, array in (
            ("quantiles_", np.ones((10, 2))),
            ("quantiles_", np.ones((9, 3))),
            ("references_", np.ones(9)),
            ("references_", np.ones((10, 1))),
        ):
            with (
                self.subTest(name=name, shape=array.shape),
                self.assertRaises(ValueError),
            ):
                _ = ScalerLoader.from_state(
                    replace(state, arrays={**state.arrays, name: array})
                )

    def test_rejects_wrongly_empty_statistics(self) -> None:
        source: IncrementalScaler = StandardScalingParameters().build_scaler()
        source.fit(X)
        state: ScalerState = source.to_state()
        arrays: dict[str, NDArray[np.generic]] = {
            name: array for name, array in state.arrays.items() if name != "mean_"
        }
        with self.assertRaisesRegex(ValueError, "empty"):
            _ = ScalerLoader.from_state(
                replace(state, arrays=arrays, empty_attributes=("mean_",))
            )

    def test_rejects_statistics_that_should_be_empty(self) -> None:
        source: IncrementalScaler = StandardScalingParameters(
            is_scaled=False
        ).build_scaler()
        source.fit(X)
        state: ScalerState = source.to_state()
        with self.assertRaisesRegex(ValueError, "empty"):
            _ = ScalerLoader.from_state(
                replace(
                    state,
                    arrays={
                        **state.arrays,
                        "scale_": np.ones(3),
                        "var_": np.ones(3),
                    },
                    empty_attributes=(),
                )
            )

    def test_rejects_unordered_quantile_statistics(self) -> None:
        source: Scaler = QuantileScalingParameters(n_quantiles=10).build_scaler()
        source.fit(X)
        state: ScalerState = source.to_state()
        quantiles: NDArray[np.generic] = np.array(state.arrays["quantiles_"])[::-1]
        references: NDArray[np.generic] = np.zeros(10)
        for name, array in (
            ("quantiles_", quantiles),
            ("references_", references),
            ("n_quantiles_", np.array(9)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _ = ScalerLoader.from_state(
                    replace(state, arrays={**state.arrays, name: array})
                )

    def test_state_is_independent_of_the_scaler(self) -> None:
        parameters: StandardScalingParameters = StandardScalingParameters(
            is_centered=False, is_scaled=False, is_nan_allowed=True
        )
        scaler: IncrementalScaler = parameters.build_scaler()
        scaler.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))
        state: ScalerState = scaler.to_state()
        counts: NDArray[np.generic] = np.array(state.arrays["n_samples_seen_"])
        restored: IncrementalScaler = ScalerLoader.from_state(state, IncrementalScaler)
        batch: NDArray[np.float64] = np.ones((2, 2))
        scaler.partial_fit(batch)
        restored.partial_fit(batch)
        np.testing.assert_array_equal(state.arrays["n_samples_seen_"], counts)

    def test_state_statistics_are_read_only(self) -> None:
        scaler: IncrementalScaler = StandardScalingParameters().build_scaler()
        scaler.fit(X)
        expected: FloatArray = scaler.transform(UNSEEN)
        array: NDArray[np.generic] = scaler.to_state().arrays["mean_"]
        with self.assertRaises(ValueError):
            array[0] = 999.0
        np.testing.assert_array_equal(scaler.transform(UNSEEN), expected)

    def test_load_narrows_to_the_requested_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "scaler.npz"
            scaler: IncrementalScaler = StandardScalingParameters().build_scaler()
            scaler.fit(X)
            scaler.save(path)
            incremental: IncrementalScaler = ScalerLoader.load(path, IncrementalScaler)
            invertible: InvertibleScaler = ScalerLoader.load(path, InvertibleScaler)
            self.assertIs(type(incremental), IncrementalScaler)
            self.assertIs(type(invertible), IncrementalScaler)

    def test_load_rejects_another_type_before_restoring(self) -> None:
        scaler: Scaler = UnitNormScalingParameters().build_scaler()
        scaler.fit(X)
        state: ScalerState = replace(scaler.to_state(), scikit_learn_version="0.0")
        with self.assertRaisesRegex(TypeError, "InvertibleScaler"):
            _ = ScalerLoader.from_state(state, InvertibleScaler)

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


class TestImpossibleStatistics(unittest.TestCase):
    def test_every_configuration_round_trips_at_extreme_magnitudes(self) -> None:
        for parameters in ALL_METHODS:
            for magnitude in (1e-150, 1e150):
                with (
                    self.subTest(parameters=parameters, magnitude=magnitude),
                    warnings.catch_warnings(),
                ):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    data: NDArray[np.float64] = X * magnitude
                    scaler: Scaler = parameters.build_scaler()
                    try:
                        scaler.fit(data)
                    except ValueError:
                        # Refusing to learn is fine, but learning promises a usable state.
                        continue
                    _ = ScalerLoader.from_state(scaler.to_state())

    def test_robust_rejects_unit_variance_with_infinite_quantiles(self) -> None:
        for quantile_range in ((0.0, 100.0), (0.0, 50.0), (50.0, 100.0)):
            with self.subTest(quantile_range=quantile_range):
                with self.assertRaises(ValueError):
                    _ = RobustScalingParameters(
                        quantile_range=quantile_range, is_unit_variance=True
                    )
                _ = RobustScalingParameters(quantile_range=quantile_range)
                _ = RobustScalingParameters(
                    quantile_range=quantile_range,
                    is_unit_variance=True,
                    is_scaled=False,
                )

    def test_fit_rejects_scales_that_underflow(self) -> None:
        scaler: Scaler = MinMaxScalingParameters(
            feature_range=(0.0, 1e-300)
        ).build_scaler()
        with self.assertRaisesRegex(ValueError, "positive"):
            scaler.fit(np.array([[-5e307], [5e307]]))
        with self.assertRaises(NotFittedError):
            _ = scaler.transform(np.array([[1.0]]))

    def test_load_rejects_statistics_that_disagree(self) -> None:
        cases: list[tuple[ScalingParameters, str, NDArray[np.generic]]] = [
            (MinMaxScalingParameters(), "data_min_", np.full(3, 9e9)),
            (MinMaxScalingParameters(), "data_range_", np.full(3, -1.0)),
            (MinMaxScalingParameters(), "data_range_", np.full(3, 1e9)),
            (MinMaxScalingParameters(), "scale_", np.full(3, 1e9)),
            (MinMaxScalingParameters(), "min_", np.full(3, 1e9)),
            (MaxAbsScalingParameters(), "max_abs_", np.full(3, -1.0)),
            (MaxAbsScalingParameters(), "scale_", np.full(3, 1e9)),
            (StandardScalingParameters(), "scale_", np.full(3, 1e9)),
            (StandardScalingParameters(), "n_samples_seen_", np.array(1.5)),
            (StandardScalingParameters(), "n_features_in_", np.array(3.0)),
            (PowerScalingParameters(), "standard.scale_", np.full(3, 1e9)),
            (QuantileScalingParameters(n_quantiles=10), "references_", np.arange(10.0)),
        ]
        for parameters, name, array in cases:
            with self.subTest(parameters=parameters, name=name):
                scaler: Scaler = parameters.build_scaler()
                scaler.fit(X)
                state: ScalerState = scaler.to_state()
                with self.assertRaises(ValueError):
                    _ = ScalerLoader.from_state(
                        replace(state, arrays={**state.arrays, name: array})
                    )

    def test_load_rejects_more_quantiles_than_configured(self) -> None:
        scaler: Scaler = QuantileScalingParameters(n_quantiles=10).build_scaler()
        scaler.fit(X)
        state: ScalerState = scaler.to_state()
        smaller: dict[str, JsonValue] = {**state.parameters, "n_quantiles": 5}
        with self.assertRaisesRegex(ValueError, "configured"):
            _ = ScalerLoader.from_state(replace(state, parameters=smaller))

    def test_quantile_subsample_without_observed_values_is_rejected(self) -> None:
        sparse: NDArray[np.float64] = np.full((20000, 1), np.nan)
        sparse[-1, 0] = 1.0
        scaler: Scaler = QuantileScalingParameters(
            n_quantiles=10, subsample=1000, is_nan_allowed=True
        ).build_scaler()
        with warnings.catch_warnings(), self.assertRaisesRegex(ValueError, "subsample"):
            warnings.simplefilter("ignore", RuntimeWarning)
            scaler.fit(sparse)


class TestOutputs(unittest.TestCase):
    def test_inverse_rejects_values_outside_the_invertible_range(self) -> None:
        scaler: InvertibleScaler = PowerScalingParameters().build_scaler()
        skewed: NDArray[np.float64] = np.random.default_rng(1).lognormal(
            sigma=1.5, size=(1000, 1)
        )
        scaler.fit(skewed)
        with warnings.catch_warnings(), self.assertRaisesRegex(ValueError, "finite"):
            warnings.simplefilter("ignore")
            _ = scaler.inverse_transform(np.array([[20.0]]))

    def test_inverse_rejects_overflow(self) -> None:
        scaler: IncrementalScaler = StandardScalingParameters().build_scaler()
        scaler.fit(np.array([[0.0], [1e30]], dtype=np.float32))
        with warnings.catch_warnings(), self.assertRaisesRegex(ValueError, "finite"):
            warnings.simplefilter("ignore", RuntimeWarning)
            _ = scaler.inverse_transform(np.array([[1e10]], dtype=np.float32))

    def test_allowed_nan_does_not_count_as_a_new_non_finite_value(self) -> None:
        scaler: IncrementalScaler = StandardScalingParameters(
            is_nan_allowed=True
        ).build_scaler()
        scaler.fit(np.array([[1.0], [2.0]]))
        self.assertTrue(np.isnan(scaler.transform(np.array([[np.nan]]))).all())

    def test_unit_norm_rejects_samples_of_extreme_magnitude(self) -> None:
        for norm, values in (
            (Norm.L2, (1e200, 1e200)),
            (Norm.L1, (1.7e308, 1.7e308)),
        ):
            with (
                self.subTest(norm=norm, values=values),
                warnings.catch_warnings(),
                self.assertRaisesRegex(ValueError, "length"),
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                scaler: Scaler = UnitNormScalingParameters(norm=norm).build_scaler()
                scaler.fit(np.ones((2, 2)))
                _ = scaler.transform(np.array([values]))

    def test_unit_norm_keeps_samples_of_nearly_zero_length(self) -> None:
        for norm in Norm:
            for value in (0.0, 1e-200, 1e-17):
                with self.subTest(norm=norm, value=value):
                    scaler: Scaler = UnitNormScalingParameters(norm=norm).build_scaler()
                    tiny: NDArray[np.float64] = np.full((1, 2), value)
                    np.testing.assert_array_equal(scaler.fit_transform(tiny), tiny)

    def test_ignores_the_global_output_configuration(self) -> None:
        for parameters in ALL_METHODS:
            with (
                self.subTest(parameters=parameters),
                config_context(transform_output="pandas"),
            ):
                scaled: FloatArray = parameters.build_scaler().fit_transform(X)
                self.assertIs(type(scaled), np.ndarray)


class TestAtomicSave(unittest.TestCase):
    def test_failed_save_keeps_the_existing_file_and_no_temporary(self) -> None:
        scaler: IncrementalScaler = StandardScalingParameters().build_scaler()
        scaler.fit(X)
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "scaler.npz"
            scaler.save(path)
            content: bytes = path.read_bytes()
            fail: Callable[..., None] = mock.Mock(side_effect=OSError("disk full"))
            with (
                mock.patch("data_scaler.scaler_state.np.savez", fail),
                self.assertRaises(OSError),
            ):
                scaler.save(path)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(
                [entry.name for entry in Path(directory).iterdir()], [path.name]
            )


class TestScikitLearnInterop(unittest.TestCase):
    def test_every_strategy_works_in_a_cloneable_pipeline(self) -> None:
        for parameters in ALL_METHODS:
            with self.subTest(parameters=parameters):
                pipeline: PipelineLike[np.integer] = parameters.build_pipeline(
                    IsolationForest(n_estimators=5, random_state=0)
                )
                copied: PipelineLike[np.integer] = clone(pipeline)
                self.assertEqual(copied.fit(X).predict(X).shape, (300,))

    def test_shorthand_builds_the_same_steps(self) -> None:
        pipeline: PipelineLike[np.integer] = FeatureScaling.ROBUST.build_pipeline(
            IsolationForest(n_estimators=5, random_state=0)
        )
        self.assertEqual(list(pipeline.named_steps), ["scaler", "estimator"])
