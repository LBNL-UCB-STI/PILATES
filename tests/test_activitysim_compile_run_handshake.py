from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from pilates.activitysim.runner import ActivitysimCompileRunner
from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
from pilates.generic.records import FileRecord, RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    ASIM_SHARROW_CACHE_DIR,
    ArtifactKeys,
    ZARR_SKIMS,
)
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows.steps import activitysim as activitysim_steps


class _DummyCoupler:
    def __init__(self) -> None:
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value) -> None:
        self._data[key] = value


class _DummyWorkspace:
    def __init__(self, root: Path, asim_output_dir: Path) -> None:
        self.full_path = str(root)
        self._asim_output_dir = asim_output_dir

    def get_asim_output_dir(self) -> str:
        return str(self._asim_output_dir)


class _CompileRunner:
    def __init__(self, zarr_path: Path, cache_path: Optional[Path] = None) -> None:
        self._zarr_path = zarr_path
        self._cache_path = cache_path

    def run(self, input_store: RecordStore, workspace: _DummyWorkspace) -> RecordStore:
        records = [
            FileRecord(
                file_path=str(self._zarr_path),
                short_name=ZARR_SKIMS,
                description="Compiled ActivitySim skims (Zarr)",
            )
        ]
        if self._cache_path is not None:
            records.append(
                FileRecord(
                    file_path=str(self._cache_path),
                    short_name=ASIM_SHARROW_CACHE_DIR,
                    description="ActivitySim persisted compile cache",
                )
            )
        return RecordStore(recordList=records)


def _settings(*, persist_sharrow_cache: bool) -> SimpleNamespace:
    return SimpleNamespace(
        activitysim=SimpleNamespace(
            file_format="parquet",
            persist_sharrow_cache=persist_sharrow_cache,
        )
    )


def test_activitysim_compile_run_zarr_handshake(monkeypatch, tmp_path: Path) -> None:
    asim_output_dir = tmp_path / "asim_output"
    asim_output_dir.mkdir(parents=True)
    zarr_path = asim_output_dir / "cache" / "skims.zarr"
    zarr_path.parent.mkdir(parents=True)
    zarr_path.write_text("dummy-zarr")

    asim_mutable_dir = tmp_path / "asim_mutable"
    asim_mutable_dir.mkdir(parents=True)
    land_use = asim_mutable_dir / "land_use.csv"
    households = asim_mutable_dir / "households.csv"
    persons = asim_mutable_dir / "persons.csv"
    for path in (land_use, households, persons):
        path.write_text("dummy")

    workspace = _DummyWorkspace(tmp_path, asim_output_dir)
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, forecast_year=2020, iteration=0)
    coupler = _DummyCoupler()
    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_mutable_dir,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
    )

    compile_runner = _CompileRunner(zarr_path)
    run_runner = object()

    def _get_runner(self, model_name, state=None, major_stage=None):
        if model_name == "activitysim_compile":
            return compile_runner
        if model_name == "activitysim":
            return run_runner
        raise AssertionError(f"Unexpected model_name: {model_name}")

    captured: dict = {}

    def _capture_execute_run(runner, workspace, outputs_holder, **kwargs):
        captured["runner"] = runner
        captured["extra_inputs"] = kwargs.get("extra_inputs")
        return RecordStore()

    monkeypatch.setattr(activitysim_steps.ModelFactory, "get_runner", _get_runner)
    monkeypatch.setattr(activitysim_steps, "_execute_run", _capture_execute_run)

    compile_step = activitysim_steps.make_activitysim_compile_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )
    compile_step(
        settings=settings,
        state=state,
        workspace=workspace,
        expected_outputs={ZARR_SKIMS: str(zarr_path)},
    )

    coupler_zarr_path = artifact_to_path(coupler.get(ZARR_SKIMS), workspace)
    assert coupler_zarr_path == str(zarr_path)

    run_step = activitysim_steps.make_activitysim_run_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )
    run_step(settings=settings, state=state, workspace=workspace)

    assert captured["runner"] is run_runner
    extra_inputs = captured["extra_inputs"]
    assert isinstance(extra_inputs, RecordStore)
    assert extra_inputs.to_mapping().get(ZARR_SKIMS) == str(zarr_path)


def test_activitysim_compile_cache_key_and_schema_gating(tmp_path: Path) -> None:
    asim_output_dir = tmp_path / "asim_output"
    workspace = _DummyWorkspace(tmp_path, asim_output_dir)
    state = SimpleNamespace()
    expected_cache_dir = tmp_path / "shared_cache" / "numba"

    settings_on = _settings(persist_sharrow_cache=True)
    settings_off = _settings(persist_sharrow_cache=False)

    outputs_on = ActivitysimCompileRunner.expected_outputs(settings_on, state, workspace)
    outputs_off = ActivitysimCompileRunner.expected_outputs(settings_off, state, workspace)

    assert ArtifactKeys.ASIM_SHARROW_CACHE_DIR == ASIM_SHARROW_CACHE_DIR
    assert ASIM_SHARROW_CACHE_DIR in outputs_on
    assert outputs_on[ASIM_SHARROW_CACHE_DIR] == str(expected_cache_dir)
    assert ASIM_SHARROW_CACHE_DIR not in outputs_off

    coupler = _DummyCoupler()
    holder = StepOutputsHolder()
    compile_step = activitysim_steps.make_activitysim_compile_step(
        coupler=coupler,
        outputs_holder=holder,
    )
    schema_on = build_coupler_schema([compile_step], settings=settings_on, include_extras=False)
    schema_off = build_coupler_schema(
        [compile_step], settings=settings_off, include_extras=False
    )
    assert ASIM_SHARROW_CACHE_DIR in schema_on
    assert ASIM_SHARROW_CACHE_DIR not in schema_off


def test_activitysim_compile_logs_cache_output_when_gate_on(
    monkeypatch, tmp_path: Path
) -> None:
    asim_output_dir = tmp_path / "asim_output"
    asim_output_dir.mkdir(parents=True)
    zarr_path = asim_output_dir / "cache" / "skims.zarr"
    zarr_path.parent.mkdir(parents=True)
    zarr_path.write_text("dummy-zarr")
    cache_dir = tmp_path / "shared_cache" / "numba"
    (cache_dir / "sub").mkdir(parents=True)
    (cache_dir / "sub" / "entry.bin").write_text("cache")

    asim_mutable_dir = tmp_path / "asim_mutable"
    asim_mutable_dir.mkdir(parents=True)
    land_use = asim_mutable_dir / "land_use.csv"
    households = asim_mutable_dir / "households.csv"
    persons = asim_mutable_dir / "persons.csv"
    for path in (land_use, households, persons):
        path.write_text("dummy")

    workspace = _DummyWorkspace(tmp_path, asim_output_dir)
    settings = _settings(persist_sharrow_cache=True)
    state = SimpleNamespace(year=2020, forecast_year=2020, iteration=0)
    coupler = _DummyCoupler()
    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_mutable_dir,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
    )

    compile_runner = _CompileRunner(zarr_path, cache_dir)

    def _get_runner(self, model_name, state=None, major_stage=None):
        if model_name != "activitysim_compile":
            raise AssertionError(f"Unexpected model_name: {model_name}")
        return compile_runner

    logged_outputs = {}

    def _capture_log_and_set_output(**kwargs):
        logged_outputs[kwargs["key"]] = kwargs["path"]

    monkeypatch.setattr(activitysim_steps.ModelFactory, "get_runner", _get_runner)
    monkeypatch.setattr(activitysim_steps, "log_and_set_output", _capture_log_and_set_output)

    compile_step = activitysim_steps.make_activitysim_compile_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )
    compile_step(
        settings=settings,
        state=state,
        workspace=workspace,
        expected_outputs={
            ZARR_SKIMS: str(zarr_path),
            ASIM_SHARROW_CACHE_DIR: str(cache_dir),
        },
    )

    assert ZARR_SKIMS in logged_outputs
    assert ASIM_SHARROW_CACHE_DIR in logged_outputs
    assert logged_outputs[ASIM_SHARROW_CACHE_DIR] == str(cache_dir)


def test_activitysim_compile_does_not_log_cache_output_when_gate_off(
    monkeypatch, tmp_path: Path
) -> None:
    asim_output_dir = tmp_path / "asim_output"
    asim_output_dir.mkdir(parents=True)
    zarr_path = asim_output_dir / "cache" / "skims.zarr"
    zarr_path.parent.mkdir(parents=True)
    zarr_path.write_text("dummy-zarr")
    cache_dir = tmp_path / "shared_cache" / "numba"
    cache_dir.mkdir(parents=True)
    (cache_dir / "entry.bin").write_text("cache")

    asim_mutable_dir = tmp_path / "asim_mutable"
    asim_mutable_dir.mkdir(parents=True)
    land_use = asim_mutable_dir / "land_use.csv"
    households = asim_mutable_dir / "households.csv"
    persons = asim_mutable_dir / "persons.csv"
    for path in (land_use, households, persons):
        path.write_text("dummy")

    workspace = _DummyWorkspace(tmp_path, asim_output_dir)
    settings = _settings(persist_sharrow_cache=False)
    state = SimpleNamespace(year=2020, forecast_year=2020, iteration=0)
    coupler = _DummyCoupler()
    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_mutable_dir,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
    )

    compile_runner = _CompileRunner(zarr_path, cache_dir)

    def _get_runner(self, model_name, state=None, major_stage=None):
        if model_name != "activitysim_compile":
            raise AssertionError(f"Unexpected model_name: {model_name}")
        return compile_runner

    logged_keys = []

    def _capture_log_and_set_output(**kwargs):
        logged_keys.append(kwargs["key"])

    monkeypatch.setattr(activitysim_steps.ModelFactory, "get_runner", _get_runner)
    monkeypatch.setattr(activitysim_steps, "log_and_set_output", _capture_log_and_set_output)

    compile_step = activitysim_steps.make_activitysim_compile_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )
    compile_step(
        settings=settings,
        state=state,
        workspace=workspace,
        expected_outputs={
            ZARR_SKIMS: str(zarr_path),
            ASIM_SHARROW_CACHE_DIR: str(cache_dir),
        },
    )

    assert ZARR_SKIMS in logged_keys
    assert ASIM_SHARROW_CACHE_DIR not in logged_keys
