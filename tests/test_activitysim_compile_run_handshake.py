from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from pilates.activitysim.runner import ActivitysimCompileRunner
from pilates.activitysim.runner import ActivitysimRunner
from pilates.activitysim.runner import _cleanup_activitysim_compile_artifacts
from pilates.activitysim.outputs import (
    ActivitySimCompileOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
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

    def run(
        self,
        inputs: ActivitySimPreprocessOutputs,
        workspace: _DummyWorkspace,
    ) -> ActivitySimCompileOutputs:
        assert isinstance(inputs, ActivitySimPreprocessOutputs)
        return ActivitySimCompileOutputs(
            zarr_skims=self._zarr_path,
            sharrow_cache_dir=self._cache_path,
        )


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
    captured: dict = {}

    class _RunRunner:
        def run(self, inputs, workspace, *, extra_inputs=None):
            captured["inputs"] = inputs
            captured["extra_inputs"] = extra_inputs
            return ActivitySimRunOutputs(output_dir=asim_output_dir, raw_outputs={})

    run_runner = _RunRunner()

    def _get_runner(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "activitysim_compile":
            return compile_runner
        if model_name == "activitysim":
            return run_runner
        raise AssertionError(f"Unexpected model_name: {model_name}")

    monkeypatch.setattr(activitysim_steps.ModelFactory, "get_runner", _get_runner)

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

    assert captured["inputs"] is outputs_holder.activitysim_preprocess
    extra_inputs = captured["extra_inputs"]
    assert extra_inputs == {ZARR_SKIMS: str(zarr_path)}


def test_activitysim_run_carries_preprocess_and_compile_hash_metadata(
    monkeypatch, tmp_path: Path
) -> None:
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
    state = SimpleNamespace(year=2020, forecast_year=2020, iteration=0)
    coupler = _DummyCoupler()
    coupler.set(
        ZARR_SKIMS,
        SimpleNamespace(path=str(zarr_path), hash="hash_zarr_compile"),
    )
    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_mutable_dir,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
        input_hashes={
            ASIM_LAND_USE_IN: "hash_land_use",
            ASIM_HOUSEHOLDS_IN: "hash_households",
            ASIM_PERSONS_IN: "hash_persons",
        },
    )

    class _RunRunner:
        def run(self, inputs, workspace, *, extra_inputs=None):
            output_path = workspace.get_asim_output_dir() + "/households.parquet"
            Path(output_path).write_text("out")
            return ActivitySimRunOutputs(
                output_dir=Path(workspace.get_asim_output_dir()),
                raw_outputs={
                    "households_asim_out_temp": Path(output_path),
                },
            )

    def _get_runner(self, model_name, state=None, *_args, **_kwargs):
        if model_name != "activitysim":
            raise AssertionError(f"Unexpected model_name: {model_name}")
        return _RunRunner()

    monkeypatch.setattr(activitysim_steps.ModelFactory, "get_runner", _get_runner)
    monkeypatch.setattr(
        activitysim_steps.cr,
        "log_output",
        lambda *args, **kwargs: SimpleNamespace(hash="hash_households_out"),
    )

    run_step = activitysim_steps.make_activitysim_run_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )
    run_step(settings=SimpleNamespace(), state=state, workspace=workspace)

    run_outputs = outputs_holder.activitysim_run
    assert isinstance(run_outputs, ActivitySimRunOutputs)
    assert run_outputs.raw_output_hashes["households_asim_out_temp"] == "hash_households_out"
    assert run_outputs.source_input_paths[ASIM_LAND_USE_IN] == land_use
    assert run_outputs.source_input_paths[ASIM_HOUSEHOLDS_IN] == households
    assert run_outputs.source_input_paths[ASIM_PERSONS_IN] == persons
    assert run_outputs.source_input_paths[ZARR_SKIMS] == zarr_path
    assert run_outputs.source_input_hashes == {
        ASIM_LAND_USE_IN: "hash_land_use",
        ASIM_HOUSEHOLDS_IN: "hash_households",
        ASIM_PERSONS_IN: "hash_persons",
        ZARR_SKIMS: "hash_zarr_compile",
    }


def test_activitysim_compile_passes_typed_preprocess_outputs_to_runner(
    monkeypatch, tmp_path: Path
) -> None:
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
    omx_skims = asim_mutable_dir / "skims.omx"
    for path in (land_use, households, persons, omx_skims):
        path.write_text("dummy")

    workspace = _DummyWorkspace(tmp_path, asim_output_dir)
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, forecast_year=2020, iteration=0)
    coupler = _DummyCoupler()
    outputs_holder = StepOutputsHolder()
    preprocess_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_mutable_dir,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
        omx_skims=omx_skims,
    )
    outputs_holder.activitysim_preprocess = preprocess_outputs

    captured: dict = {}

    class _CaptureCompileRunner:
        def run(
            self,
            inputs: ActivitySimPreprocessOutputs,
            workspace: _DummyWorkspace,
        ) -> ActivitySimCompileOutputs:
            captured["inputs"] = inputs
            return ActivitySimCompileOutputs(zarr_skims=zarr_path)

    def _get_runner(self, model_name, state=None, *_args, **_kwargs):
        if model_name != "activitysim_compile":
            raise AssertionError(f"Unexpected model_name: {model_name}")
        return _CaptureCompileRunner()

    monkeypatch.setattr(activitysim_steps.ModelFactory, "get_runner", _get_runner)

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

    assert captured["inputs"] is preprocess_outputs
    assert captured["inputs"].omx_skims == omx_skims


def test_activitysim_compile_publishes_expected_zarr_path_without_runner_record(
    monkeypatch, tmp_path: Path
) -> None:
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
    omx_skims = asim_mutable_dir / "skims.omx"
    for path in (land_use, households, persons, omx_skims):
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
        omx_skims=omx_skims,
    )

    class _NoRecordCompileRunner:
        def run(
            self,
            inputs: ActivitySimPreprocessOutputs,
            workspace: _DummyWorkspace,
        ) -> ActivitySimCompileOutputs:
            assert isinstance(inputs, ActivitySimPreprocessOutputs)
            return ActivitySimCompileOutputs()

    def _get_runner(self, model_name, state=None, *_args, **_kwargs):
        if model_name != "activitysim_compile":
            raise AssertionError(f"Unexpected model_name: {model_name}")
        return _NoRecordCompileRunner()

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
        expected_outputs={ZARR_SKIMS: str(zarr_path)},
    )

    assert logged_outputs[ZARR_SKIMS] == str(zarr_path)


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


def test_activitysim_compile_cleanup_removes_stale_retry_artifacts(tmp_path: Path) -> None:
    asim_output_dir = tmp_path / "activitysim" / "output"
    stale_zarr = asim_output_dir / "cache" / "skims.zarr"
    stale_output_numba = asim_output_dir / "cache" / "numba"
    stale_shared_numba = tmp_path / "shared_cache" / "numba"

    stale_zarr.mkdir(parents=True)
    (stale_zarr / ".zgroup").write_text("{}")
    stale_output_numba.mkdir(parents=True)
    (stale_output_numba / "cache.bin").write_text("stale")
    stale_shared_numba.mkdir(parents=True)
    (stale_shared_numba / "entry.bin").write_text("stale")

    workspace = _DummyWorkspace(tmp_path, asim_output_dir)

    _cleanup_activitysim_compile_artifacts(workspace)

    assert not stale_zarr.exists()
    assert not stale_output_numba.exists()
    assert not stale_shared_numba.exists()


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

    def _get_runner(self, model_name, state=None, *_args, **_kwargs):
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

    def _get_runner(self, model_name, state=None, *_args, **_kwargs):
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


def test_activitysim_run_raises_when_container_execution_fails(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    asim_output_dir = root / "activitysim" / "output"
    asim_output_dir.mkdir(parents=True)
    asim_data_dir = root / "activitysim" / "data"
    asim_data_dir.mkdir(parents=True)
    for name in ("land_use.csv", "households.csv", "persons.csv"):
        (asim_data_dir / name).write_text("dummy")
    asim_configs_dir = root / "activitysim" / "configs" / "configs_extended"
    asim_configs_dir.mkdir(parents=True)
    (root / "activitysim" / "configs" / "configs_mp").mkdir(parents=True)
    (root / "activitysim" / "configs" / "configs_sh_compile").mkdir(parents=True)
    zarr_dir = asim_output_dir / "cache" / "skims.zarr"
    zarr_dir.mkdir(parents=True)

    workspace = SimpleNamespace(
        full_path=str(root),
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_asim_mutable_data_dir=lambda: str(asim_data_dir),
        get_asim_mutable_configs_dir=lambda: str(root / "activitysim" / "configs"),
    )
    state = SimpleNamespace(
        current_year=2023,
        current_inner_iter=0,
        forecast_year=2029,
        set_sub_stage_progress=lambda _value: None,
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        activitysim=SimpleNamespace(
            local_output_folder="activitysim/output",
            local_mutable_data_folder="activitysim/data",
            local_mutable_configs_folder="activitysim/configs",
            main_configs_dir="configs_extended",
            household_sample_size=0,
            num_processes=1,
            chunk_size=0,
            sharrow=False,
            file_format="parquet",
            region_mappings={"region_to_subdir": {"test": "prototype"}},
        ),
        infrastructure=SimpleNamespace(container_manager="singularity"),
    )
    state.full_settings = settings

    runner = ActivitysimRunner("activitysim", state)

    monkeypatch.setattr(runner, "get_model_and_image", lambda *_args, **_kwargs: ("activitysim", "docker://fake"))
    monkeypatch.setattr(runner, "get_asim_docker_vols", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "get_base_asim_cmd", lambda *_args, **_kwargs: "asim run")
    monkeypatch.setattr(
        runner,
        "get_asim_additional_args",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(runner, "run_container", lambda **_kwargs: False)

    preprocess_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_data_dir,
        land_use_table=asim_data_dir / "land_use.csv",
        households_table=asim_data_dir / "households.csv",
        persons_table=asim_data_dir / "persons.csv",
    )
    with pytest.raises(RuntimeError, match="ASIM run failed for year 2023 iteration 0"):
        runner.run(preprocess_outputs, workspace)


def test_activitysim_run_stages_external_zarr_input_into_runtime_cache(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "run"
    asim_output_dir = root / "activitysim" / "output"
    asim_output_dir.mkdir(parents=True)
    asim_data_dir = root / "activitysim" / "data"
    asim_data_dir.mkdir(parents=True)
    for name in ("land_use.csv", "households.csv", "persons.csv"):
        (asim_data_dir / name).write_text("dummy")

    external_zarr = tmp_path / "external" / "skims.zarr"
    external_zarr.mkdir(parents=True)
    (external_zarr / ".zarray").write_text("{}")

    workspace = _DummyWorkspace(root, asim_output_dir)
    state = SimpleNamespace(
        current_year=2023,
        current_inner_iter=0,
        forecast_year=2029,
        set_sub_stage_progress=lambda _value: None,
    )
    settings = SimpleNamespace()
    state.full_settings = settings
    runner = ActivitysimRunner("activitysim", state)

    preprocess_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_data_dir,
        land_use_table=asim_data_dir / "land_use.csv",
        households_table=asim_data_dir / "households.csv",
        persons_table=asim_data_dir / "persons.csv",
    )

    captured = {}

    def _fake_run(inputs, _workspace, *, extra_inputs=None):
        captured["inputs"] = inputs
        captured["extra_inputs"] = dict(extra_inputs or {})
        return ActivitySimRunOutputs(output_dir=asim_output_dir)

    monkeypatch.setattr(runner, "_run", _fake_run)

    runner.run(
        preprocess_outputs,
        workspace,
        extra_inputs={ZARR_SKIMS: str(external_zarr)},
    )

    staged_zarr = asim_output_dir / "cache" / "skims.zarr"
    assert staged_zarr.is_dir()
    assert (staged_zarr / ".zarray").read_text() == "{}"
    assert captured["inputs"] is preprocess_outputs
    assert captured["extra_inputs"][ZARR_SKIMS] == str(staged_zarr)
