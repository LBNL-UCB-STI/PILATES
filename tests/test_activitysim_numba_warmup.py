from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim import runner as activitysim_runner
from pilates.activitysim.outputs import (
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.runner import ActivitysimRunner
from pilates.activitysim.runner import ActivitySimLaunchContext
from pilates.workflows.artifact_keys import ZARR_SKIMS


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.full_path = str(root)

    def get_asim_output_dir(self) -> str:
        return str(Path(self.full_path) / "activitysim" / "output")

    def get_asim_mutable_data_dir(self) -> str:
        return str(Path(self.full_path) / "activitysim" / "data")

    def get_asim_mutable_configs_dir(self) -> str:
        return str(Path(self.full_path) / "activitysim" / "configs")

    def get_asim_runtime_cache_dir(self) -> str:
        return str(Path(self.get_asim_output_dir()) / "cache")


def _launch_context(root: Path) -> ActivitySimLaunchContext:
    output_dir = root / "activitysim" / "output"
    runtime_cache_dir = output_dir / "cache"
    return ActivitySimLaunchContext(
        workspace_root=root,
        mutable_data_dir=root / "activitysim" / "data",
        output_dir=output_dir,
        compile_output_dir=root / "activitysim" / "compile-output",
        mutable_configs_dir=root / "activitysim" / "configs",
        runtime_cache_dir=runtime_cache_dir,
        runtime_zarr_path=runtime_cache_dir / "skims.zarr",
        shared_cache_dir=root / "shared_cache",
        shared_tmp_dir=root / "tmp",
    )


def _write_structurally_valid_zarr(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")


def test_activitysim_run_outputs_round_trip_generated_zarr(tmp_path: Path) -> None:
    """OMX-mode primary output persists Zarr as a first-class record."""
    workspace = _Workspace(tmp_path)
    zarr_path = tmp_path / "activitysim" / "output" / "cache" / "skims.zarr"
    outputs = ActivitySimRunOutputs(
        output_dir=Path(workspace.get_asim_output_dir()),
        zarr_skims=zarr_path,
    )

    records = outputs.to_record_store()
    recovered = ActivitySimRunOutputs.from_record_store(records, workspace)

    assert recovered.zarr_skims == zarr_path
    assert [record.short_name for record in records.all_records()] == [ZARR_SKIMS]


def _activitysim_inputs(tmp_path: Path) -> ActivitySimPreprocessOutputs:
    return ActivitySimPreprocessOutputs(
        mutable_data_dir=tmp_path / "mutable_data",
        land_use_table=tmp_path / "land_use.csv",
        households_table=tmp_path / "households.csv",
        persons_table=tmp_path / "persons.csv",
    )


def _runner_state(*, persist_cache: bool, num_processes: int) -> SimpleNamespace:
    return SimpleNamespace(
        full_settings=SimpleNamespace(
            activitysim=SimpleNamespace(
                persist_sharrow_cache=persist_cache,
                num_processes=num_processes,
            )
        ),
        set_sub_stage_progress=lambda _stage: None,
    )


@pytest.mark.parametrize(
    (
        "skip_numba_warmup",
        "persist_cache",
        "num_processes",
        "cache_present",
        "expected_decision",
        "expected_warmup_calls",
    ),
    [
        (False, True, 2, False, "ActivitySim Numba warmup: new private epoch", 1),
        (
            True,
            True,
            2,
            False,
            "ActivitySim Numba warmup: not required (explicit rewind skip)",
            0,
        ),
        (
            False,
            False,
            2,
            False,
            "ActivitySim Numba warmup: not required (persistent cache disabled)",
            0,
        ),
        (
            False,
            False,
            1,
            True,
            "ActivitySim Numba warmup: not required (persistent cache disabled)",
            0,
        ),
        (
            False,
            True,
            1,
            False,
            "ActivitySim Numba warmup: not required (single-process run)",
            0,
        ),
        (
            False,
            True,
            1,
            True,
            "ActivitySim Numba warmup: not required (single-process run)",
            0,
        ),
        (
            False,
            True,
            2,
            True,
            "ActivitySim Numba warmup: new private epoch",
            1,
        ),
        (
            True,
            False,
            1,
            True,
            "ActivitySim Numba warmup: not required (explicit rewind skip)",
            0,
        ),
    ],
)
def test_activitysim_runner_logs_one_numba_warmup_decision_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    *,
    skip_numba_warmup: bool,
    persist_cache: bool,
    num_processes: int,
    cache_present: bool,
    expected_decision: str,
    expected_warmup_calls: int,
) -> None:
    workspace = _Workspace(tmp_path)
    cache_dir = tmp_path / "shared_cache" / "numba"
    if cache_present:
        cache_dir.mkdir(parents=True)
        (cache_dir / "numba.nbi").touch()

    runner = ActivitysimRunner(
        "activitysim",
        _runner_state(persist_cache=persist_cache, num_processes=num_processes),
    )
    inputs = _activitysim_inputs(tmp_path)
    warmup_calls: list[None] = []

    warmed_contexts: list[ActivitySimLaunchContext] = []

    def warm_cache(
        _warmup: object,
        _inputs: ActivitySimPreprocessOutputs,
        warmup_context: ActivitySimLaunchContext,
        **_kwargs: object,
    ) -> None:
        warmup_calls.append(None)
        warmed_contexts.append(warmup_context)
        epoch_numba_dir = warmup_context.shared_cache_dir / "numba"
        epoch_numba_dir.mkdir(parents=True, exist_ok=True)
        (epoch_numba_dir / "numba.nbi").touch()
        _write_structurally_valid_zarr(warmup_context.runtime_zarr_path)

    def run_after_decision(*_args: object, **_kwargs: object) -> ActivitySimRunOutputs:
        decisions = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("ActivitySim Numba warmup:")
        ]
        assert decisions == [expected_decision]
        assert len(warmup_calls) == expected_warmup_calls
        return ActivitySimRunOutputs(output_dir=tmp_path / "activitysim" / "output")

    monkeypatch.setattr(activitysim_runner.ActivitysimNumbaWarmup, "run", warm_cache)
    monkeypatch.setattr(runner, "_run", run_after_decision)
    monkeypatch.setattr(
        activitysim_runner,
        "finalize_activitysim_zarr_skims",
        lambda path, *_args: Path(path),
    )
    caplog.set_level(logging.INFO, logger="pilates.activitysim.runner")

    launch_context = _launch_context(tmp_path)
    runner.run(
        inputs,
        launch_context,
        skim_mode="omx",
        skip_numba_warmup=skip_numba_warmup,
        workspace=workspace,
    )

    assert expected_decision in caplog.text
    assert len(warmup_calls) == expected_warmup_calls
    if expected_warmup_calls:
        assert warmed_contexts[0].shared_cache_dir != launch_context.shared_cache_dir


def test_activitysim_runner_reuses_private_compile_epoch_only_for_same_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stale workspace cache cannot replace this process's first preparation."""
    workspace = _Workspace(tmp_path)
    ambient_numba_dir = tmp_path / "shared_cache" / "numba"
    ambient_numba_dir.mkdir(parents=True)
    (ambient_numba_dir / "stale.nbi").touch()
    inputs = _activitysim_inputs(tmp_path)
    launch_context = _launch_context(tmp_path)
    state = _runner_state(persist_cache=True, num_processes=2)
    warmup_contexts: list[ActivitySimLaunchContext] = []
    warmup_runtime_zarr_exists: list[bool] = []
    body_contexts: list[ActivitySimLaunchContext] = []
    body_skim_modes: list[str] = []

    def warm_cache(
        _warmup: object,
        _inputs: ActivitySimPreprocessOutputs,
        warmup_context: ActivitySimLaunchContext,
        **_kwargs: object,
    ) -> None:
        warmup_contexts.append(warmup_context)
        warmup_runtime_zarr_exists.append(warmup_context.runtime_zarr_path.exists())
        epoch_numba_dir = warmup_context.shared_cache_dir / "numba"
        epoch_numba_dir.mkdir(parents=True, exist_ok=True)
        (epoch_numba_dir / "compiled.nbi").touch()
        _write_structurally_valid_zarr(warmup_context.runtime_zarr_path)

    def run_body(
        _runner: ActivitysimRunner,
        _inputs: ActivitySimPreprocessOutputs,
        body_context: ActivitySimLaunchContext,
        **kwargs: object,
    ) -> ActivitySimRunOutputs:
        body_contexts.append(body_context)
        body_skim_modes.append(str(kwargs["skim_mode"]))
        return ActivitySimRunOutputs(output_dir=body_context.output_dir)

    monkeypatch.setattr(activitysim_runner.ActivitysimNumbaWarmup, "run", warm_cache)
    monkeypatch.setattr(ActivitysimRunner, "_run", run_body)
    monkeypatch.setattr(
        activitysim_runner,
        "finalize_activitysim_zarr_skims",
        lambda path, *_args: Path(path),
    )
    caplog.set_level(logging.INFO, logger="pilates.activitysim.runner")

    ActivitysimRunner("activitysim", state).run(
        inputs,
        launch_context,
        skim_mode="omx",
        workspace=workspace,
    )
    ActivitysimRunner("activitysim", state).run(
        inputs,
        launch_context,
        skim_mode="omx",
        workspace=workspace,
    )
    ActivitysimRunner(
        "activitysim", _runner_state(persist_cache=True, num_processes=2)
    ).run(
        inputs,
        launch_context,
        skim_mode="omx",
        workspace=workspace,
    )

    assert len(warmup_contexts) == 3
    assert body_contexts[0].shared_cache_dir == body_contexts[1].shared_cache_dir
    assert body_contexts[0].shared_cache_dir != ambient_numba_dir.parent
    assert body_contexts[0].shared_cache_dir.is_relative_to(ambient_numba_dir)
    assert body_contexts[2].shared_cache_dir != body_contexts[0].shared_cache_dir
    assert warmup_contexts[0].shared_cache_dir == warmup_contexts[1].shared_cache_dir
    assert warmup_contexts[2].shared_cache_dir != warmup_contexts[0].shared_cache_dir
    assert warmup_runtime_zarr_exists == [False, False, False]
    assert body_skim_modes == ["zarr", "zarr", "zarr"]
    decisions = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("ActivitySim Numba warmup:")
    ]
    assert decisions == [
        "ActivitySim Numba warmup: new private epoch",
        "ActivitySim Numba warmup: reused private epoch",
        "ActivitySim Numba warmup: new private epoch",
    ]


def test_activitysim_runner_reuses_compile_cache_without_regenerating_selected_zarr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected Zarr stays input-only when its compile cache is reused."""
    launch_context = _launch_context(tmp_path)
    selected_zarr = tmp_path / "selected" / "skims.zarr"
    _write_structurally_valid_zarr(selected_zarr)
    selected_bytes = (selected_zarr / ".zgroup").read_bytes()
    state = _runner_state(persist_cache=True, num_processes=2)
    warmup_contexts: list[ActivitySimLaunchContext] = []
    body_modes: list[str] = []
    body_zarr_paths: list[str] = []

    def warm_cache(
        _warmup: object,
        _inputs: ActivitySimPreprocessOutputs,
        warmup_context: ActivitySimLaunchContext,
        **kwargs: object,
    ) -> None:
        warmup_contexts.append(warmup_context)
        assert kwargs["skim_mode"] == "zarr"
        epoch_numba_dir = warmup_context.shared_cache_dir / "numba"
        epoch_numba_dir.mkdir(parents=True, exist_ok=True)
        (epoch_numba_dir / "compiled.nbi").touch()

    def run_body(
        _runner: ActivitysimRunner,
        _inputs: ActivitySimPreprocessOutputs,
        _body_context: ActivitySimLaunchContext,
        **kwargs: object,
    ) -> ActivitySimRunOutputs:
        body_modes.append(str(kwargs["skim_mode"]))
        extra_inputs = kwargs["extra_inputs"]
        assert isinstance(extra_inputs, dict)
        body_zarr_paths.append(str(extra_inputs[ZARR_SKIMS]))
        return ActivitySimRunOutputs(output_dir=launch_context.output_dir)

    monkeypatch.setattr(activitysim_runner.ActivitysimNumbaWarmup, "run", warm_cache)
    monkeypatch.setattr(ActivitysimRunner, "_run", run_body)
    monkeypatch.setattr(
        activitysim_runner,
        "finalize_activitysim_zarr_skims",
        lambda *_args: pytest.fail("selected Zarr must not be finalized"),
    )

    first_outputs = ActivitysimRunner("activitysim", state).run(
        _activitysim_inputs(tmp_path),
        launch_context,
        skim_mode="zarr",
        extra_inputs={ZARR_SKIMS: selected_zarr},
    )
    second_outputs = ActivitysimRunner("activitysim", state).run(
        _activitysim_inputs(tmp_path),
        launch_context,
        skim_mode="zarr",
        extra_inputs={ZARR_SKIMS: selected_zarr},
    )

    assert len(warmup_contexts) == 1
    assert body_modes == ["zarr", "zarr"]
    assert body_zarr_paths == [
        str(launch_context.runtime_zarr_path),
        str(launch_context.runtime_zarr_path),
    ]
    assert (selected_zarr / ".zgroup").read_bytes() == selected_bytes
    assert first_outputs.zarr_skims is None
    assert second_outputs.zarr_skims is None


def test_activitysim_runner_aborts_the_body_when_private_preparation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed compile cannot fall back to stale cache bytes before the body."""
    workspace = _Workspace(tmp_path)
    ambient_numba_dir = tmp_path / "shared_cache" / "numba"
    ambient_numba_dir.mkdir(parents=True)
    (ambient_numba_dir / "stale.nbi").touch()
    runner = ActivitysimRunner(
        "activitysim", _runner_state(persist_cache=True, num_processes=2)
    )
    body_calls: list[None] = []

    def fail_warmup(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("compile failed")

    def fail_if_body_runs(*_args: object, **_kwargs: object) -> ActivitySimRunOutputs:
        body_calls.append(None)
        raise AssertionError("the ActivitySim body must not follow a failed compile")

    monkeypatch.setattr(activitysim_runner.ActivitysimNumbaWarmup, "run", fail_warmup)
    monkeypatch.setattr(runner, "_run", fail_if_body_runs)

    with pytest.raises(RuntimeError, match="compile failed"):
        runner.run(
            _activitysim_inputs(tmp_path),
            _launch_context(tmp_path),
            skim_mode="omx",
            workspace=workspace,
        )

    assert body_calls == []


def test_numba_warmup_writes_only_to_isolated_compile_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compile container write must not reach production's output mount."""
    workspace = _Workspace(tmp_path)
    production_output = Path(workspace.get_asim_output_dir())
    production_runtime_cache = production_output / "cache"
    production_output.mkdir(parents=True)
    sentinel = production_output / "production-only.txt"
    sentinel.write_text("keep", encoding="utf-8")
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        activitysim=SimpleNamespace(
            region_mappings={"region_to_subdir": {"seattle": "example"}},
            local_mutable_data_folder="activitysim/data",
            local_output_folder="activitysim/output",
            local_mutable_configs_folder="activitysim/configs",
            main_configs_dir="configs",
            file_format="parquet",
            persist_sharrow_cache=True,
            command_template="activitysim",
            household_sample_size=0,
            num_processes=2,
            chunk_size=0,
        ),
    )
    warmup = activitysim_runner.ActivitysimNumbaWarmup(
        "activitysim_numba_warmup", SimpleNamespace(full_settings=settings)
    )
    captured_volumes: dict[str, dict[str, str]] = {}
    attempted_production_mutations: list[str] = []

    def is_production_path(value: object) -> bool:
        if not isinstance(value, (str, Path)):
            return False
        try:
            value_path = os.path.abspath(value)
            return (
                os.path.commonpath([value_path, str(production_output)])
                == str(production_output)
                and os.path.commonpath([value_path, str(production_runtime_cache)])
                != str(production_runtime_cache)
            )
        except ValueError:
            return False

    def guard_mutation(operation: str):
        original = getattr(activitysim_runner.os, operation)

        def guarded(*args: object, **kwargs: object):
            if any(is_production_path(value) for value in args):
                attempted_production_mutations.append(operation)
                raise AssertionError(
                    f"warmup attempted {operation} under production output"
                )
            return original(*args, **kwargs)

        return guarded

    for operation in (
        "mkdir",
        "makedirs",
        "remove",
        "rename",
        "replace",
        "unlink",
        "utime",
    ):
        monkeypatch.setattr(activitysim_runner.os, operation, guard_mutation(operation))

    monkeypatch.setattr(
        warmup,
        "get_model_and_image",
        lambda *_args: ("activity_demand_model", "image"),
    )

    def run_compile_container(*_args: object, **kwargs: object) -> bool:
        captured_volumes.update(kwargs["volumes"])
        output_root = next(
            Path(local)
            for local, mount in kwargs["volumes"].items()
            if mount["bind"].endswith("/output")
        )
        assert not is_production_path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "compile-only.txt").write_text("sample", encoding="utf-8")
        return True

    monkeypatch.setattr(warmup, "run_container", run_compile_container)

    warmup.run(_activitysim_inputs(tmp_path), _launch_context(tmp_path))

    mounted_output = next(
        Path(local)
        for local, mount in captured_volumes.items()
        if mount["bind"].endswith("/output")
    )
    assert mounted_output != production_output
    assert captured_volumes[str(production_runtime_cache)]["mode"] == "rw"
    assert (mounted_output / "compile-only.txt").read_text(encoding="utf-8") == "sample"
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (production_output / "compile-only.txt").exists()
    assert attempted_production_mutations == []


def test_numba_warmup_mounts_the_staged_zarr_as_its_read_only_compile_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warmup uses the body Zarr rather than a private copied replacement."""
    workspace = _Workspace(tmp_path)
    launch_context = _launch_context(tmp_path)
    zarr_path = launch_context.runtime_zarr_path
    _write_structurally_valid_zarr(zarr_path)
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        activitysim=SimpleNamespace(
            region_mappings={"region_to_subdir": {"seattle": "example"}},
            local_mutable_data_folder="activitysim/data",
            local_output_folder="activitysim/output",
            local_mutable_configs_folder="activitysim/configs",
            main_configs_dir="configs",
            file_format="parquet",
            persist_sharrow_cache=True,
            command_template="activitysim",
            household_sample_size=0,
            num_processes=2,
            chunk_size=0,
        ),
    )
    warmup = activitysim_runner.ActivitysimNumbaWarmup(
        "activitysim_numba_warmup", SimpleNamespace(full_settings=settings)
    )
    captured_volumes: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(
        warmup,
        "get_model_and_image",
        lambda *_args: ("activity_demand_model", "image"),
    )
    monkeypatch.setattr(
        warmup,
        "run_container",
        lambda *_args, **kwargs: captured_volumes.update(kwargs["volumes"]) or True,
    )

    warmup.run(
        _activitysim_inputs(tmp_path),
        launch_context,
        skim_mode="zarr",
        zarr_input_path=str(zarr_path),
    )

    compile_zarr = tmp_path / "activitysim" / "compile-output" / "cache" / "skims.zarr"
    assert captured_volumes[str(launch_context.runtime_cache_dir)]["mode"] == "ro"
    assert not compile_zarr.exists()
