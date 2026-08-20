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
        (False, True, 2, False, "ActivitySim Numba warmup: running", 1),
        (
            True,
            True,
            2,
            False,
            "ActivitySim Numba warmup: skipped (explicit rewind skip)",
            0,
        ),
        (
            False,
            False,
            2,
            False,
            "ActivitySim Numba warmup: skipped (persistent cache disabled)",
            0,
        ),
        (
            False,
            False,
            1,
            True,
            "ActivitySim Numba warmup: skipped (persistent cache disabled)",
            0,
        ),
        (
            False,
            True,
            1,
            False,
            "ActivitySim Numba warmup: skipped (single-process run)",
            0,
        ),
        (
            False,
            True,
            1,
            True,
            "ActivitySim Numba warmup: skipped (single-process run)",
            0,
        ),
        (
            False,
            True,
            2,
            True,
            "ActivitySim Numba warmup: skipped (node-local cache present)",
            0,
        ),
        (
            True,
            False,
            1,
            True,
            "ActivitySim Numba warmup: skipped (explicit rewind skip)",
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

    def warm_cache(
        _warmup: object,
        _inputs: ActivitySimPreprocessOutputs,
        _workspace: _Workspace,
        **_kwargs: object,
    ) -> None:
        warmup_calls.append(None)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "numba.nbi").touch()

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

    runner.run(
        inputs,
        _launch_context(tmp_path),
        skim_mode="omx",
        skip_numba_warmup=skip_numba_warmup,
        workspace=workspace,
    )

    assert expected_decision in caplog.text
    assert len(warmup_calls) == expected_warmup_calls


def test_numba_warmup_writes_only_to_isolated_compile_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compile container write must not reach production's output mount."""
    workspace = _Workspace(tmp_path)
    production_output = Path(workspace.get_asim_output_dir())
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
            return os.path.commonpath(
                [os.path.abspath(value), str(production_output)]
            ) == str(production_output)
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
    assert (mounted_output / "compile-only.txt").read_text(encoding="utf-8") == "sample"
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (production_output / "compile-only.txt").exists()
    assert attempted_production_mutations == []


def test_numba_warmup_copies_the_selected_zarr_into_private_compile_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warmup must not mount the selected Zarr into the writable container."""
    workspace = _Workspace(tmp_path)
    zarr_parent = tmp_path / "selected-zarr"
    zarr_path = zarr_parent / "skims.zarr"
    zarr_path.mkdir(parents=True)
    (zarr_path / ".zgroup").write_text("{}\n", encoding="utf-8")
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
        _launch_context(tmp_path),
        skim_mode="zarr",
        zarr_input_path=str(zarr_path),
    )

    compile_zarr = tmp_path / "activitysim" / "compile-output" / "cache" / "skims.zarr"
    assert str(zarr_path) not in captured_volumes
    assert (compile_zarr / ".zgroup").read_text(encoding="utf-8") == "{}\n"
