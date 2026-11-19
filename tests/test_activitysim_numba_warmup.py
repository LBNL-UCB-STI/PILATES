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
    """OMX-mode primary output emits Zarr as a first-class record item."""
    workspace = _Workspace(tmp_path)
    zarr_path = tmp_path / "activitysim" / "output" / "cache" / "skims.zarr"
    outputs = ActivitySimRunOutputs(
        output_dir=Path(workspace.get_asim_output_dir()),
        zarr_skims=zarr_path,
    )

    items = list(outputs._iter_record_items())
    assert [short_name for short_name, _, _ in items] == [ZARR_SKIMS]
    assert items[0][1] == zarr_path


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
        _activitysim_compile_epoch=None,
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
            return os.path.commonpath([value_path, str(production_output)]) == str(
                production_output
            ) and os.path.commonpath(
                [value_path, str(production_runtime_cache)]
            ) != str(production_runtime_cache)
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


def test_numba_warmup_mounts_the_staged_zarr_as_its_compile_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warmup mounts the staged Zarr in place (read-write for ActivitySim cache .gitignore) rather than copying."""
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
    assert (launch_context.compile_output_dir / "cache").is_dir()
    assert captured_volumes[str(launch_context.runtime_cache_dir)]["mode"] == "rw"
    assert not compile_zarr.exists()


def test_runner_skips_numba_warmup_and_syncs_staged_configs_when_sharrow_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit sharrow=False syncs staged settings.yaml, skips warmup, and omits --persist-sharrow-cache."""
    import yaml
    from pilates.workflows.consist_metadata import build_activitysim_identity_config

    workspace = _Workspace(tmp_path)
    launch_context = _launch_context(tmp_path)
    for dirname in ("configs", "configs_extended", "configs_sh_compile"):
        cfg_dir = launch_context.mutable_configs_dir / dirname
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "settings.yaml").write_text(
            "sharrow: true\nmodels:\n  - initialize_landuse\n",
            encoding="utf-8",
        )
    mp_dir = launch_context.mutable_configs_dir / "configs_mp"
    mp_dir.mkdir(parents=True, exist_ok=True)
    (mp_dir / "settings.yaml").write_text("inherit_settings: true\n", encoding="utf-8")

    state = SimpleNamespace(
        full_settings=SimpleNamespace(
            activitysim=SimpleNamespace(
                sharrow=False,
                persist_sharrow_cache=True,
                num_processes=8,
                file_format="parquet",
                main_configs_dir="configs_extended",
                household_sample_size=100,
                chunk_size=0,
                replan_iters=0,
                replan_hh_samp_size=0,
                replan_after="non_mandatory_tour_scheduling",
                random_seed=42,
            )
        ),
        _activitysim_compile_epoch=None,
        set_sub_stage_progress=lambda _stage: None,
    )
    zarr_path = launch_context.runtime_zarr_path
    _write_structurally_valid_zarr(zarr_path)

    warmup_calls: list[str] = []

    class _FakeWarmup:
        def __init__(self, name: str, _state: object) -> None:
            warmup_calls.append(name)

        def run(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("Warmup should not run when sharrow is disabled")

    monkeypatch.setattr(activitysim_runner, "ActivitysimNumbaWarmup", _FakeWarmup)
    runner = ActivitysimRunner("activitysim", state)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *_args, **_kwargs: ActivitySimRunOutputs(
            output_dir=Path(workspace.get_asim_output_dir())
        ),
    )

    caplog.set_level(logging.INFO, logger=activitysim_runner.__name__)
    runner.run(
        _activitysim_inputs(tmp_path),
        launch_context,
        skim_mode="zarr",
        extra_inputs={ZARR_SKIMS: str(zarr_path)},
        workspace=workspace,
    )

    assert warmup_calls == []
    assert "ActivitySim Numba warmup: not required (sharrow disabled)" in [
        record.getMessage() for record in caplog.records
    ]
    for dirname in ("configs", "configs_extended", "configs_sh_compile"):
        loaded = yaml.safe_load(
            (launch_context.mutable_configs_dir / dirname / "settings.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert loaded["sharrow"] is False
    mp_loaded = yaml.safe_load((mp_dir / "settings.yaml").read_text(encoding="utf-8"))
    assert "sharrow" not in mp_loaded

    args = ActivitysimRunner.get_asim_additional_args(
        state.full_settings,
        {"/host/configs": {"bind": "/app/configs"}},
        compile=False,
    )
    assert "--persist-sharrow-cache" not in args

    identity_cfg = build_activitysim_identity_config(state.full_settings)
    assert identity_cfg["sharrow"] is False


def test_runner_respects_staged_settings_yaml_sharrow_false_when_config_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When activitysim.sharrow is None, staged settings.yaml sharrow: false skips warmup."""
    from pilates.workflows.consist_metadata import build_activitysim_identity_config

    workspace = _Workspace(tmp_path)
    launch_context = _launch_context(tmp_path)
    cfg_dir = launch_context.mutable_configs_dir / "configs_extended"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "settings.yaml").write_text("sharrow: false\n", encoding="utf-8")

    state = _runner_state(persist_cache=True, num_processes=4)
    zarr_path = launch_context.runtime_zarr_path
    _write_structurally_valid_zarr(zarr_path)

    monkeypatch.setattr(
        activitysim_runner,
        "ActivitysimNumbaWarmup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Warmup should not be instantiated")
        ),
    )
    runner = ActivitysimRunner("activitysim", state)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *_args, **_kwargs: ActivitySimRunOutputs(
            output_dir=Path(workspace.get_asim_output_dir())
        ),
    )

    caplog.set_level(logging.INFO, logger=activitysim_runner.__name__)
    runner.run(
        _activitysim_inputs(tmp_path),
        launch_context,
        skim_mode="zarr",
        extra_inputs={ZARR_SKIMS: str(zarr_path)},
        workspace=workspace,
    )

    assert "ActivitySim Numba warmup: not required (sharrow disabled)" in [
        record.getMessage() for record in caplog.records
    ]
    identity_cfg = build_activitysim_identity_config(
        SimpleNamespace(
            activitysim=SimpleNamespace(
                sharrow=None,
                household_sample_size=100,
                chunk_size=0,
                num_processes=4,
                file_format="parquet",
                replan_iters=0,
                replan_hh_samp_size=0,
                replan_after="non_mandatory_tour_scheduling",
                random_seed=42,
            )
        )
    )
    assert "sharrow" not in identity_cfg


def test_omx_mode_clears_stale_runtime_zarr_when_warmup_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When warmup is skipped in OMX mode, any stale runtime skims.zarr is removed before _run."""
    workspace = _Workspace(tmp_path)
    launch_context = _launch_context(tmp_path)
    stale_marker = launch_context.runtime_zarr_path / "stale.txt"
    _write_structurally_valid_zarr(launch_context.runtime_zarr_path)
    stale_marker.write_text("stale", encoding="utf-8")

    state = SimpleNamespace(
        full_settings=SimpleNamespace(
            activitysim=SimpleNamespace(
                sharrow=False,
                persist_sharrow_cache=True,
                num_processes=8,
            )
        ),
        _activitysim_compile_epoch=None,
        set_sub_stage_progress=lambda _stage: None,
    )
    runner = ActivitysimRunner("activitysim", state)
    observed_stale_during_run: list[bool] = []
    finalized_zarr = tmp_path / "finalized" / "skims.zarr"

    def fake_run(
        _inputs: ActivitySimPreprocessOutputs,
        body_context: ActivitySimLaunchContext,
        *,
        skim_mode: str,
        extra_inputs: object = None,
    ) -> ActivitySimRunOutputs:
        del extra_inputs
        assert skim_mode == "omx"
        observed_stale_during_run.append(body_context.runtime_zarr_path.exists())
        _write_structurally_valid_zarr(body_context.runtime_zarr_path)
        return ActivitySimRunOutputs(output_dir=Path(workspace.get_asim_output_dir()))

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(
        activitysim_runner,
        "finalize_activitysim_zarr_skims",
        lambda runtime_path, _settings, _ws: (
            finalized_zarr
            if Path(runtime_path) == launch_context.runtime_zarr_path
            and not stale_marker.exists()
            else (_ for _ in ()).throw(
                AssertionError("Stale runtime zarr was not cleared")
            )
        ),
    )

    outputs = runner.run(
        _activitysim_inputs(tmp_path),
        launch_context,
        skim_mode="omx",
        workspace=workspace,
    )

    assert observed_stale_during_run == [False]
    assert outputs.zarr_skims == finalized_zarr
