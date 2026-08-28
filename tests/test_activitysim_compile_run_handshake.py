"""Regression coverage for the private ActivitySim Numba warmup handshake.

The filename is retained for test-discovery continuity.  The public workflow
compile step it formerly covered no longer exists.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from consist import BindingResult, ExecutionOptions, Tracker, define_step

from pilates.activitysim import runner as activitysim_runner
from pilates.activitysim.outputs import (
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.runner import (
    ActivitySimLaunchContext,
    ActivitysimSkimMode,
    ActivitysimNumbaWarmup,
    ActivitysimRunner,
    asim_compile_output_dir,
)
from pilates.utils import consist_runtime as cr
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_definition import InputContract, StepDefinition
from pilates.workflows.step_execution import execute_step


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


def _matrix_settings(*, num_processes: int) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            region="seattle",
            models=SimpleNamespace(
                land_use="urbansim",
                travel="beam",
                activity_demand="activitysim",
                vehicle_ownership=None,
            ),
        ),
        infrastructure=SimpleNamespace(
            container_manager="docker",
            docker_images={"activitysim": "activitysim-image"},
        ),
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
            num_processes=num_processes,
            chunk_size=0,
        ),
    )


def _matrix_inputs(tmp_path: Path) -> ActivitySimPreprocessOutputs:
    return ActivitySimPreprocessOutputs(
        mutable_data_dir=tmp_path / "mutable-data",
        land_use_table=tmp_path / "land_use.csv",
        households_table=tmp_path / "households.csv",
        persons_table=tmp_path / "persons.csv",
    )


def _write_generated_zarr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"time": (("otaz", "dtaz"), np.array([[1.0, 2.0], [2.0, 1.0]]))},
        coords={"otaz": [0, 1], "dtaz": [0, 1]},
    ).to_zarr(path, mode="w", consolidated=True, zarr_format=2)


@pytest.mark.parametrize(
    ("selected_mode", "preparation_required"),
    [
        ("zarr", True),
        ("zarr", False),
        ("omx", True),
        ("omx", False),
    ],
)
def test_activitysim_prepare_body_handshake_preserves_selected_skim_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selected_mode: ActivitysimSkimMode,
    preparation_required: bool,
) -> None:
    """The body consumes selected Zarrs or a finalized warmup-generated Zarr."""
    launch_context = _launch_context(tmp_path)
    runtime_zarr = launch_context.runtime_zarr_path
    selected_zarr = tmp_path / "selected" / "skims.zarr"
    selected_zarr_bytes = b'{"zarr_format": 2}\n'
    if selected_mode == "zarr":
        selected_zarr.mkdir(parents=True)
        (selected_zarr / ".zgroup").write_bytes(selected_zarr_bytes)

    state = SimpleNamespace(
        full_settings=_matrix_settings(
            num_processes=2 if preparation_required else 1
        ),
        current_year=2018,
        current_inner_iter=0,
        set_sub_stage_progress=lambda _progress: None,
    )
    runner = ActivitysimRunner("activitysim", state)
    calls: list[str] = []

    def run_container(*_args: object, **kwargs: object) -> bool:
        model_name = kwargs["model_name"]
        volumes = kwargs["volumes"]
        assert isinstance(model_name, str)
        assert isinstance(volumes, dict)
        if model_name == "activitysim_numba_warmup":
            calls.append("warmup")
            cache_mount = next(
                Path(local)
                for local, mount in volumes.items()
                if mount["bind"] == "/app/numba_cache"
            )
            (cache_mount / "numba").mkdir(parents=True, exist_ok=True)
            (cache_mount / "numba" / "compiled.nbi").write_text(
                "cache", encoding="utf-8"
            )
            assert str(launch_context.runtime_cache_dir) in volumes
            expected_mode = "ro" if selected_mode == "zarr" else "rw"
            assert volumes[str(launch_context.runtime_cache_dir)]["mode"] == expected_mode
            if selected_mode == "zarr":
                assert (runtime_zarr / ".zgroup").read_bytes() == selected_zarr_bytes
            else:
                _write_generated_zarr(runtime_zarr)
            return True

        assert model_name == "activitysim"
        calls.append("body")
        if selected_mode == "zarr" or preparation_required:
            assert str(launch_context.runtime_cache_dir) in volumes
            assert volumes[str(launch_context.runtime_cache_dir)]["mode"] == "ro"
            assert runtime_zarr.exists()
        else:
            assert str(launch_context.runtime_cache_dir) not in volumes
            assert not runtime_zarr.exists()
            _write_generated_zarr(runtime_zarr)
        return True

    monkeypatch.setattr(
        activitysim_runner.GenericRunner, "run_container", staticmethod(run_container)
    )

    outputs = runner.run(
        _matrix_inputs(tmp_path),
        launch_context,
        skim_mode=selected_mode,
        extra_inputs=(
            {"zarr_skims": selected_zarr} if selected_mode == "zarr" else None
        ),
        workspace=SimpleNamespace(full_path=str(tmp_path)),
    )

    expected_calls = ["warmup", "body"] if preparation_required else ["body"]
    assert calls == expected_calls
    if selected_mode == "zarr":
        assert (runtime_zarr / ".zgroup").read_bytes() == selected_zarr_bytes
        assert outputs.zarr_skims is None
    else:
        assert outputs.zarr_skims == runtime_zarr
        with xr.open_zarr(runtime_zarr) as skims:
            assert skims["otaz"].attrs["preprocessed"] == "zero-based-contiguous"


def test_numba_warmup_is_private_and_never_publishes_its_local_zarr(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_output_dir=lambda: str(tmp_path / "activitysim" / "output"),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        activitysim=SimpleNamespace(
            region_mappings={"region_to_subdir": {"seattle": "example"}},
            local_output_folder="activitysim/output",
        ),
    )
    state = SimpleNamespace(full_settings=settings)
    warmup = ActivitysimNumbaWarmup("activitysim_numba_warmup", state)
    production_zarr_path = tmp_path / "activitysim" / "output" / "cache" / "skims.zarr"
    compile_output = Path(asim_compile_output_dir(workspace))

    monkeypatch.setattr(warmup, "get_asim_docker_vols", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        warmup, "get_model_and_image", lambda *_args: ("activitysim", "image")
    )
    monkeypatch.setattr(
        warmup, "get_base_asim_cmd", lambda *_args, **_kwargs: ["activitysim"]
    )
    monkeypatch.setattr(warmup, "get_asim_additional_args", lambda *_args: [])
    monkeypatch.setattr(
        "pilates.activitysim.runner._log_activitysim_launch_context",
        lambda **_kwargs: None,
    )

    def _run_container(**kwargs):
        assert kwargs["model_name"] == "activitysim_numba_warmup"
        assert kwargs["lineage_mode"] == "none"
        assert kwargs["output_paths"] == [str(compile_output)]
        (compile_output / "cache" / "skims.zarr").mkdir(parents=True)
        return True

    monkeypatch.setattr(warmup, "run_container", _run_container)
    result = warmup.run(
        ActivitySimPreprocessOutputs(
            mutable_data_dir=tmp_path,
            land_use_table=tmp_path / "land_use.csv",
            households_table=tmp_path / "households.csv",
            persons_table=tmp_path / "persons.csv",
        ),
        _launch_context(tmp_path),
    )

    assert result is None
    assert not production_zarr_path.exists()
    assert not hasattr(warmup, "expected_outputs")


def test_execute_step_cache_hit_skips_activitysim_runner_preparation_and_container(
    monkeypatch, tmp_path: Path
) -> None:
    """A native cache hit hydrates without entering the ActivitySim body."""
    workspace = SimpleNamespace(full_path=str(tmp_path / "workspace"))
    settings = SimpleNamespace(
        activitysim=SimpleNamespace(persist_sharrow_cache=True, num_processes=2)
    )
    state = SimpleNamespace(
        full_settings=settings,
        set_sub_stage_progress=lambda _progress: None,
    )
    runner = ActivitysimRunner("activitysim", state)
    inputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=tmp_path,
        land_use_table=tmp_path / "land_use.csv",
        households_table=tmp_path / "households.csv",
        persons_table=tmp_path / "persons.csv",
    )
    runner_body_calls: list[None] = []
    warmup_calls: list[None] = []
    warmup_contexts: list[ActivitySimLaunchContext] = []
    runner_run_calls: list[None] = []
    container_calls: list[None] = []

    def warm_cache(
        _warmup: object,
        _inputs: ActivitySimPreprocessOutputs,
        warmup_context: ActivitySimLaunchContext,
        **_kwargs: object,
    ) -> None:
        warmup_calls.append(None)
        warmup_contexts.append(warmup_context)
        cache_dir = warmup_context.shared_cache_dir / "numba"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "numba.nbi").touch()
        runtime_zarr = warmup_context.runtime_zarr_path
        runtime_zarr.mkdir(parents=True, exist_ok=True)
        (runtime_zarr / ".zgroup").write_text(
            '{"zarr_format": 2}\n', encoding="utf-8"
        )

    def run_production(*_args, **_kwargs) -> ActivitySimRunOutputs:
        runner_body_calls.append(None)
        return ActivitySimRunOutputs(output_dir=tmp_path / "workspace" / "output")

    def fail_if_container_runs(*_args: object, **_kwargs: object) -> bool:
        container_calls.append(None)
        raise AssertionError("the test runner body must not start a container")

    monkeypatch.setattr(ActivitysimNumbaWarmup, "run", warm_cache)
    monkeypatch.setattr(runner, "_run", run_production)
    monkeypatch.setattr(runner, "run_container", fail_if_container_runs)
    monkeypatch.setattr(
        activitysim_runner,
        "finalize_activitysim_zarr_skims",
        lambda path, *_args: Path(path),
    )

    @define_step(
        model="activitysim_cache_admission_probe",
        inputs={"source": None},
        output_paths={"result": "result.txt"},
        input_binding="paths",
    )
    def native_activitysim_body(
        source: Path,
        ctx,
        *,
        settings: object,
        state: object,
        workspace: object,
    ) -> None:
        del source
        assert settings is not None
        assert state is not None
        assert workspace is not None
        runner_run_calls.append(None)
        runner.run(
            inputs,
            _launch_context(tmp_path / "workspace"),
            skim_mode="omx",
            workspace=workspace,
        )
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "result.txt").write_text("complete\n", encoding="utf-8")

    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    definition = StepDefinition(
        name="activitysim_cache_admission_probe",
        function=native_activitysim_body,
        resolve_inputs=lambda **_: ResolvedStepInputs(
            step_name="activitysim_cache_admission_probe",
            binding=BindingResult(inputs={"source": source}),
        ),
        project_outputs=lambda outputs, **_: outputs,
        execution_options=lambda **_: ExecutionOptions(
            input_binding="paths", inject_context="ctx"
        ),
        input_contract=InputContract(
            status="incomplete",
            reason="test-only cache admission probe",
        ),
        archive_outputs=False,
    )
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )
    results = []
    for name in ("activitysim-first", "activitysim-second"):
        with tracker.scenario(name) as scenario:
            result, _ = execute_step(
                scenario=scenario,
                definition=definition,
                settings=settings,
                state=state,
                workspace=workspace,
                stage="activitysim",
                year=None,
                iteration=None,
                phase=None,
            )
            results.append(result)

    assert results[0].cache_hit is False
    assert results[1].cache_hit is True
    assert runner_run_calls == [None]
    assert runner_body_calls == [None]
    assert warmup_calls == [None]
    assert container_calls == []
    epoch_path = warmup_contexts[0].shared_cache_dir
    resolved_inputs = definition.resolve_inputs()
    assert all(
        str(epoch_path) not in str(value)
        for value in (resolved_inputs.binding.inputs or {}).values()
    )
    assert str(epoch_path) not in str(results[0].outputs)
    assert str(epoch_path) not in str(definition.execution_options())


def test_numba_warmup_passes_the_staged_zarr_to_consist_as_read_only_input(
    monkeypatch, tmp_path: Path
) -> None:
    """The actual Consist call receives the same staged Zarr as the body."""
    workspace = SimpleNamespace(
        full_path=str(tmp_path / "workspace"),
        get_asim_output_dir=lambda: str(
            tmp_path / "workspace" / "activitysim" / "output"
        ),
        get_asim_runtime_cache_dir=lambda: str(
            tmp_path / "workspace" / "activitysim" / "output" / "cache"
        ),
        get_asim_mutable_data_dir=lambda: str(
            tmp_path / "workspace" / "activitysim" / "data"
        ),
        get_asim_mutable_configs_dir=lambda: str(
            tmp_path / "workspace" / "activitysim" / "configs"
        ),
    )
    launch_context = _launch_context(tmp_path / "workspace")
    selected_zarr = launch_context.runtime_zarr_path
    selected_zarr.mkdir(parents=True)
    (selected_zarr / ".zgroup").write_text(
        '{"zarr_format": 2}\n', encoding="utf-8"
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle", use_stubs=False),
        infrastructure=SimpleNamespace(
            container_manager="docker",
            docker_config=SimpleNamespace(pull_latest=False, stdout=False),
        ),
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
    warmup = ActivitysimNumbaWarmup(
        "activitysim_numba_warmup", SimpleNamespace(full_settings=settings)
    )
    called: dict[str, object] = {}
    monkeypatch.setattr(
        warmup,
        "get_model_and_image",
        lambda *_args: ("activity_demand_model", "image"),
    )
    monkeypatch.setattr(
        "consist.integrations.containers.run_container",
        lambda **kwargs: called.update(kwargs) or True,
    )
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    with cr.use_tracker(tracker):
        with tracker.start_run("warmup-container", model="activitysim"):
            warmup.run(
                ActivitySimPreprocessOutputs(
                    mutable_data_dir=tmp_path,
                    land_use_table=tmp_path / "land_use.csv",
                    households_table=tmp_path / "households.csv",
                    persons_table=tmp_path / "persons.csv",
                ),
                launch_context,
                skim_mode="zarr",
                zarr_input_path=str(selected_zarr),
            )

    volumes = called["volumes"]
    assert volumes[str(selected_zarr.parent)].endswith("/output/cache")
    assert volumes[str(asim_compile_output_dir(workspace))].endswith("/output")
    command = called["command"]
    assert command.count("-o") == 1
    assert command[command.index("-o") + 1].endswith("/output")
