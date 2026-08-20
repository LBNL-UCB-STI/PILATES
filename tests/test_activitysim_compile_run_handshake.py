"""Regression coverage for the private ActivitySim Numba warmup handshake.

The filename is retained for test-discovery continuity.  The public workflow
compile step it formerly covered no longer exists.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from consist import BindingResult, ExecutionOptions, Tracker, define_step

from pilates.activitysim import runner as activitysim_runner
from pilates.activitysim.outputs import (
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.runner import (
    ActivitySimLaunchContext,
    ActivitysimNumbaWarmup,
    ActivitysimRunner,
    asim_compile_output_dir,
)
from pilates.utils import consist_runtime as cr


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


def test_cache_hit_skips_activitysim_warmup_with_the_native_runner_body(
    monkeypatch, tmp_path: Path
) -> None:
    """Consist admission skips the whole runner body, including warmup."""
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

    def warm_cache(*_args, **_kwargs) -> None:
        warmup_calls.append(None)
        cache_dir = tmp_path / "workspace" / "shared_cache" / "numba"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "numba.nbi").touch()

    def run_production(*_args, **_kwargs) -> ActivitySimRunOutputs:
        runner_body_calls.append(None)
        return ActivitySimRunOutputs(output_dir=tmp_path / "workspace" / "output")

    monkeypatch.setattr(ActivitysimNumbaWarmup, "run", warm_cache)
    monkeypatch.setattr(runner, "_run", run_production)
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
    def native_activitysim_body(source: Path, ctx) -> None:
        del source
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
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )
    results = []
    for name in ("activitysim-first", "activitysim-second"):
        with tracker.scenario(name) as scenario:
            results.append(
                scenario.run(
                    fn=native_activitysim_body,
                    binding=BindingResult(inputs={"source": source}),
                    execution_options=ExecutionOptions(
                        input_binding="paths", inject_context="ctx"
                    ),
                )
            )

    assert results[0].cache_hit is False
    assert results[1].cache_hit is True
    assert runner_body_calls == [None]
    assert warmup_calls == [None]


def test_numba_warmup_passes_only_private_zarr_bytes_to_consist_container(
    monkeypatch, tmp_path: Path
) -> None:
    """The actual Consist call must receive no mount for the selected Zarr."""
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
    selected_zarr = tmp_path / "selected" / "skims.zarr"
    selected_zarr.mkdir(parents=True)
    (selected_zarr / ".zgroup").write_text("{}\n", encoding="utf-8")
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
                _launch_context(tmp_path / "workspace"),
                skim_mode="zarr",
                zarr_input_path=str(selected_zarr),
            )

    volumes = called["volumes"]
    assert str(selected_zarr) not in volumes
    assert volumes[str(asim_compile_output_dir(workspace))].endswith("/output")
    command = called["command"]
    assert command.count("-o") == 1
    assert command[command.index("-o") + 1].endswith("/output")
