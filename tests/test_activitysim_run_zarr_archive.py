"""Regression coverage for generated ActivitySim Zarr skim ownership."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pilates.activitysim.outputs import (
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.runner import (
    ActivitySimLaunchContext,
    ActivitysimRunner,
    asim_runtime_zarr_path,
)


def _workspace(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_output_dir=lambda: str(tmp_path / "activitysim" / "output"),
        get_asim_runtime_cache_dir=lambda: str(
            tmp_path / "activitysim" / "output" / "cache"
        ),
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        activitysim=SimpleNamespace(
            region_mappings={"region_to_subdir": {"seattle": "example"}},
            local_output_folder="activitysim/output",
            command_template="activitysim",
            household_sample_size=None,
            num_processes=1,
            chunk_size=None,
        ),
    )


def _launch_context(tmp_path: Path) -> ActivitySimLaunchContext:
    output_dir = tmp_path / "activitysim" / "output"
    runtime_cache_dir = output_dir / "cache"
    return ActivitySimLaunchContext(
        workspace_root=tmp_path,
        mutable_data_dir=tmp_path / "activitysim" / "data",
        output_dir=output_dir,
        compile_output_dir=tmp_path / "activitysim" / "compile-output",
        mutable_configs_dir=tmp_path / "activitysim" / "configs",
        runtime_cache_dir=runtime_cache_dir,
        runtime_zarr_path=runtime_cache_dir / "skims.zarr",
        shared_cache_dir=tmp_path / "shared_cache",
        shared_tmp_dir=tmp_path / "tmp",
    )


def test_omx_mode_leaves_finalized_zarr_skims_to_central_archive(monkeypatch, tmp_path):
    workspace = _workspace(tmp_path)
    settings = _settings()
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        current_inner_iter=0,
        set_sub_stage_progress=lambda _progress: None,
    )
    runner = ActivitysimRunner("activitysim", state)
    zarr_path = Path(asim_runtime_zarr_path(workspace))
    archived: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        runner,
        "_run",
        lambda *_args, **_kwargs: ActivitySimRunOutputs(
            output_dir=Path(workspace.get_asim_output_dir())
        ),
    )

    def _finalize(path, *_args):
        finalized = Path(path)
        finalized.mkdir(parents=True)
        (finalized / ".zgroup").write_text("{}\n", encoding="utf-8")
        return finalized

    monkeypatch.setattr(
        "pilates.activitysim.runner.finalize_activitysim_zarr_skims", _finalize
    )
    monkeypatch.setattr(
        "pilates.activitysim.runner.enqueue_archive_copy",
        lambda *, key, path: archived.append((key, Path(path))),
        raising=False,
    )

    outputs = runner.run(
        ActivitySimPreprocessOutputs(
            mutable_data_dir=tmp_path,
            land_use_table=tmp_path / "land_use.csv",
            households_table=tmp_path / "households.csv",
            persons_table=tmp_path / "persons.csv",
        ),
        _launch_context(tmp_path),
        skim_mode="omx",
        workspace=workspace,
    )

    assert outputs.zarr_skims == zarr_path
    assert (zarr_path / ".zgroup").exists()
    assert archived == []


def test_zarr_mode_does_not_enqueue_its_input_skims_for_archiving(
    monkeypatch, tmp_path
):
    workspace = _workspace(tmp_path)
    settings = _settings()
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        current_inner_iter=0,
        set_sub_stage_progress=lambda _progress: None,
    )
    runner = ActivitysimRunner("activitysim", state)
    zarr_path = Path(asim_runtime_zarr_path(workspace))
    zarr_path.mkdir(parents=True)
    archived: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        runner,
        "_run",
        lambda *_args, **_kwargs: ActivitySimRunOutputs(
            output_dir=Path(workspace.get_asim_output_dir())
        ),
    )
    monkeypatch.setattr(
        "pilates.activitysim.runner.enqueue_archive_copy",
        lambda *, key, path: archived.append((key, Path(path))),
        raising=False,
    )

    runner.run(
        ActivitySimPreprocessOutputs(
            mutable_data_dir=tmp_path,
            land_use_table=tmp_path / "land_use.csv",
            households_table=tmp_path / "households.csv",
            persons_table=tmp_path / "persons.csv",
        ),
        _launch_context(tmp_path),
        skim_mode="zarr",
        extra_inputs={"zarr_skims": zarr_path},
    )

    assert archived == []
