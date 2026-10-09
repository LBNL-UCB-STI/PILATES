"""Regression coverage for generated ActivitySim Zarr skim ownership."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim.outputs import (
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.runner import (
    ActivitySimLaunchContext,
    ActivitysimRunner,
    asim_runtime_zarr_path,
    validate_activitysim_zarr_skims,
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


def _preparation_settings() -> SimpleNamespace:
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
            num_processes=2,
            chunk_size=0,
        ),
    )


@pytest.mark.parametrize("generated_zarr", ["missing", "invalid"])
def test_omx_preparation_rejects_missing_or_invalid_zarr_before_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    generated_zarr: str,
) -> None:
    """A warmup success cannot admit the body without a valid runtime Zarr."""
    launch_context = _launch_context(tmp_path)
    state = SimpleNamespace(
        full_settings=_preparation_settings(),
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
            if generated_zarr == "invalid":
                zarr_path = launch_context.runtime_zarr_path
                zarr_path.mkdir(parents=True)
                (zarr_path / ".zgroup").write_text("{}\n", encoding="utf-8")
            return True

        assert model_name == "activitysim"
        calls.append("body")
        return True

    monkeypatch.setattr(
        "pilates.activitysim.runner.GenericRunner.run_container",
        staticmethod(run_container),
    )

    with pytest.raises(RuntimeError):
        runner.run(
            ActivitySimPreprocessOutputs(
                mutable_data_dir=tmp_path / "mutable-data",
                land_use_table=tmp_path / "land_use.csv",
                households_table=tmp_path / "households.csv",
                persons_table=tmp_path / "persons.csv",
            ),
            launch_context,
            skim_mode="omx",
            workspace=SimpleNamespace(full_path=str(tmp_path)),
        )

    assert calls == ["warmup"]


def test_zarr_validation_is_read_only_for_a_structurally_valid_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "selected" / "skims.zarr"
    zarr_path.mkdir(parents=True)
    (zarr_path / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    (zarr_path / "zone_flags.json").write_text('{"zero_based": false}\n', encoding="utf-8")
    before = {
        path.relative_to(zarr_path): path.read_bytes()
        for path in zarr_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        "pilates.activitysim.runner.ensure_0_based_and_flag_zarr_skims",
        lambda *_args: pytest.fail("read-only validation must not normalize skims"),
    )

    assert validate_activitysim_zarr_skims(zarr_path) is None

    after = {
        path.relative_to(zarr_path): path.read_bytes()
        for path in zarr_path.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("metadata_name", "metadata", "rejection"),
    [
        (".zgroup", "{}\n", ".zgroup must declare zarr_format 2"),
        (
            ".zgroup",
            '{"zarr_format": 1}\n',
            ".zgroup must declare zarr_format 2",
        ),
        (
            "zarr.json",
            '{"zarr_format": 2, "node_type": "group"}\n',
            "zarr.json must declare zarr_format 3",
        ),
        (
            "zarr.json",
            '{"zarr_format": 3, "node_type": "array"}\n',
            "zarr.json root node_type must be 'group'",
        ),
    ],
)
def test_zarr_validation_rejects_json_with_an_invalid_root_format(
    tmp_path: Path,
    metadata_name: str,
    metadata: str,
    rejection: str,
) -> None:
    zarr_path = tmp_path / "selected" / "skims.zarr"
    zarr_path.mkdir(parents=True)
    (zarr_path / metadata_name).write_text(metadata, encoding="utf-8")

    assert validate_activitysim_zarr_skims(zarr_path) == rejection


def test_zarr_validation_accepts_a_v3_group_root(tmp_path: Path) -> None:
    zarr_path = tmp_path / "selected" / "skims.zarr"
    zarr_path.mkdir(parents=True)
    (zarr_path / "zarr.json").write_text(
        '{"attributes": {}, "zarr_format": 3, "node_type": "group"}\n',
        encoding="utf-8",
    )

    assert validate_activitysim_zarr_skims(zarr_path) is None


def test_zarr_validation_rejects_ambiguous_v2_and_v3_root_metadata(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "selected" / "skims.zarr"
    zarr_path.mkdir(parents=True)
    (zarr_path / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    (zarr_path / "zarr.json").write_text(
        '{"zarr_format": 3, "node_type": "group"}\n', encoding="utf-8"
    )

    assert validate_activitysim_zarr_skims(zarr_path) == (
        "ambiguous Zarr root metadata (.zgroup and zarr.json)"
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
