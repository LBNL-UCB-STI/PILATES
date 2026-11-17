from pathlib import Path
import inspect

import pytest

import pilates.atlas.preprocessor as atlas_preprocessor_module
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_stage_selected_atlas_static_inputs_uses_fallback_source_for_missing_required_files(
    tmp_path, monkeypatch
):
    primary_source = tmp_path / "primary_atlas_input"
    project_root = tmp_path / "project_root"
    fallback_source = project_root / "pilates" / "atlas" / "atlas_input"
    output_dir = tmp_path / "output"

    required_relpaths = (
        "psid_names.Rdat",
        "adopt/zev_mandate/new_vehicles_biannual_values_2021.csv",
    )
    _touch(fallback_source / required_relpaths[0])
    _touch(fallback_source / required_relpaths[1], content="Year,value\n2021,1\n")

    # Primary source intentionally does not contain required files.
    primary_source.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        atlas_preprocessor_module,
        "atlas_static_input_relpaths",
        lambda _settings: required_relpaths,
    )
    monkeypatch.setattr(
        atlas_preprocessor_module,
        "find_project_root",
        lambda start_path=None: str(project_root),
    )

    settings = {
        "atlas": {
            "host_input_folder": str(primary_source),
            "scenario": "zev_mandate",
            "adscen": "zev_mandate",
        }
    }

    staged_paths = atlas_preprocessor_module.stage_selected_atlas_static_inputs(
        settings=settings,
        output_dir=str(output_dir),
    )

    output_paths = {
        Path(path).relative_to(output_dir).as_posix() for path in staged_paths.values()
    }
    assert set(required_relpaths) <= output_paths


def test_stage_selected_atlas_static_inputs_raises_when_required_static_file_missing(
    tmp_path, monkeypatch
):
    primary_source = tmp_path / "primary_atlas_input"
    project_root = tmp_path / "project_root"
    fallback_source = project_root / "pilates" / "atlas" / "atlas_input"
    output_dir = tmp_path / "output"

    required_relpaths = ("adopt/zev_mandate/new_vehicles_biannual_values_2021.csv",)
    primary_source.mkdir(parents=True, exist_ok=True)
    fallback_source.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        atlas_preprocessor_module,
        "atlas_static_input_relpaths",
        lambda _settings: required_relpaths,
    )
    monkeypatch.setattr(
        atlas_preprocessor_module,
        "find_project_root",
        lambda start_path=None: str(project_root),
    )

    settings = {
        "atlas": {
            "host_input_folder": str(primary_source),
            "scenario": "zev_mandate",
            "adscen": "zev_mandate",
        }
    }

    with pytest.raises(RuntimeError, match="Missing required ATLAS static input files"):
        atlas_preprocessor_module.stage_selected_atlas_static_inputs(
            settings=settings,
            output_dir=str(output_dir),
        )


def test_vehicle_ownership_stage_does_not_archive_atlas_members_directly():
    """Native producer archival is centralized in ``execute_step``."""

    source = inspect.getsource(run_vehicle_ownership_stage)

    assert "archive_copy_now" not in source
    assert "flush_archive_queue" not in source


def test_land_use_stage_does_not_archive_urbansim_members_directly():
    """UrbanSim producer archival is centralized in ``execute_step``."""

    source = inspect.getsource(run_land_use_stage)

    assert "archive_copy_now" not in source
    assert "flush_archive_queue" not in source
