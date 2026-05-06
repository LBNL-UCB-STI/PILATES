from pathlib import Path

import pytest

from pilates.atlas.preprocessor import AtlasPreprocessor
import pilates.atlas.preprocessor as atlas_preprocessor_module
from pilates.workflows import binding as workflow_binding
from workflow_state import WorkflowState


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_copy_data_to_mutable_location_uses_fallback_source_for_missing_required_files(
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

    preprocessor = AtlasPreprocessor.__new__(AtlasPreprocessor)
    settings = {
        "atlas": {
            "host_input_folder": str(primary_source),
            "scenario": "zev_mandate",
            "adscen": "zev_mandate",
        }
    }

    _inputs, outputs = preprocessor.copy_data_to_mutable_location(
        settings=settings,
        output_dir=str(output_dir),
    )

    output_paths = {
        Path(record.file_path).relative_to(output_dir).as_posix()
        for record in outputs.all_records()
    }
    assert set(required_relpaths) <= output_paths

    inputs_by_key = {record.short_name: record for record in _inputs.all_records()}
    psid_record = inputs_by_key["psid_names"]
    assert psid_record.metadata["atlas_static_input"] is True
    assert psid_record.metadata["atlas_relpath"] == "psid_names.Rdat"
    assert psid_record.metadata["atlas_source_origin"] == "fallback"
    assert psid_record.metadata["atlas_input_group"] == "global"

    adopt_record = inputs_by_key["adopt/zev_mandate/new_vehicles_biannual_values_2021"]
    assert adopt_record.metadata["atlas_static_input"] is True
    assert adopt_record.metadata["atlas_source_origin"] == "fallback"
    assert adopt_record.metadata["atlas_input_group"] == "adopt"
    assert adopt_record.metadata["atlas_scenario"] == "zev_mandate"
    assert adopt_record.metadata["atlas_input_year"] == 2021
    assert adopt_record.metadata["profile_file_schema"] is True


def test_copy_data_to_mutable_location_raises_when_required_static_file_missing(
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

    preprocessor = AtlasPreprocessor.__new__(AtlasPreprocessor)
    settings = {
        "atlas": {
            "host_input_folder": str(primary_source),
            "scenario": "zev_mandate",
            "adscen": "zev_mandate",
        }
    }

    with pytest.raises(RuntimeError, match="Missing required ATLAS static input files"):
        preprocessor.copy_data_to_mutable_location(
            settings=settings,
            output_dir=str(output_dir),
        )


def test_restore_restart_atlas_year_inputs_copies_base_and_prior_subyear(tmp_path):
    previous_run_dir = tmp_path / "previous-run"
    current_run_dir = tmp_path / "current-run"
    previous_atlas_input = previous_run_dir / "atlas" / "atlas_input"
    current_atlas_input = current_run_dir / "atlas" / "atlas_input"

    _touch(previous_atlas_input / "year2017" / "households.csv")
    _touch(previous_atlas_input / "year2021" / "households.csv")

    workspace = type(
        "Workspace",
        (),
        {
            "get_atlas_mutable_input_dir": lambda self: str(current_atlas_input),
        },
    )()

    atlas_preprocessor_module._restore_restart_atlas_year_inputs(
        previous_run_dir=str(previous_run_dir),
        workspace=workspace,
        start_year=2017,
        atlas_year=2023,
    )

    assert (current_atlas_input / "year2017" / "households.csv").exists()
    assert (current_atlas_input / "year2021" / "households.csv").exists()


def test_restore_restart_atlas_year_inputs_repairs_partial_prior_subyear_directory(
    tmp_path,
):
    previous_run_dir = tmp_path / "previous-run"
    current_run_dir = tmp_path / "current-run"
    previous_atlas_input = previous_run_dir / "atlas" / "atlas_input"
    current_atlas_input = current_run_dir / "atlas" / "atlas_input"

    _touch(previous_atlas_input / "year2017" / "households.csv")
    _touch(previous_atlas_input / "year2017" / "blocks.csv")
    _touch(previous_atlas_input / "year2021" / "households.csv")
    _touch(previous_atlas_input / "year2021" / "grave.csv")
    _touch(previous_atlas_input / "year2021" / "vehicles_output.RData")
    _touch(previous_atlas_input / "year2021" / "households_output.RData")

    # Simulate a partial local restore: the year directory exists, but core CSVs do not.
    _touch(current_atlas_input / "year2021" / "vehicles_output.RData")
    _touch(current_atlas_input / "year2021" / "households_output.RData")

    workspace = type(
        "Workspace",
        (),
        {
            "get_atlas_mutable_input_dir": lambda self: str(current_atlas_input),
        },
    )()

    atlas_preprocessor_module._restore_restart_atlas_year_inputs(
        previous_run_dir=str(previous_run_dir),
        workspace=workspace,
        start_year=2017,
        atlas_year=2023,
    )

    assert (current_atlas_input / "year2021" / "households.csv").exists()
    assert (current_atlas_input / "year2021" / "grave.csv").exists()


def test_restart_required_atlas_input_years_uses_previous_subyear_not_start_year_minus_two():
    assert atlas_preprocessor_module._restart_required_atlas_input_years(
        start_year=2017,
        atlas_year=2023,
    ) == [2017, 2021]


def test_restart_atlas_required_artifacts_include_prior_subyear_directory(tmp_path):
    atlas_input_dir = tmp_path / "atlas" / "atlas_input"
    workspace = type(
        "Workspace",
        (),
        {
            "get_atlas_mutable_input_dir": lambda self: str(atlas_input_dir),
        },
    )()
    settings = type(
        "Settings",
        (),
        {
            "run": type(
                "RunCfg",
                (),
                {"models": type("Models", (), {"vehicle_ownership": "atlas"})()},
            )()
        },
    )()
    state = type(
        "State",
        (),
        {
            "start_year": 2017,
            "year": 2023,
            "current_year": 2023,
            "current_major_stage": WorkflowState.Stage.vehicle_ownership_model,
        },
    )()

    required = workflow_binding._restart_atlas_required_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
        atlas_static_input_relpaths_fn=lambda _settings: ("psid_names.Rdat",),
        workflow_stage=WorkflowState.Stage,
    )

    assert required is not None
    assert required["atlas_static::psid_names.Rdat"] == str(
        atlas_input_dir / "psid_names.Rdat"
    )
    assert required["atlas_restart_seed::2017::households"] == str(
        atlas_input_dir / "year2017" / "households.csv"
    )
    assert required["atlas_restart_seed::2017::blocks"] == str(
        atlas_input_dir / "year2017" / "blocks.csv"
    )
    assert required["atlas_restart_prior::2021::households"] == str(
        atlas_input_dir / "year2021" / "households.csv"
    )
    assert required["atlas_restart_prior::2021::grave"] == str(
        atlas_input_dir / "year2021" / "grave.csv"
    )
    assert required["atlas_restart_prior::2021::vehicles_output_RData"] == str(
        atlas_input_dir / "year2021" / "vehicles_output.RData"
    )
    assert required["atlas_restart_prior::2021::households_output_RData"] == str(
        atlas_input_dir / "year2021" / "households_output.RData"
    )
