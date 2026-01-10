import os

import pytest
import yaml

from pilates.config import load_config
from workflow_state import WorkflowState


def _make_settings(tmp_path, start_year=2020, end_year=2021):
    state_path = tmp_path / "state.yaml"
    config = {
        "run": {
            "start_year": start_year,
            "end_year": end_year,
            "travel_model_freq": 1,
            "supply_demand_iters": 1,
            "scenario": "test",
            "region": "test",
            "output_run_name": "test_run",
            "state_file_loc": str(state_path),
            "output_directory": str(tmp_path),
            "models": {
                "land_use": "urbansim",
                "travel": None,
                "activity_demand": None,
                "vehicle_ownership": None,
            },
        },
        "shared": {
            "geography": {
                "FIPS": {"county": ["06001"]},
                "local_crs": "EPSG:32048",
            },
            "skims": {
                "zone_type": "taz",
                "fname": "skims.h5",
                "geoms_fname": "geoms.geojson",
                "geoms_index_col": "TAZ",
            },
            "database": {
                "enabled": True,
                "type": "duckdb",
                "path": str(tmp_path / "db.duckdb"),
                "use_consist": False,
            },
        },
        "urbansim": {
            "local_data_input_folder": str(tmp_path / "usim_input"),
            "local_mutable_data_folder": str(tmp_path / "usim_mutable"),
            "client_base_folder": "/app",
            "input_file_template": "input_{region_id}.h5",
            "input_file_template_year": "input_{region_id}_{year}.h5",
            "output_file_template": "output_{year}.h5",
            "region_mappings": {"region_to_region_id": {"test": "123"}},
            "client_data_folder": "/tmp",
            "command_template": "echo",
        },
        "infrastructure": {
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
    }
    config_path = tmp_path / "settings.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    settings = load_config(str(config_path))
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = False
    settings.activity_demand_enabled = False
    settings.traffic_assignment_enabled = False
    settings.replanning_enabled = False
    settings.state_file_loc = str(state_path)
    return settings


def test_state_write_and_load_roundtrip(tmp_path):
    settings = _make_settings(tmp_path)
    state = WorkflowState.from_settings(settings)
    state.current_year = 2020
    state.current_major_stage = WorkflowState.Stage.land_use
    state.current_inner_iter = 2
    state.sub_stage_progress = "halfway"
    state.run_info_path = os.path.join(str(tmp_path), "run_info.yaml")
    state.data_initialized = True
    state.write_state()

    (
        year,
        stage,
        iteration,
        asim_compiled,
        sub_stage_progress,
        run_info_path,
        data_initialized,
    ) = WorkflowState.read_current_stage(state.file_loc)

    assert year == 2020
    assert stage == WorkflowState.Stage.land_use
    assert iteration == 2
    assert asim_compiled is False
    assert sub_stage_progress == "halfway"
    assert run_info_path == state.run_info_path
    assert data_initialized is True


def test_state_resume_after_interruption(tmp_path):
    settings = _make_settings(tmp_path)
    state = WorkflowState.from_settings(settings)
    state.current_year = 2020
    state.current_major_stage = WorkflowState.Stage.land_use
    state.current_inner_iter = 1
    state.sub_stage_progress = "preprocess"
    state.write_state()

    resumed = WorkflowState.from_settings(settings)
    assert resumed.current_year == 2020
    assert resumed.current_major_stage == WorkflowState.Stage.land_use
    assert resumed.current_inner_iter == 1
    assert resumed.sub_stage_progress == "preprocess"


def test_state_corruption_detection(tmp_path):
    settings = _make_settings(tmp_path)
    state_path = settings.state_file_loc
    with open(state_path, "w", encoding="utf-8") as handle:
        handle.write("bad: [")

    with pytest.raises(yaml.YAMLError):
        WorkflowState.from_settings(settings)


def test_state_consistency_across_years(tmp_path):
    settings = _make_settings(tmp_path, start_year=2020, end_year=2021)
    state = WorkflowState.from_settings(settings)
    state.current_year = 2020
    state.current_major_stage = WorkflowState.Stage.land_use
    state.current_inner_iter = 0

    state.complete_step(WorkflowState.Stage.land_use)
    assert state.current_year == 2021
    assert state.current_major_stage == WorkflowState.Stage.land_use

    state.complete_step(WorkflowState.Stage.land_use)
    assert state.current_year == 2022
    assert state.current_major_stage is None
