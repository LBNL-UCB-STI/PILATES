import os

import pytest
import yaml

from pilates.config import load_config
from workflow_state import WorkflowState


def _make_settings(tmp_path, start_year=2020, end_year=2021, travel_model_freq=1):
    state_path = tmp_path / "state.yaml"
    config = {
        "run": {
            "start_year": start_year,
            "end_year": end_year,
            "travel_model_freq": travel_model_freq,
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


def test_state_write_mirrors_to_secondary_path(tmp_path):
    settings = _make_settings(tmp_path)
    state = WorkflowState.from_settings(settings)
    state.current_year = 2020
    state.current_major_stage = WorkflowState.Stage.land_use
    mirror_path = tmp_path / "mirror" / "run_state.yaml"
    state.mirror_file_loc = str(mirror_path)

    state.write_state()

    assert os.path.exists(state.file_loc)
    assert mirror_path.exists()
    with open(state.file_loc, encoding="utf-8") as primary:
        with open(mirror_path, encoding="utf-8") as mirror:
            assert yaml.safe_load(primary) == yaml.safe_load(mirror)


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


def test_interval_only_progression_uses_forecast_boundaries(tmp_path):
    settings = _make_settings(
        tmp_path, start_year=2017, end_year=2030, travel_model_freq=6
    )
    state = WorkflowState.from_settings(settings)

    assert state.current_year == 2017
    assert state.forecast_year == 2023

    state.complete_step(WorkflowState.Stage.land_use)
    assert state.current_year == 2023
    assert state.forecast_year == 2029
    assert state.current_major_stage == WorkflowState.Stage.land_use

    state.complete_step(WorkflowState.Stage.land_use)
    assert state.current_year == 2029
    assert state.forecast_year == 2030
    assert state.current_major_stage == WorkflowState.Stage.land_use

    state.complete_step(WorkflowState.Stage.land_use)
    assert state.current_year == 2030
    assert state.forecast_year == 2030
    assert state.current_major_stage == WorkflowState.Stage.land_use

    state.complete_step(WorkflowState.Stage.land_use)
    assert state.current_year == 2031
    assert state.current_major_stage is None


def test_interval_progression_waits_until_last_enabled_major_stage(tmp_path):
    settings = _make_settings(
        tmp_path, start_year=2017, end_year=2030, travel_model_freq=6
    )
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = False

    state = WorkflowState.from_settings(settings)

    assert state.current_year == 2017
    assert state.forecast_year == 2023
    assert state.current_major_stage == WorkflowState.Stage.land_use

    state.complete_step(WorkflowState.Stage.land_use)
    assert state.current_year == 2017
    assert state.current_major_stage == WorkflowState.Stage.vehicle_ownership_model

    state.complete_step(WorkflowState.Stage.vehicle_ownership_model)
    assert state.current_year == 2017
    assert state.current_major_stage == WorkflowState.Stage.supply_demand_loop
    assert state.current_sub_stage == WorkflowState.Stage.activity_demand

    state.complete_step(
        WorkflowState.Stage.supply_demand_loop,
        completed_inner_iter=0,
        completed_sub=WorkflowState.Stage.activity_demand,
    )
    assert state.current_year == 2023
    assert state.forecast_year == 2029
    assert state.current_major_stage == WorkflowState.Stage.land_use


def test_land_use_disabled_forecast_year_matches_current_year(tmp_path):
    settings = _make_settings(tmp_path, start_year=2020, end_year=2030)
    settings.land_use_enabled = False
    settings.vehicle_ownership_model_enabled = True

    state = WorkflowState.from_settings(settings)

    assert state.current_year == 2020
    assert state.forecast_year == 2020
    assert state.current_major_stage == WorkflowState.Stage.vehicle_ownership_model


def test_land_use_disabled_exits_after_single_outer_cycle(tmp_path):
    settings = _make_settings(tmp_path, start_year=2020, end_year=2030)
    settings.land_use_enabled = False
    settings.vehicle_ownership_model_enabled = True

    state = WorkflowState.from_settings(settings)
    state.complete_step(WorkflowState.Stage.vehicle_ownership_model)

    assert state.current_year == settings.run.end_year + 1
    assert state.current_major_stage is None
    assert state.current_year != settings.run.start_year + 1
    with pytest.raises(StopIteration):
        next(state)


def test_2010_special_case_progression_has_no_backward_jumps(tmp_path):
    settings = _make_settings(
        tmp_path, start_year=2010, end_year=2025, travel_model_freq=6
    )
    state = WorkflowState.from_settings(settings)

    assert state.current_year == 2010
    assert state.forecast_year == 2017

    observed_years = [state.current_year]
    observed_forecasts = [state.forecast_year]

    for _ in range(3):
        state.complete_step(WorkflowState.Stage.land_use)
        observed_years.append(state.current_year)
        observed_forecasts.append(state.forecast_year)

    assert observed_years == [2010, 2017, 2023, 2025]
    assert observed_forecasts == [2017, 2023, 2025, 2025]
    assert all(
        later > earlier for earlier, later in zip(observed_years, observed_years[1:])
    )

    state.complete_step(WorkflowState.Stage.land_use)
    assert state.current_year == 2026
    assert state.current_major_stage is None


def test_terminal_state_resume_does_not_reinitialize_stages(tmp_path):
    settings = _make_settings(
        tmp_path, start_year=2020, end_year=2025, travel_model_freq=6
    )
    terminal_year = settings.run.end_year + 1

    WorkflowState.write_stage(
        terminal_year,
        None,
        settings.state_file_loc,
        0,
        False,
        None,
        settings.state_file_loc,
        True,
    )

    resumed = WorkflowState.from_settings(settings)
    assert resumed.current_year == terminal_year
    assert resumed.current_major_stage is None

    with pytest.raises(StopIteration):
        next(resumed)

    assert not os.path.exists(settings.state_file_loc)
