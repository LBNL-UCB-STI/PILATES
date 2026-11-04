import pytest

from pilates.config import PilatesConfig


def _minimal_config() -> dict:
    return {
        "run": {
            "region": "test",
            "scenario": "test",
            "start_year": 2020,
            "end_year": 2021,
            "use_stubs": False,
            "land_use_freq": 1,
            "travel_model_freq": 1,
            "vehicle_ownership_freq": 1,
            "supply_demand_iters": 1,
            "output_directory": "/tmp/output",
            "output_run_name": "test-run",
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
                "path": "/tmp/test.duckdb",
            },
        },
        "urbansim": {
            "local_data_input_folder": "usim_input",
            "local_mutable_data_folder": "usim_mutable",
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


def test_runtime_properties_forward_to_typed_runtime_container():
    settings = PilatesConfig(**_minimal_config())

    assert settings.runtime.flags_initialized is True
    assert settings.land_use_enabled is True
    assert settings.activity_demand_enabled is False

    settings.land_use_enabled = True
    settings.activity_demand_enabled = False
    settings.state_file_loc = "/tmp/state.yaml"
    settings.settings_file = "settings.yaml"
    settings.allow_rewind_resume = True

    assert settings.runtime.flags.land_use_enabled is True
    assert settings.runtime.flags.activity_demand_enabled is False
    assert settings.runtime.options.state_file_loc == "/tmp/state.yaml"
    assert settings.runtime.options.settings_file == "settings.yaml"
    assert settings.runtime.options.allow_rewind_resume is True


def test_runtime_flags_derive_live_and_preserve_explicit_overrides():
    settings = PilatesConfig(**_minimal_config())

    assert settings.land_use_enabled is True
    settings.run.models.land_use = None
    assert settings.land_use_enabled is False

    # Explicit override via property or runtime.flags pins the flag
    settings.land_use_enabled = True
    settings.run.models.land_use = None
    assert settings.land_use_enabled is True
    settings.runtime.flags.activity_demand_enabled = True
    assert settings.activity_demand_enabled is True

    # Partial test stubs without replan_iters fall back to 0 cleanly
    from types import SimpleNamespace

    object.__setattr__(settings, "activitysim", SimpleNamespace(file_format="parquet"))
    assert settings.replanning_enabled is False


def test_final_skim_export_setting_is_typed_and_defaults_off():
    settings = PilatesConfig(**_minimal_config())

    assert settings.write_skims_to_omx is False

    settings.write_skims_to_omx = True
    assert settings.write_skims_to_omx is True


def test_max_year_intervals_is_part_of_initialization_identity():
    config = _minimal_config()
    config["run"]["max_year_intervals"] = 1

    settings = PilatesConfig(**config)

    assert (
        settings.get_initialization_signature()["orchestration"]["max_year_intervals"]
        == 1
    )


def test_runtime_values_are_excluded_from_model_dump():
    settings = PilatesConfig(**_minimal_config())
    settings.land_use_enabled = True
    settings.state_file_loc = "/tmp/state.yaml"

    dumped = settings.model_dump()

    assert "runtime" not in dumped
    assert "land_use_enabled" not in dumped
    assert "state_file_loc" not in dumped


def test_workflow_rejects_retired_manifest_recovery_config():
    config = _minimal_config()
    config["workflow"] = {"manifests": {"disabled_stages": ["supply_demand"]}}

    with pytest.raises(ValueError, match=r"workflow\.manifests"):
        PilatesConfig(**config)


def test_recovery_archive_roots_expand_environment_variables(monkeypatch):
    monkeypatch.setenv("PILATES_RECOVERY_ROOT", "/tmp/recovery-root")
    config = _minimal_config()
    config["run"]["recovery_archive_roots"] = [
        "${PILATES_RECOVERY_ROOT}",
        "/tmp/static-root",
    ]

    settings = PilatesConfig(**config)

    assert settings.run.recovery_archive_roots == [
        "/tmp/recovery-root",
        "/tmp/static-root",
    ]


def test_model_copy_isolates_runtime_flags_and_options():
    settings = PilatesConfig(**_minimal_config())
    settings.land_use_enabled = False
    settings.state_file_loc = "/tmp/original_state.yaml"

    copied = settings.model_copy()
    copied.land_use_enabled = True
    copied.state_file_loc = "/tmp/copied_state.yaml"

    assert settings.land_use_enabled is False
    assert settings.state_file_loc == "/tmp/original_state.yaml"
    assert copied.land_use_enabled is True
    assert copied.state_file_loc == "/tmp/copied_state.yaml"
    assert settings._runtime is not copied._runtime
    assert settings._runtime.flags is not copied._runtime.flags
    assert settings._runtime.options is not copied._runtime.options


def test_traffic_assignment_flag_derivation_matches_compute_model_enabled_flags():
    from pilates.utils.io import compute_model_enabled_flags

    config = _minimal_config()
    config["run"]["models"]["travel"] = None
    config["traffic_assignment"] = "beam"

    settings = PilatesConfig(**config)

    assert settings.traffic_assignment_enabled is True
    assert compute_model_enabled_flags(settings)["traffic_assignment_enabled"] is True
