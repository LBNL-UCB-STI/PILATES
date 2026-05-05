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


def test_runtime_values_are_excluded_from_model_dump():
    settings = PilatesConfig(**_minimal_config())
    settings.land_use_enabled = True
    settings.state_file_loc = "/tmp/state.yaml"

    dumped = settings.model_dump()

    assert "runtime" not in dumped
    assert "land_use_enabled" not in dumped
    assert "state_file_loc" not in dumped


def test_deprecated_use_consist_warns_and_is_ignored():
    config = _minimal_config()
    config["shared"]["database"]["use_consist"] = False

    with pytest.warns(DeprecationWarning, match="shared.database.use_consist"):
        settings = PilatesConfig(**config)

    assert settings.shared.database.use_consist is True


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
