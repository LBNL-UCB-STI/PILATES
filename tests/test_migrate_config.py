import textwrap
import runpy
import sys
from pathlib import Path

import pytest
import yaml

import scripts.migrate_config as migrate_config_module
from scripts.migrate_config import ConfigMigrator, migrate_config_file


def test_migrate_legacy_config_defaults_to_consist_db_on():
    legacy = {
        "region": "sfbay",
        "start_year": 2010,
        "end_year": 2012,
        "travel_model": "beam",
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["run"]["consist_db_local_run"] is True
    assert migrated["run"]["consist_db_filename"] == "provenance.duckdb"
    assert migrated["shared"]["database"] == {
        "enabled": True,
        "type": "duckdb",
        "path": "pilates/database/sfbay_pilates_data.duckdb",
    }


def test_migrate_legacy_config_preserves_explicit_database_settings():
    legacy = {
        "region": "seattle",
        "start_year": 2018,
        "end_year": 2020,
        "database": {
            "enabled": False,
            "type": "duckdb",
            "path": "/tmp/custom.duckdb",
        },
        "consist_db_local_run": False,
        "consist_db_filename": "custom.duckdb",
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["run"]["consist_db_local_run"] is False
    assert migrated["run"]["consist_db_filename"] == "custom.duckdb"
    assert migrated["shared"]["database"] == {
        "enabled": False,
        "type": "duckdb",
        "path": "/tmp/custom.duckdb",
    }


def test_migrate_config_only_emits_enabled_model_sections():
    legacy = {
        "region": "seattle",
        "scenario": "base",
        "start_year": 2018,
        "end_year": 2019,
        "travel_model": "beam",
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert "urbansim" in migrated
    assert "beam" in migrated
    assert "activitysim" not in migrated
    assert "atlas" not in migrated
    assert migrated["run"]["models"] == {
        "land_use": None,
        "travel": "beam",
        "activity_demand": None,
        "vehicle_ownership": None,
    }


def test_migrate_config_filters_empty_images_and_preserves_pull_latest():
    legacy = {
        "region": "sfbay",
        "start_year": 2018,
        "end_year": 2019,
        "travel_model": "beam",
        "container_manager": "singularity",
        "pull_latest": True,
        "docker_stdout": True,
        "singularity_images": {
            "urbansim": "docker://example/urbansim:latest",
            "beam": "docker://example/beam:latest",
            "activitysim": None,
        },
        "docker_images": {
            "urbansim": "example/urbansim:latest",
            "beam": "example/beam:latest",
            "activitysim": None,
        },
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["infrastructure"] == {
        "container_manager": "singularity",
        "singularity_images": {
            "urbansim": "docker://example/urbansim:latest",
            "beam": "docker://example/beam:latest",
        },
        "docker_images": {
            "urbansim": "example/urbansim:latest",
            "beam": "example/beam:latest",
        },
        "docker_config": {
            "stdout": True,
            "pull_latest": True,
        },
    }


def test_migrate_config_stringifies_region_ids_and_preserves_activitysim_mappings():
    legacy = {
        "region": "seattle",
        "start_year": 2018,
        "end_year": 2020,
        "travel_model": "beam",
        "activity_demand_model": "activitysim",
        "skims_zone_type": "taz",
        "region_to_region_id": {
            "seattle": 53199100,
            "sfbay": "06197001",
        },
        "region_to_asim_subdir": {"seattle": "seattle"},
        "region_to_asim_bucket": {"seattle": "seattle-activitysim"},
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["urbansim"]["region_id"] == "53199100"
    assert migrated["urbansim"]["region_mappings"]["region_to_region_id"] == {
        "seattle": "53199100",
        "sfbay": "06197001",
    }
    assert migrated["activitysim"]["region_mappings"] == {
        "region_to_subdir": {"seattle": "seattle"},
        "region_to_bucket": {"seattle": "seattle-activitysim"},
    }
    assert migrated["shared"]["geography"]["zones"] == {
        "zone_type": "block_group",
        "source_file": "pilates/activitysim/data/seattle/block_groups_seattle_4326.geojson",
        "canonical_id_col": "OBJECTID",
        "activitysim_index_col": "TAZ",
        "source_crs": "EPSG:4326",
    }
    assert migrated["shared"]["geography"]["alternative_zones"] == {
        "zone_type": "block_group",
        "source_file": "pilates/beam/production/seattle/shape/block-groups-32048.shp",
        "canonical_id_col": "OBJECTID",
        "activitysim_index_col": "TAZ",
        "source_crs": "EPSG:32048",
    }


def test_migrate_config_adds_sfbay_zone_fallback_source():
    legacy = {
        "region": "sfbay",
        "start_year": 2018,
        "end_year": 2018,
        "travel_model": "beam",
        "skims_zone_type": "taz",
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["shared"]["geography"]["zones"] == {
        "zone_type": "taz",
        "source_file": "pilates/activitysim/data/sfbay/taz_sfbay.geojson",
        "canonical_id_col": "taz1454",
        "activitysim_index_col": "TAZ",
        "source_crs": "EPSG:4326",
    }
    assert migrated["shared"]["geography"]["alternative_zones"] == {
        "zone_type": "taz",
        "source_file": "pilates/beam/production/sfbay/shape/sfbay-tazs-epsg-26910.shp",
        "canonical_id_col": "taz1454",
        "activitysim_index_col": "TAZ",
        "source_crs": "EPSG:26910",
    }


def test_migrate_config_builds_full_skim_parallelism_ratio_from_percent():
    legacy = {
        "region": "seattle",
        "start_year": 2018,
        "end_year": 2020,
        "travel_model": "beam",
        "beam_full_skim_run_schedule": "standalone",
        "beam_full_skim_router_type": "r5",
        "beam_full_skim_skims_geo_type": "block_group",
        "beam_full_skim_skims_kind": "od",
        "beam_full_skim_peak_hours": [8.5, 17.5],
        "beam_full_skim_modes_drive": True,
        "beam_full_skim_modes_walk": True,
        "beam_full_skim_modes_transit": False,
        "beam_full_skim_parallelism": 80,
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["beam"]["full_skim"] == {
        "run_schedule": "standalone",
        "router_type": "r5",
        "skims_geo_type": "block_group",
        "skims_kind": "od",
        "peak_hours": [8.5, 17.5],
        "modes_to_build": {
            "drive": True,
            "walk": True,
            "transit": False,
        },
        "parallelism_thread_ratio": 0.8,
    }


def test_migrate_config_includes_atlas_when_vehicle_ownership_model_enabled():
    legacy = {
        "region": "sfbay",
        "start_year": 2018,
        "end_year": 2019,
        "travel_model": "beam",
        "vehicle_ownership_model": "atlas",
        "atlas_host_input_folder": "host/in",
        "atlas_formattable_command": "--atlas",
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["run"]["models"]["vehicle_ownership"] == "atlas"
    assert migrated["atlas"]["host_input_folder"] == "host/in"
    assert migrated["atlas"]["command_template"] == "--atlas"


def test_migrate_config_builds_full_skim_defaults_when_modes_omitted():
    legacy = {
        "region": "seattle",
        "start_year": 2018,
        "end_year": 2020,
        "travel_model": "beam",
        "beam_full_skim_run_schedule": "after_final_iteration",
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["beam"]["full_skim"]["modes_to_build"] == {
        "drive": True,
        "walk": False,
        "transit": False,
    }


def test_migrate_config_uses_full_skim_parallelism_thread_ratio_directly():
    legacy = {
        "region": "seattle",
        "start_year": 2018,
        "end_year": 2020,
        "travel_model": "beam",
        "beam_full_skim_run_schedule": "standalone",
        "beam_full_skim_parallelism_thread_ratio": 0.5,
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["beam"]["full_skim"]["parallelism_thread_ratio"] == 0.5


def test_migrate_config_converts_full_skim_parallelism_thread_pct():
    legacy = {
        "region": "seattle",
        "start_year": 2018,
        "end_year": 2020,
        "travel_model": "beam",
        "beam_full_skim_run_schedule": "standalone",
        "beam_full_skim_parallelism_thread_pct": 75,
    }

    migrated = ConfigMigrator(legacy).migrate()

    assert migrated["beam"]["full_skim"]["parallelism_thread_ratio"] == 0.75


def test_migrate_config_file_writes_hierarchical_yaml_with_expected_sections(tmp_path):
    legacy_path = tmp_path / "legacy.yaml"
    output_path = tmp_path / "migrated.yaml"
    legacy_path.write_text(
        textwrap.dedent(
            """
            region: seattle
            scenario: base
            start_year: 2018
            end_year: 2019
            output_directory: /tmp/output
            output_run_name: demo
            land_use_model:
            travel_model: beam
            activity_demand_model: activitysim
            vehicle_ownership_model:
            beam_config: beam.conf
            beam_memory: "${BEAM_MEMORY}"
            region_to_region_id:
              seattle: 53199100
            region_to_asim_subdir:
              seattle: seattle
            region_to_asim_bucket:
              seattle: seattle-activitysim
            singularity_images:
              beam: docker://haitamlaarabi/beam:1.1-beta-v260314
            docker_images:
              beam: haitamlaarabi/beam:1.1-beta-v260314
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        migrate_config_file(str(legacy_path), str(output_path), validate=True) is True
    )

    written = output_path.read_text(encoding="utf-8")
    assert written.startswith("# PILATES Configuration File (Hierarchical Format)\n")
    assert "# RUN CONFIGURATION" in written
    assert "# ACTIVITYSIM CONFIGURATION" in written
    assert '"${BEAM_MEMORY}"' in written
    assert '"53199100"' in written

    migrated = yaml.safe_load(written)
    assert migrated["run"]["region"] == "seattle"
    assert migrated["beam"]["memory"] == "${BEAM_MEMORY}"
    assert migrated["urbansim"]["region_id"] == "53199100"
    assert (
        migrated["activitysim"]["region_mappings"]["region_to_subdir"]["seattle"]
        == "seattle"
    )


def test_migrate_config_file_validation_error_logs_and_still_writes(
    tmp_path, monkeypatch, caplog
):
    legacy_path = tmp_path / "legacy.yaml"
    output_path = tmp_path / "migrated.yaml"
    legacy_path.write_text(
        "region: seattle\nstart_year: 2018\nend_year: 2019\ntravel_model: beam\n",
        encoding="utf-8",
    )

    def _boom(_payload):
        raise ValueError("validation exploded")

    monkeypatch.setattr(migrate_config_module, "PilatesConfig", _boom)

    with caplog.at_level("INFO"):
        result = migrate_config_file(str(legacy_path), str(output_path), validate=True)

    assert result is True
    assert output_path.exists()
    assert "Configuration validation failed" in caplog.text
    assert "Writing config anyway" in caplog.text


def test_migrate_config_file_writes_atlas_section_and_reports_warnings(
    tmp_path, monkeypatch, caplog
):
    legacy_path = tmp_path / "legacy.yaml"
    output_path = tmp_path / "migrated.yaml"
    legacy_path.write_text(
        textwrap.dedent(
            """
            region: sfbay
            start_year: 2018
            end_year: 2019
            output_directory: /tmp/output
            output_run_name: atlas-demo
            travel_model: beam
            vehicle_ownership_model: atlas
            atlas_host_input_folder: host/in
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    original_get_warnings = migrate_config_module.ConfigMigrator.get_warnings

    def _fake_get_warnings(self):
        return ["warning one", "warning two"]

    monkeypatch.setattr(
        migrate_config_module.ConfigMigrator, "get_warnings", _fake_get_warnings
    )

    with caplog.at_level("WARNING"):
        result = migrate_config_file(str(legacy_path), str(output_path), validate=False)

    monkeypatch.setattr(
        migrate_config_module.ConfigMigrator, "get_warnings", original_get_warnings
    )

    assert result is True
    written = output_path.read_text(encoding="utf-8")
    assert "# ATLAS CONFIGURATION" in written
    assert "Migration completed with 2 warnings" in caplog.text
    assert "warning one" in caplog.text
    assert "warning two" in caplog.text


def test_migrate_config_file_returns_false_on_invalid_input(tmp_path):
    legacy_path = tmp_path / "legacy.yaml"
    output_path = tmp_path / "migrated.yaml"
    legacy_path.write_text("[not valid yaml", encoding="utf-8")

    assert (
        migrate_config_file(str(legacy_path), str(output_path), validate=False) is False
    )
    assert not output_path.exists()


def test_main_uses_cli_args_and_exits_zero(monkeypatch):
    called = {}

    def _fake_migrate(input_path, output_path, validate):
        called["args"] = (input_path, output_path, validate)
        return True

    monkeypatch.setattr(migrate_config_module, "migrate_config_file", _fake_migrate)
    monkeypatch.setattr(
        migrate_config_module.sys,
        "argv",
        ["migrate_config.py", "in.yaml", "out.yaml", "--no-validate", "--verbose"],
    )

    with pytest.raises(SystemExit) as excinfo:
        migrate_config_module.main()

    assert excinfo.value.code == 0
    assert called["args"] == ("in.yaml", "out.yaml", False)


def test_main_exits_one_on_migration_failure(monkeypatch):
    monkeypatch.setattr(
        migrate_config_module, "migrate_config_file", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        migrate_config_module.sys,
        "argv",
        ["migrate_config.py", "in.yaml", "out.yaml"],
    )

    with pytest.raises(SystemExit) as excinfo:
        migrate_config_module.main()

    assert excinfo.value.code == 1


def test_module_main_entrypoint_executes_when_run_as_main(tmp_path, monkeypatch):
    legacy_path = tmp_path / "legacy.yaml"
    output_path = tmp_path / "migrated.yaml"
    legacy_path.write_text(
        "region: seattle\nstart_year: 2018\nend_year: 2019\ntravel_model: beam\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/migrate_config.py",
            str(legacy_path),
            str(output_path),
            "--no-validate",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(
            str(Path("scripts/migrate_config.py")),
            run_name="__main__",
        )

    assert excinfo.value.code == 0
    assert output_path.exists()
