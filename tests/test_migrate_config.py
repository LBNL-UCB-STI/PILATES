from scripts.migrate_config import ConfigMigrator


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
