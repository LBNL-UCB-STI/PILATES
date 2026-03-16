from pathlib import Path

import yaml

from pilates.config.models import PilatesConfig, load_config
from pilates.utils.settings_helper import get as get_setting


def test_all_active_scenarios_use_hierarchical_schema_and_validate():
    """Active runtime scenarios should use the current hierarchical schema."""
    scenario_paths = sorted(Path("scenarios").rglob("*.yaml"))
    active_paths = [
        path
        for path in scenario_paths
        if "archive" not in path.parts
        and "old-settings" not in path.parts
        and "_legacy" not in path.parts
    ]

    assert active_paths, "Expected active scenario configs under scenarios/"

    for path in active_paths:
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), f"{path} did not parse to a mapping"
        assert "run" in data, f"{path} is not using the hierarchical schema"
        PilatesConfig(**data)


def test_active_scenarios_load_via_runtime_config_loader():
    """Only active runtime scenarios must load through the current config loader."""
    scenario_paths = sorted(Path("scenarios").rglob("*.yaml"))
    active_paths = [
        path
        for path in scenario_paths
        if "archive" not in path.parts
        and "old-settings" not in path.parts
        and "_legacy" not in path.parts
    ]

    for path in active_paths:
        settings = load_config(str(path))
        assert settings.run.region in {"seattle", "sfbay"}
        assert settings.run.scenario
        assert settings.beam is not None
        assert get_setting(settings, "beam.router_directory")


def test_active_seattle_scenarios_use_expected_router_directory():
    seattle_paths = sorted(Path("scenarios/seattle").glob("*.yaml"))

    assert seattle_paths, "Expected active Seattle scenario configs"

    for path in seattle_paths:
        settings = load_config(str(path))
        assert settings.run.region == "seattle"
        assert settings.beam is not None
        assert settings.beam.router_directory == "r5/seattle-cbg120-ferry-weakConn-network"


def test_active_sfbay_scenarios_use_expected_router_directory():
    sfbay_paths = sorted(Path("scenarios/sfbay").glob("*.yaml"))
    sfbay_paths.append(Path("scenarios/breathe/settings--sfbay--2018-Baseline.yaml"))

    for path in sfbay_paths:
        settings = load_config(str(path))
        assert settings.run.region == "sfbay"
        assert settings.beam is not None
        assert settings.beam.router_directory == "r5/sfbay-cbg5500-weakConn-network"


def test_active_seattle_scenarios_define_primary_and_fallback_zone_sources():
    seattle_paths = sorted(Path("scenarios/seattle").glob("*.yaml"))

    for path in seattle_paths:
        settings = load_config(str(path))
        zones = settings.shared.geography.zones
        assert zones is not None
        assert (
            zones.source_file
            == "pilates/activitysim/data/seattle/block_groups_seattle_4326.geojson"
        )
        assert zones.source_crs == "EPSG:4326"
        assert settings.shared.geography.alternative_zones is not None
        assert (
            settings.shared.geography.alternative_zones.source_file
            == "pilates/beam/production/seattle/shape/block-groups-32048.shp"
        )
        assert settings.shared.geography.alternative_zones.source_crs == "EPSG:32048"
        assert (
            settings.shared.geography.alternative_zones.canonical_id_col == "OBJECTID"
        )
        assert (
            settings.shared.geography.alternative_zones.activitysim_index_col == "TAZ"
        )


def test_active_sfbay_scenarios_define_expected_zone_sources():
    sfbay_paths = sorted(Path("scenarios/sfbay").glob("*.yaml"))
    sfbay_paths.append(Path("scenarios/breathe/settings--sfbay--2018-Baseline.yaml"))

    for path in sfbay_paths:
        settings = load_config(str(path))
        zones = settings.shared.geography.zones
        assert zones is not None
        assert zones.source_file == "pilates/activitysim/data/sfbay/taz_sfbay.geojson"
        assert zones.source_crs == "EPSG:4326"
        assert settings.shared.geography.alternative_zones is not None
        assert (
            settings.shared.geography.alternative_zones.source_file
            == "pilates/beam/production/sfbay/shape/sfbay-tazs-epsg-26910.shp"
        )
        assert settings.shared.geography.alternative_zones.source_crs == "EPSG:26910"
        assert (
            settings.shared.geography.alternative_zones.canonical_id_col == "taz1454"
        )
        assert (
            settings.shared.geography.alternative_zones.activitysim_index_col == "TAZ"
        )


def test_legacy_scenario_copies_preserve_pre_migration_configs():
    """
    Migrated scenarios keep a `_legacy/` copy for historical reference only.

    This is a repository preservation contract, not a runtime dependency.
    The active scenarios above are the ones that must load and execute.
    """
    migrated_active_paths = [
        Path("scenarios/breathe/settings--sfbay--2018-Baseline.yaml"),
        Path("scenarios/seattle/pilates-run--seattle--fullskim-FC07-0.yaml"),
        Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC05-0.yaml"),
        Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC06-0.yaml"),
        Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC07-0.yaml"),
        Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC07-5.yaml"),
        Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC08-0.yaml"),
        Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC08-5.yaml"),
        Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC09-0.yaml"),
        Path("scenarios/seattle/run--seattle-pilates--20250721--2018-Baseline.yaml"),
        Path("scenarios/seattle/settings-new-asim-seattle-null.yaml"),
        Path("scenarios/seattle/settings-new-asim-seattle.yaml"),
        Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC05-0.yaml"),
        Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC06-0.yaml"),
        Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC07-0.yaml"),
        Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC07-5.yaml"),
        Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC08-0.yaml"),
        Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC08-5.yaml"),
        Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC09-0.yaml"),
        Path("scenarios/sfbay/run--sfbay-pilates--20250730--2018-Baseline.yaml"),
    ]

    for active_path in migrated_active_paths:
        legacy_path = Path("scenarios/_legacy") / active_path.relative_to("scenarios")
        assert legacy_path.exists(), f"Missing legacy copy for {active_path}"

        legacy = yaml.safe_load(legacy_path.read_text())
        assert isinstance(legacy, dict), f"{legacy_path} did not parse to a mapping"
        assert "run" not in legacy, (
            f"{legacy_path} should remain in the legacy schema as a preserved "
            "pre-migration reference"
        )
