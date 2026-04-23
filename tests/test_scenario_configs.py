from pathlib import Path

import yaml

from pilates.config.models import PilatesConfig, load_config
from pilates.utils.settings_helper import get as get_setting


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SCENARIO_ROOTS = (
    REPO_ROOT / "scenarios/breathe",
    REPO_ROOT / "scenarios/seattle",
    REPO_ROOT / "scenarios/sfbay",
)
ACTIVE_SCENARIO_PATHS = sorted(
    path
    for root in ACTIVE_SCENARIO_ROOTS
    for path in root.glob("*.yaml")
)

SEATTLE_ROUTER_DIRECTORY = "r5/seattle-cbg120-ferry-weakConn-network"
SFBAY_ROUTER_DIRECTORY = "r5/sfbay-cbg5500-weakConn-network"


def _load_raw_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def test_all_active_scenarios_use_hierarchical_schema_and_validate():
    assert ACTIVE_SCENARIO_PATHS, "Expected active scenario configs under scenarios/"

    for path in ACTIVE_SCENARIO_PATHS:
        data = _load_raw_yaml(path)
        assert "run" in data, f"{path} is not using the hierarchical schema"
        PilatesConfig(**data)


def test_active_scenarios_load_via_runtime_config_loader():
    for path in ACTIVE_SCENARIO_PATHS:
        settings = load_config(str(path))
        assert settings.run.region in {"seattle", "sfbay"}
        assert settings.run.scenario
        assert settings.beam is not None
        assert get_setting(settings, "beam.router_directory")


def test_active_scenarios_use_expected_router_directories():
    seattle_paths = sorted((REPO_ROOT / "scenarios/seattle").glob("*.yaml"))
    sfbay_paths = sorted((REPO_ROOT / "scenarios/sfbay").glob("*.yaml"))
    breathe_path = REPO_ROOT / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml"

    for path in seattle_paths:
        settings = load_config(str(path))
        assert settings.run.region == "seattle"
        assert settings.beam is not None
        assert settings.beam.router_directory == SEATTLE_ROUTER_DIRECTORY

    for path in [*sfbay_paths, breathe_path]:
        settings = load_config(str(path))
        assert settings.run.region == "sfbay"
        assert settings.beam is not None
        assert settings.beam.router_directory == SFBAY_ROUTER_DIRECTORY


def test_active_seattle_scenarios_define_primary_and_fallback_zone_sources():
    seattle_paths = sorted((REPO_ROOT / "scenarios/seattle").glob("*.yaml"))

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
    sfbay_paths = sorted((REPO_ROOT / "scenarios/sfbay").glob("*.yaml"))
    sfbay_paths.append(REPO_ROOT / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml")

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
