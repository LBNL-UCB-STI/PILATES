from pathlib import Path
import subprocess

import pytest
import yaml

from pilates.config.models import PilatesConfig, load_config, resolve_config_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SCENARIO_ROOTS = (
    REPO_ROOT / "scenarios/breathe",
    REPO_ROOT / "scenarios/seattle",
    REPO_ROOT / "scenarios/sfbay",
)


def _tracked_yaml_paths(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", str(root / "*.yaml")],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return sorted(root.glob("*.yaml"))
    return sorted(
        REPO_ROOT / line for line in result.stdout.splitlines() if line.strip()
    )


ACTIVE_SCENARIO_PATHS = sorted(
    path for root in ACTIVE_SCENARIO_ROOTS for path in _tracked_yaml_paths(root)
)


@pytest.fixture(autouse=True)
def _set_inmap_path_for_scenario_config_tests(monkeypatch):
    monkeypatch.setenv("IMPACTS_INMAP_DATA_DIR", "/tmp/impacts-inmap-test")


SEATTLE_ROUTER_DIRECTORY = "r5/seattle-cbg120-ferry-weakConn-network"
SFBAY_ROUTER_DIRECTORY = "r5/sfbay-cbg5500-weakConn-network"
BEAM_IMAGE = "ghcr.io/lbnl-ucb-sti/beam:1.1.2"

EXPECTED_WARMSTART_PATHS = {
    REPO_ROOT / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml": (
        "{router_directory}/init.linkstats.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC06-0.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/5.linkstats.FC06.v1.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC07-0.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/4.linkstats.FC07.v1.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC08-0.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/5.linkstats.FC08.v1.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--fy26-task1--2018-Baseline.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/5.linkstats.FC07.v3.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-BaseFuel-060Elec.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/5.linkstats.FC07.v3.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-BaseFuel-100Elec.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/5.linkstats.FC07.v3.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-HighFuel-100Elec.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/5.linkstats.FC07.v3.csv.gz"
    ),
    REPO_ROOT / "scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC06-0.yaml": (
        "r5/sfbay-cbg5500-weakConn-network/2018-baseline-linkstats/3.linkstats.FC06.v1.csv.gz"
    ),
    REPO_ROOT / "scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC07-0.yaml": (
        "r5/sfbay-cbg5500-weakConn-network/2018-baseline-linkstats/3.linkstats.FC07.v1.csv.gz"
    ),
    REPO_ROOT / "scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC08-0.yaml": (
        "r5/sfbay-cbg5500-weakConn-network/2018-baseline-linkstats/3.linkstats.FC08.v1.csv.gz"
    ),
}


def _load_raw_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def test_all_active_scenarios_use_hierarchical_schema_and_validate():
    assert ACTIVE_SCENARIO_PATHS, "Expected active scenario configs under scenarios/"

    for path in ACTIVE_SCENARIO_PATHS:
        raw_data = _load_raw_yaml(path)
        assert "run" in raw_data or "base" in raw_data, (
            f"{path} is not using the hierarchical schema"
        )
        data = resolve_config_dict(path)
        assert "run" in data, f"{path} resolved without a 'run' block"
        PilatesConfig(**data)


def test_active_scenarios_load_via_runtime_config_loader():
    for path in ACTIVE_SCENARIO_PATHS:
        settings = load_config(str(path))
        assert settings.run.region in {"seattle", "sfbay"}
        assert settings.run.scenario
        assert settings.beam is not None
        assert settings.beam.router_directory


def test_default_and_active_scenarios_pin_beam_images():
    defaults = _load_raw_yaml(REPO_ROOT / "settings.yaml")
    assert defaults["docker_images"]["beam"] == BEAM_IMAGE
    assert defaults["singularity_images"]["beam"] == f"docker://{BEAM_IMAGE}"

    for path in ACTIVE_SCENARIO_PATHS:
        infra = resolve_config_dict(path)["infrastructure"]
        assert infra["docker_images"]["beam"] == BEAM_IMAGE, path
        assert infra["singularity_images"]["beam"] == f"docker://{BEAM_IMAGE}", path


def test_active_scenarios_use_expected_router_directories():
    seattle_paths = _tracked_yaml_paths(REPO_ROOT / "scenarios/seattle")
    sfbay_paths = _tracked_yaml_paths(REPO_ROOT / "scenarios/sfbay")
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
    sfbay_paths.append(
        REPO_ROOT / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml"
    )

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
        assert settings.shared.geography.alternative_zones.canonical_id_col == "taz1454"
        assert (
            settings.shared.geography.alternative_zones.activitysim_index_col == "TAZ"
        )


@pytest.mark.skip
def test_active_scenarios_use_expected_warmstart_paths():
    for path in ACTIVE_SCENARIO_PATHS:
        warmstart_path = resolve_config_dict(path)["beam"]["warmstart_linkstats_path"]
        expected = EXPECTED_WARMSTART_PATHS.get(path)
        assert warmstart_path == expected, (
            f"{path} warmstart mismatch: expected {expected!r}, got {warmstart_path!r}"
        )
