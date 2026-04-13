from pathlib import Path

import yaml

from pilates.config.models import PilatesConfig, load_config


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
DEFAULT_BEAM_SINGULARITY_IMAGE = "docker://haitamlaarabi/beam:1.1-beta-v260408"
DEFAULT_BEAM_DOCKER_IMAGE = "haitamlaarabi/beam:1.1-beta-v260408"
BREATHE_BEAM_SINGULARITY_IMAGE = "docker://haitamlaarabi/beam:1.1-beta-v260329"
BREATHE_BEAM_DOCKER_IMAGE = "haitamlaarabi/beam:1.1-beta-v260329"

EXPECTED_WARMSTART_PATHS = {
    REPO_ROOT / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml": (
        "{router_directory}/init.linkstats.csv.gz"
    ),
    REPO_ROOT / "scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC06-0.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC06.v1.csv.gz"
    ),
    REPO_ROOT / "scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC07-0.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "4.linkstats.FC07.v1.csv.gz"
    ),
    REPO_ROOT / "scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC08-0.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC08.v1.csv.gz"
    ),
    REPO_ROOT / "scenarios/seattle/pilates-run--seattle--fy26-task1--2018-Baseline.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC07.v3.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-BaseFuel-060Elec.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC07.v3.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-BaseFuel-100Elec.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC07.v3.csv.gz"
    ),
    REPO_ROOT
    / "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-HighFuel-100Elec.yaml": (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC07.v3.csv.gz"
    ),
    REPO_ROOT / "scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC06-0.yaml": (
        "r5/sfbay-cbg5500-weakConn-network/2018-baseline-linkstats/"
        "3.linkstats.FC06.v1.csv.gz"
    ),
    REPO_ROOT / "scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC07-0.yaml": (
        "r5/sfbay-cbg5500-weakConn-network/2018-baseline-linkstats/"
        "3.linkstats.FC07.v1.csv.gz"
    ),
    REPO_ROOT / "scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC08-0.yaml": (
        "r5/sfbay-cbg5500-weakConn-network/2018-baseline-linkstats/"
        "3.linkstats.FC08.v1.csv.gz"
    ),
}


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
        assert settings.beam.router_directory


def test_active_scenarios_pin_beam_images_and_explicit_warmstart_setting():
    for path in ACTIVE_SCENARIO_PATHS:
        data = _load_raw_yaml(path)
        infra = data["infrastructure"]
        beam = data["beam"]
        if path == REPO_ROOT / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml":
            expected_singularity = BREATHE_BEAM_SINGULARITY_IMAGE
            expected_docker = BREATHE_BEAM_DOCKER_IMAGE
        else:
            expected_singularity = DEFAULT_BEAM_SINGULARITY_IMAGE
            expected_docker = DEFAULT_BEAM_DOCKER_IMAGE
        assert infra["singularity_images"]["beam"] == expected_singularity
        assert infra["docker_images"]["beam"] == expected_docker
        assert "warmstart_linkstats_path" in beam


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


def test_active_scenarios_define_expected_zone_sources():
    seattle_paths = sorted((REPO_ROOT / "scenarios/seattle").glob("*.yaml"))
    sfbay_paths = sorted((REPO_ROOT / "scenarios/sfbay").glob("*.yaml"))
    breathe_path = REPO_ROOT / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml"

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

    for path in [*sfbay_paths, breathe_path]:
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


def test_active_scenarios_use_expected_warmstart_paths():
    for path in ACTIVE_SCENARIO_PATHS:
        warmstart_path = _load_raw_yaml(path)["beam"]["warmstart_linkstats_path"]
        expected = EXPECTED_WARMSTART_PATHS.get(path)
        assert warmstart_path == expected, (
            f"{path} warmstart mismatch: expected {expected!r}, got {warmstart_path!r}"
        )


def test_archive_scenarios_have_been_retired():
    archive_paths = sorted((REPO_ROOT / "scenarios/archive").glob("*.yaml"))
    assert archive_paths == []
