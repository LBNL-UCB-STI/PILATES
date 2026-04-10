from pathlib import Path

import yaml

from pilates.config.models import PilatesConfig, load_config


ACTIVE_SCENARIO_ROOTS = (
    Path("scenarios/breathe"),
    Path("scenarios/seattle"),
    Path("scenarios/sfbay"),
)

ACTIVE_SCENARIO_PATHS = sorted(
    path
    for root in ACTIVE_SCENARIO_ROOTS
    for path in root.glob("*.yaml")
)

SEATTLE_ROUTER_DIRECTORY = "r5/seattle-cbg120-ferry-weakConn-network"
SFBAY_ROUTER_DIRECTORY = "r5/sfbay-cbg5500-weakConn-network"
BEAM_SINGULARITY_IMAGE = "docker://haitamlaarabi/beam:1.1-beta-v260408"
BEAM_DOCKER_IMAGE = "haitamlaarabi/beam:1.1-beta-v260408"

EXPECTED_WARMSTART_PATHS = {
    Path("scenarios/breathe/settings--sfbay--2018-Baseline.yaml"): (
        "{router_directory}/init.linkstats.csv.gz"
    ),
    Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC06-0.yaml"): (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC06.v1.csv.gz"
    ),
    Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC07-0.yaml"): (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "4.linkstats.FC07.v1.csv.gz"
    ),
    Path("scenarios/seattle/pilates-run--seattle--jdeq-calibration-FC08-0.yaml"): (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC08.v1.csv.gz"
    ),
    Path("scenarios/seattle/pilates-run--seattle--fy26-task1--2018-Baseline.yaml"): (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC07.v3.csv.gz"
    ),
    Path(
        "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-BaseFuel-060Elec.yaml"
    ): (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC07.v3.csv.gz"
    ),
    Path(
        "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-BaseFuel-100Elec.yaml"
    ): (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC07.v3.csv.gz"
    ),
    Path(
        "scenarios/seattle/pilates-run--seattle--fy26-task1--2050-HighFuel-100Elec.yaml"
    ): (
        "r5/seattle-cbg120-ferry-weakConn-network/2018-baseline-linkstats/"
        "5.linkstats.FC07.v3.csv.gz"
    ),
    Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC06-0.yaml"): (
        "r5/sfbay-cbg5500-weakConn-network/2018-baseline-linkstats/"
        "3.linkstats.FC06.v1.csv.gz"
    ),
    Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC07-0.yaml"): (
        "r5/sfbay-cbg5500-weakConn-network/2018-baseline-linkstats/"
        "3.linkstats.FC07.v1.csv.gz"
    ),
    Path("scenarios/sfbay/pilates-run--sfbay--jdeq-calibration-FC08-0.yaml"): (
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
        assert infra["singularity_images"]["beam"] == BEAM_SINGULARITY_IMAGE
        assert infra["docker_images"]["beam"] == BEAM_DOCKER_IMAGE
        assert "warmstart_linkstats_path" in beam


def test_active_scenarios_use_expected_router_directories():
    seattle_paths = sorted(Path("scenarios/seattle").glob("*.yaml"))
    sfbay_paths = sorted(Path("scenarios/sfbay").glob("*.yaml"))
    breathe_path = Path("scenarios/breathe/settings--sfbay--2018-Baseline.yaml")

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
    seattle_paths = sorted(Path("scenarios/seattle").glob("*.yaml"))
    sfbay_paths = sorted(Path("scenarios/sfbay").glob("*.yaml"))
    breathe_path = Path("scenarios/breathe/settings--sfbay--2018-Baseline.yaml")

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


def test_archive_scenarios_remain_legacy_schema():
    archive_paths = sorted(Path("scenarios/archive").glob("*.yaml"))
    assert archive_paths, "Expected legacy archive configs under scenarios/archive"

    for path in archive_paths:
        data = _load_raw_yaml(path)
        assert "run" not in data, (
            f"{path} should remain in the legacy flat schema as an archive reference"
        )
