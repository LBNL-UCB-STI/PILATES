from pathlib import Path
from typing import Any

import pytest
import yaml

from pilates.config import PilatesConfig, load_config, resolve_config_dict
from pilates.config.models import main as config_cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO_ROOT / "scenarios"


def _minimal_base_dict() -> dict[str, Any]:
    return {
        "run": {
            "region": "seattle",
            "scenario": "base",
            "start_year": 2018,
            "end_year": 2019,
            "supply_demand_iters": 4,
            "output_directory": "/scratch/pilates-outputs",
            "output_run_name": "regional-base",
            "local_workspace_root": "/local/scratch/workspace",
            "use_stubs": False,
            "models": {
                "land_use": "urbansim",
                "travel": "beam",
                "activity_demand": "activitysim",
                "vehicle_ownership": None,
            },
        },
        "shared": {
            "geography": {
                "FIPS": {
                    "state": "53",
                    "counties": ["061", "035", "033", "053"],
                },
                "local_crs": "EPSG:32048",
            },
            "skims": {
                "fname": "as-base-skims-seattle-bg.omx",
                "periods": ["EA", "AM", "MD", "PM", "EV"],
            },
            "database": {
                "enabled": True,
                "type": "duckdb",
                "path": "/scratch/provenance.duckdb",
            },
        },
        "infrastructure": {
            "container_manager": "singularity",
            "singularity_images": {
                "urbansim": "docker://jdcaicedo/urbansim_demos:v0.1.1",
                "activitysim": "docker://zaneedell/activitysim:0.2.18",
                "beam": "docker://zaneedell/beam:1.1-beta-v261009",
            },
            "docker_images": {
                "urbansim": "jdcaicedo/urbansim_demos:v0.1.1",
                "activitysim": "zaneedell/activitysim:0.2.18",
                "beam": "zaneedell/beam:1.1-beta-v261009",
            },
        },
        "urbansim": {
            "region_id": "53199100",
            "local_data_input_folder": "pilates/urbansim/data/",
            "local_mutable_data_folder": "urbansim/data/",
            "client_base_folder": "/base/demos_urbansim",
            "client_data_folder": "/base/demos_urbansim/data/",
            "input_file_template": "custom_mpo_{region_id}_model_data.h5",
            "input_file_template_year": (
                "custom_mpo_{region_id}_model_data_{start_year}.h5"
            ),
            "output_file_template": "model_data_{year}.h5",
            "command_template": "-r {0} -i {1} -y {2} -f {3} -t {4}",
        },
        "activitysim": {
            "household_sample_size": 250000,
            "num_processes": 25,
            "local_input_folder": "pilates/activitysim/data/",
            "local_mutable_data_folder": "activitysim/data/",
            "local_output_folder": "activitysim/output/",
            "local_configs_folder": "pilates/activitysim/configs/",
            "local_mutable_configs_folder": "activitysim/configs/",
            "validation_folder": "pilates/activitysim/validation",
            "command_template": "--households_sample_size {0} -m {1} -g {2}",
            "final_plans_folder": "pilates/activitysim/output/final_plans",
        },
        "beam": {
            "config": "seattle-pilates-base-nofreight.conf",
            "sample": 1.0,
            "local_input_folder": "pilates/beam/production/",
            "local_mutable_data_folder": "beam/input/",
            "local_output_folder": "beam/beam_output/",
            "scenario_folder": "urbansim/",
            "router_directory": "r5/seattle-cbg120-ferry-weakConn-network",
            "skims_shapefile": "shape/block-groups-32048.shp",
            "skim_zone_source_id_col": "sk_zone",
            "skim_zone_geoid_col": "geoid10",
        },
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_single_level_inheritance_and_load_config(tmp_path: Path) -> None:
    base_file = _write_yaml(tmp_path / "base.yaml", _minimal_base_dict())
    child_file = _write_yaml(
        tmp_path / "child.yaml",
        {
            "base": base_file.name,
            "run": {
                "output_directory": "/Users/dev/pilates/tmp",
                "output_run_name": "local-dev",
                "supply_demand_iters": 1,
                "use_stubs": True,
            },
        },
    )

    resolved = resolve_config_dict(child_file)
    assert "base" not in resolved
    assert resolved["run"]["region"] == "seattle"
    assert resolved["run"]["output_directory"] == "/Users/dev/pilates/tmp"
    assert resolved["run"]["output_run_name"] == "local-dev"
    assert resolved["run"]["supply_demand_iters"] == 1
    assert resolved["run"]["use_stubs"] is True

    cfg = load_config(str(child_file))
    assert cfg.run.region == "seattle"
    assert cfg.run.output_directory == "/Users/dev/pilates/tmp"
    assert cfg.run.supply_demand_iters == 1
    assert cfg.run.use_stubs is True


def test_multi_level_chained_inheritance_and_relative_parent_paths(
    tmp_path: Path,
) -> None:
    regional_base = _write_yaml(
        tmp_path / "common" / "regional_base.yaml",
        _minimal_base_dict(),
    )
    scenario_base = _write_yaml(
        tmp_path / "scenarios" / "seattle" / "scenario_base.yaml",
        {
            "base": "../../common/regional_base.yaml",
            "run": {
                "scenario": "high_transit",
                "end_year": 2025,
            },
            "beam": {
                "config": "seattle-high-transit.conf",
            },
        },
    )
    variant = _write_yaml(
        tmp_path / "scenarios" / "seattle" / "variants" / "local_variant.yaml",
        {
            "base": "../scenario_base.yaml",
            "run": {
                "output_directory": "/tmp/local-high-transit",
                "supply_demand_iters": 2,
            },
        },
    )

    assert regional_base.exists()
    assert scenario_base.exists()
    resolved = resolve_config_dict(variant)
    assert resolved["run"]["region"] == "seattle"
    assert resolved["run"]["scenario"] == "high_transit"
    assert resolved["run"]["start_year"] == 2018
    assert resolved["run"]["end_year"] == 2025
    assert resolved["run"]["supply_demand_iters"] == 2
    assert resolved["run"]["output_directory"] == "/tmp/local-high-transit"
    assert resolved["beam"]["config"] == "seattle-high-transit.conf"
    assert resolved["beam"]["sample"] == 1.0


def test_deep_merge_list_replacement_dict_recursion_and_explicit_null(
    tmp_path: Path,
) -> None:
    base_file = _write_yaml(tmp_path / "base.yaml", _minimal_base_dict())
    child_file = _write_yaml(
        tmp_path / "child.yaml",
        {
            "base": "base.yaml",
            "run": {
                "local_workspace_root": None,
                "models": {
                    "land_use": None,
                },
            },
            "shared": {
                "geography": {
                    "FIPS": {
                        "counties": ["061"],
                    },
                },
            },
        },
    )

    resolved = resolve_config_dict(child_file)
    # Lists replace rather than concatenate
    assert resolved["shared"]["geography"]["FIPS"]["counties"] == ["061"]
    # Sibling keys in nested dicts are preserved
    assert resolved["shared"]["geography"]["FIPS"]["state"] == "53"
    assert resolved["shared"]["geography"]["local_crs"] == "EPSG:32048"
    assert resolved["run"]["models"]["travel"] == "beam"
    assert resolved["run"]["models"]["activity_demand"] == "activitysim"
    # Explicit null overrides non-null base values
    assert resolved["run"]["local_workspace_root"] is None
    assert resolved["run"]["models"]["land_use"] is None


def test_cycle_detection_raises_value_error(tmp_path: Path) -> None:
    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    _write_yaml(config_a, {"base": "b.yaml", "run": {"region": "seattle"}})
    _write_yaml(config_b, {"base": "a.yaml", "run": {"scenario": "cycle"}})

    with pytest.raises(ValueError, match="Circular configuration inheritance detected"):
        resolve_config_dict(config_a)

    self_cycle = _write_yaml(tmp_path / "self.yaml", {"base": "self.yaml"})
    with pytest.raises(ValueError, match="Circular configuration inheritance detected"):
        resolve_config_dict(self_cycle)


def test_missing_base_raises_file_not_found_with_context(tmp_path: Path) -> None:
    child_file = _write_yaml(
        tmp_path / "child.yaml",
        {
            "base": "does-not-exist-base.yaml",
            "run": {"region": "seattle"},
        },
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_config_dict(child_file)

    message = str(exc_info.value)
    assert "Base config not found: does-not-exist-base.yaml" in message
    assert str(child_file) in message


def test_backward_compatibility_without_base_and_pilates_config_base_field(
    tmp_path: Path,
) -> None:
    raw = _minimal_base_dict()
    config_file = _write_yaml(tmp_path / "standalone.yaml", raw)

    resolved = resolve_config_dict(config_file)
    assert resolved == raw

    with_base_dict = {**raw, "base": "non_existent_optional_marker.yaml"}
    validated = PilatesConfig(**with_base_dict)
    assert validated.run.region == "seattle"
    assert "base" not in validated.model_dump()


def test_cli_inspection_helper_outputs_resolved_yaml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_file = _write_yaml(tmp_path / "base.yaml", _minimal_base_dict())
    child_file = _write_yaml(
        tmp_path / "child.yaml",
        {
            "base": f"  {base_file.name}  ",
            "run": {"supply_demand_iters": 9},
        },
    )

    exit_code = config_cli_main([str(child_file), "--validate"])
    assert exit_code == 0
    captured = capsys.readouterr()
    dumped = yaml.safe_load(captured.out)
    assert "base" not in dumped
    assert dumped["run"]["supply_demand_iters"] == 9
    assert dumped["run"]["region"] == "seattle"


def _tracked_scenario_paths() -> list[Path]:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "ls-files", "scenarios/**/*.yaml"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        paths = [
            REPO_ROOT / line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        if paths:
            return sorted(paths)
    except Exception:
        pass
    return sorted(SCENARIOS_DIR.rglob("*.yaml"))


def test_all_tracked_scenarios_resolve_and_validate() -> None:
    scenario_files = _tracked_scenario_paths()
    assert len(scenario_files) >= 27

    hierarchical_count = 0
    for path in scenario_files:
        resolved = resolve_config_dict(path)
        assert isinstance(resolved, dict), f"{path} did not resolve to a dict"
        assert "base" not in resolved, f"{path} retained top-level 'base' key"

        # Legacy flat files without a 'run' block are unmigrated snapshots;
        # all hierarchical scenarios must validate cleanly into PilatesConfig.
        if "run" in resolved:
            hierarchical_count += 1
            cfg = load_config(path)
            dumped = cfg.model_dump()
            assert dumped["run"]["region"] == resolved["run"]["region"]

    assert hierarchical_count >= 27


def test_pilot_refactored_scenarios_preserve_exact_resolved_configuration() -> None:
    # 1. Seattle BEAM HPC variant -> inherits from settings-seattle-consist-hpc.yaml
    seattle_hpc_path = SCENARIOS_DIR / "seattle" / "settings-seattle-consist-hpc.yaml"
    seattle_beam_hpc_path = (
        SCENARIOS_DIR / "seattle" / "settings-seattle-consist-beam-hpc.yaml"
    )
    raw_beam_hpc = yaml.safe_load(seattle_beam_hpc_path.read_text(encoding="utf-8"))
    assert raw_beam_hpc["base"] == "settings-seattle-consist-hpc.yaml"
    assert len(seattle_beam_hpc_path.read_text(encoding="utf-8").splitlines()) < 25

    expected_beam_hpc = resolve_config_dict(seattle_hpc_path)
    expected_beam_hpc["run"]["models"]["activity_demand"] = None
    expected_beam_hpc["infrastructure"]["singularity_images"]["beam"] = (
        "docker://zaneedell/beam:1.1-beta-v260615"
    )
    expected_beam_hpc["infrastructure"]["docker_images"]["beam"] = (
        "zaneedell/beam:0.9.10"
    )
    expected_beam_hpc["beam"]["config"] = (
        "scenarios/network-calibration/"
        "beam-pilates-run--seattle--jdeq-calibration-FC08-0.conf"
    )
    assert resolve_config_dict(seattle_beam_hpc_path) == expected_beam_hpc
    assert (
        load_config(seattle_beam_hpc_path).model_dump()
        == PilatesConfig(**expected_beam_hpc).model_dump()
    )

    # 2. SFBay local newconfig -> inherits from settings-sfbay-newconfig-hpc.yaml
    sfbay_hpc_path = SCENARIOS_DIR / "sfbay" / "settings-sfbay-newconfig-hpc.yaml"
    sfbay_local_path = SCENARIOS_DIR / "sfbay" / "settings-sfbay-newconfig-local.yaml"
    raw_sfbay_local = yaml.safe_load(sfbay_local_path.read_text(encoding="utf-8"))
    assert raw_sfbay_local["base"] == "settings-sfbay-newconfig-hpc.yaml"
    assert len(sfbay_local_path.read_text(encoding="utf-8").splitlines()) < 20

    expected_sfbay_local = resolve_config_dict(sfbay_hpc_path)
    expected_sfbay_local["run"]["start_year"] = 2018
    expected_sfbay_local["run"]["end_year"] = 2019
    expected_sfbay_local["run"]["output_directory"] = "/Users/zaneedell/git/PILATES/tmp"
    expected_sfbay_local["run"]["output_run_name"] = "new-asim-fast-beam-bigger-pool"
    expected_sfbay_local["urbansim"]["region_id"] = "53199100"
    assert resolve_config_dict(sfbay_local_path) == expected_sfbay_local
    assert (
        load_config(sfbay_local_path).model_dump()
        == PilatesConfig(**expected_sfbay_local).model_dump()
    )
