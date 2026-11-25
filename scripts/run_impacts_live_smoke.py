"""Run a small San Francisco IMPACTS → BEAM → IMPACTS workflow in Docker."""

from __future__ import annotations

import argparse
from filecmp import cmp
import logging
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from pilates.config import resolve_config_dict, validate_config
from pilates.generic.runner import GenericRunner
from pilates.impacts.settings_bridge import (
    BEAM_INPUT_CONTAINER_DIR,
    WORKSPACE_CONTAINER_DIR,
)
from pilates.runtime.launcher import main as run_pilates
from scripts.run_impacts_container_smoke import (
    check_dispersion_outputs,
    check_generated_fleet_files,
    check_inventory_totals,
    require_docker,
    run_logged_container,
    smoke_run_directory,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/impacts_live_sf"
BASE_SETTINGS = ROOT / "scenarios/breathe/settings--sfbay--2018-Baseline.yaml"
RUN_NAME = "impacts-live-sf"
YEAR = 2018
RUN_LABEL = f"{RUN_NAME}_y{YEAR}"
PRESIM_ACTIVITIES = Path(
    f"impacts/output/impacts_presim--sfbay--{RUN_LABEL}/activities"
)
logger = logging.getLogger(__name__)


def run_live_smoke(root: Path, *, impacts_image: str) -> None:
    docker = require_docker()
    workspace = root / f"pilates-run--sfbay--{RUN_NAME}"
    config = resolve_config_dict(BASE_SETTINGS)
    config["infrastructure"]["container_manager"] = "docker"
    config["infrastructure"]["docker_images"]["impacts"] = impacts_image
    beam_image = config["infrastructure"]["docker_images"]["beam"]
    config["run"].update(
        output_directory=str(root),
        output_run_name=RUN_NAME,
        scenario=RUN_NAME,
        start_year=YEAR,
        end_year=YEAR,
        supply_demand_iters=1,
        models={"travel": "beam", "impacts": "impacts"},
        use_stubs=False,
        local_workspace_root=None,
        enable_archive_copy=False,
    )
    config["shared"]["geography"].update(
        FIPS={"state": "06", "counties": ["075"]},
        local_crs="EPSG:26910",
        zones=None,
        alternative_zones=None,
    )
    config["beam"].update(
        local_input_folder=str(FIXTURE / "beam"),
        local_output_folder="beam/output",
        config="scenarios/breathe/sf-live.conf",
        router_directory="r5/sf-smoke",
        admission=None,
        warmstart_linkstats_path=None,
        memory="3g",
        extra_jvm_args="-XX:ActiveProcessorCount=4",
    )
    config["postprocessing"] = None
    config["impacts"].update(
        local_input_folder="impacts/input",
        local_output_folder="impacts/output",
        local_inmap_data_folder=str(FIXTURE / "inmap"),
        use_generated_fleet=True,
        population_fallback_to_beam_inputs=False,
        overrides={
            "analysis": {"targets": None, "inventory_targets": None},
            "emissions": {
                "enable_passenger_activity_correction": False,
                "enable_freight_activity_correction": False,
            },
            "population": {"population_sample": 1.0},
            "activities": {
                "region_label": "SF",
                "emissions_raw_archive": "vehicle-tech/emissions_raw.tar.zst",
            },
            "dispersions": {
                "inmap": {
                    "isrm_zarr": "/app/inmap/isrm_sample.zarr",
                    "isrm_nox_to_no2_ratios_file": "vehicle-tech/dispersions/nox_to_no2_sample.npz",
                    "grid_path": "vehicle-tech/dispersions/inmap_cells.parquet",
                    "grid_id": "isrm",
                    "grid_epsg": 26910,
                },
                "aermod": {
                    "asrv_patterns_file": "vehicle-tech/dispersions/aermod_patterns.parquet",
                    "asrv_patterns_epsg": 26910,
                },
            },
        },
    )
    settings = validate_config(config)

    def run_docker(*, image, volumes, command, working_dir, environment=None, **_):
        args = [
            docker,
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--cpus",
            "4",
            "--workdir",
            working_dir,
        ]
        if image == beam_image:
            # Java resolves the container hostname even when networking is disabled.
            args += ["--hostname", "beam-smoke", "--add-host", "beam-smoke:127.0.0.1"]
        else:
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
            args += ["--env", "MPLCONFIGDIR=/tmp/matplotlib"]
        mounts = GenericRunner._extract_mounts(volumes)
        for host, target, mode in mounts:
            readonly = ",readonly" if mode == "ro" else ""
            args += ["--mount", f"type=bind,src={host},dst={target}{readonly}"]
        for key, value in (environment or {}).items():
            args += ["--env", f"{key}={value}"]
        container_command = GenericRunner._build_full_command(command)
        if container_command[0] == "postsim":
            host = next(
                Path(host)
                for host, target, _ in mounts
                if target == WORKSPACE_CONTAINER_DIR
            )
            output = Path(environment["IMPACTS_POSTSIM_OUTPUT_DIR"]).relative_to(
                WORKSPACE_CONTAINER_DIR
            )
            shutil.copytree(
                FIXTURE / "county", host / output / "preprocess", dirs_exist_ok=True
            )
        run_logged_container([*args, image, *container_command], root)
        if container_command[0] == "presim":
            check_presim_outputs(workspace)
            logger.info(
                "Verified IMPACTS presim manifest, fleet assignments, and rates."
            )
        return True

    # Use real containers with networking disabled to verify the fixture is self-contained.
    with patch.object(GenericRunner, "run_container", staticmethod(run_docker)):
        run_pilates(settings=settings)
    check_live_outputs(workspace)


def workspace_path(workspace: Path, container_path: str) -> Path:
    return workspace / Path(container_path).relative_to(WORKSPACE_CONTAINER_DIR)


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def check_presim_outputs(workspace: Path) -> None:
    settings_path = workspace / "impacts/output/settings_presim.yaml"
    activity_path = workspace / PRESIM_ACTIVITIES / "activities_manifest.yaml"
    try:
        settings = read_yaml(settings_path)
        run = settings["run"]
        fleet = read_yaml(activity_path.with_name("fleet_manifest.yaml"))
        activities = read_yaml(activity_path)
        if (
            run["output_run_name"] != RUN_LABEL
            or run["scenario"] != RUN_NAME
            or run["start_year"] != YEAR
            or activities["calendar_year"] != YEAR
            or activities["scenario"] != settings["impacts"]["scenario"]
            or workspace_path(workspace, activities["settings_source"]) != settings_path
            or workspace_path(workspace, fleet["activities_manifest_path"])
            != activity_path
        ):
            raise RuntimeError(f"Presim manifests do not describe the {RUN_LABEL} run.")

        beam_fixture = FIXTURE / "beam/sfbay"
        for group, assignment, source_path, id_column in (
            (
                "passenger",
                "passenger_vehicles",
                beam_fixture / "urbansim/atlas-2019/vehicles.parquet",
                "vehicle_id",
            ),
            (
                "freight",
                "freight_carriers",
                beam_fixture
                / "freight/20250730/2018-Baseline/tours--2018-Baseline.parquet",
                "tourId",
            ),
        ):
            assignments = check_generated_fleet_files(
                group,
                workspace_path(workspace, fleet["outputs"][f"{group}_vehicle_types"]),
                workspace_path(workspace, fleet["outputs"][assignment]),
            )
            expected_ids = pd.read_parquet(source_path, columns=[id_column])[id_column]
            if sorted(assignments[id_column].astype(str)) != sorted(
                expected_ids.astype(str)
            ):
                raise RuntimeError(
                    f"Presim {group} assignments did not preserve the source {id_column} IDs."
                )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Presim output validation failed: {exc}") from exc


def check_live_outputs(workspace: Path) -> None:
    postsim = workspace / f"impacts/output/impacts-postsim--sfbay--{RUN_LABEL}"
    manifest = read_yaml(postsim / "pipeline_manifest.yaml")
    inputs = read_yaml(workspace_path(workspace, manifest["preprocess_manifest_path"]))[
        "inputs"
    ]
    fleet = read_yaml(workspace / PRESIM_ACTIVITIES / "fleet_manifest.yaml")
    activities = read_yaml(workspace_path(workspace, fleet["activities_manifest_path"]))
    if (
        activities["calendar_year"] != YEAR
        or inputs["rates_folder"]["source_path"]
        != activities["outputs"]["emissions_store_root"]
    ):
        raise RuntimeError(
            "Postsim rates and presim activities must come from the same 2018 run."
        )

    expected_osm = f"{BEAM_INPUT_CONTAINER_DIR}/sfbay/r5/sf-smoke/sf-smoke.osm.pbf"
    if inputs["osm_network"]["source_path"] != expected_osm:
        raise RuntimeError(
            "Postsim did not resolve the same OSM PBF bundled with BEAM: "
            f"{inputs['osm_network']['source_path']}"
        )

    beam_output = workspace / f"beam/output/sfbay/year-{YEAR}-iteration-0"
    staged = workspace / f"impacts/input/beam-output/{YEAR}"
    for key, relative in (
        ("network", "network.csv.gz"),
        ("emissions_skims_input", "ITERS/it.0/0.skimsEmissions.parquet"),
    ):
        path = workspace_path(workspace, inputs[key]["source_path"])
        if path != staged / relative or not cmp(
            path, beam_output / relative, shallow=False
        ):
            raise RuntimeError(f"Postsim did not use the final BEAM {key}.")

    skims = pd.read_parquet(beam_output / "ITERS/it.0/0.skimsEmissions.parquet")
    network = pd.read_csv(beam_output / "network.csv.gz", usecols=["linkId"])
    if (
        skims.empty
        or not 0 < len(network) < 2_000
        or not set(skims.linkId) <= set(network.linkId)
    ):
        raise RuntimeError(
            "BEAM emissions must reference links in the small San Francisco network."
        )
    launch = workspace / f".pilates-beam-launch-config/y{YEAR}/i0/sfbay"
    types_by_group = {}
    comparison = pd.read_parquet(
        postsim / "postprocess/fleet/step1_fleet_comparison.parquet"
    )
    for group, assignment, destination in (
        (
            "passenger",
            "passenger_vehicles",
            launch / "urbansim/atlas-2019/vehicles.parquet",
        ),
        (
            "freight",
            "freight_carriers",
            launch
            / ".pilates/impacts-fleet"
            / Path(fleet["outputs"]["freight_carriers"]).name,
        ),
    ):
        inventory = activities["outputs"][f"final_activity_by_emfacid_output_{group}"]
        if inputs[f"{group}_inventory_emfacid_file"]["source_path"] != inventory:
            raise RuntimeError(f"Postsim did not read the generated {group} inventory.")
        check_inventory_totals(
            pd.read_parquet(workspace_path(workspace, inventory)),
            comparison.loc[comparison.assignment_group.eq(group)],
            group,
        )
        type_key = f"{group}_vehicle_types"
        for key in (type_key, assignment):
            if inputs[f"{key}_input"]["source_path"] != fleet["outputs"][key]:
                raise RuntimeError(f"Postsim did not read the generated {key}.")
        assignment_path = workspace_path(workspace, fleet["outputs"][assignment])
        if not cmp(assignment_path, destination, shallow=False):
            raise RuntimeError(f"BEAM did not use the generated {group} assignments.")
        source = workspace_path(workspace, fleet["outputs"][type_key])
        staged_types = launch / "vehicle-tech/emissions" / f"impacts-{source.name}"
        types = pd.read_csv(source).set_index("vehicleTypeId", verify_integrity=True)
        launched = pd.read_csv(staged_types).set_index(
            "vehicleTypeId", verify_integrity=True
        )
        used = pd.read_parquet(assignment_path)["vehicleTypeId"].unique()
        if set(types.index) != set(launched.index) or not set(used) <= set(
            launched.index
        ):
            raise RuntimeError(f"BEAM launched with different {group} vehicle types.")
        for vehicle in used:
            original = source.parent / types.at[vehicle, "emissionsRatesFile"]
            actual = staged_types.parent / launched.at[vehicle, "emissionsRatesFile"]
            if not cmp(original, actual, shallow=False):
                raise RuntimeError(f"BEAM rate for {vehicle} differs from presim.")
        types_by_group.update({vehicle: group for vehicle in used})

    def read_output(key, **kwargs):
        return pd.read_parquet(
            workspace_path(workspace, manifest["outputs"][key]), **kwargs
        )

    unknown_types = set(skims.vehicleTypeId) - types_by_group.keys()
    if unknown_types:
        raise RuntimeError(
            f"Live BEAM skims reference unknown presim vehicle types: {sorted(map(str, unknown_types))}."
        )
    emitted = read_output("beam_emissions_by_county_process")
    keys = ["linkId", "vehicleTypeId", "process"]
    beam_keys = set(skims[keys].itertuples(index=False, name=None))
    county_keys = set(emitted[keys].itertuples(index=False, name=None))
    if county_keys != beam_keys:
        raise RuntimeError(
            f"Postsim county emissions differ from live BEAM skim keys: "
            f"{len(beam_keys - county_keys)} missing, {len(county_keys - beam_keys)} unexpected."
        )
    nox = emitted.groupby(emitted.vehicleTypeId.map(types_by_group))[
        "tons_per_year_NOx_county_allocated"
    ].sum()
    if (
        set(emitted.county_COUNTYFP) != {"075"}
        or set(nox.index) != {"passenger", "freight"}
        or not np.isfinite(nox).all()
        or not (nox > 0).all()
    ):
        raise RuntimeError(
            "Expected positive passenger and freight NOx emissions in San Francisco."
        )
    if manifest["outputs"]["beam_activity_correction_factors"] is not None:
        raise RuntimeError(
            "The live smoke test expects inventory corrections to be disabled."
        )
    check_emissions_mass(read_output, emitted)
    if not manifest["execution"]["dispersion_completed"]:
        raise RuntimeError("Both IMPACTS dispersion stages must complete.")

    check_dispersion_outputs(
        read_output, workspace / "beam/input/sfbay", FIXTURE / "inmap/isrm_sample.zarr"
    )
    print(
        f"Verified {len(skims):,} live emissions records over {len(network):,} San Francisco road links."
    )


def check_emissions_mass(read_output, county: pd.DataFrame) -> None:
    pollutants = ("NH3", "NOx", "PM25", "SOx", "ROG", "BC")
    columns = [f"tons_per_year_{pollutant}" for pollutant in pollutants]

    def totals(table: pd.DataFrame, label: str, suffix: str) -> np.ndarray:
        values = table[[column + suffix for column in columns]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise RuntimeError(f"Invalid pollutant mass in {label} emissions.")
        return values.sum(axis=0)

    annualized = totals(read_output("skims_emissions"), "annualized BEAM", "")
    for surface, table, tolerance in (
        ("county", county, 1e-9),
        ("inmap", read_output("beam_emissions_for_inmap"), 1e-4),
        ("aermod", read_output("beam_emissions_for_aermod"), 1e-4),
    ):
        np.testing.assert_allclose(
            totals(table, surface, f"_{surface}_allocated"),
            annualized,
            rtol=tolerance,
            atol=0,
            err_msg=f"Postsim {surface} allocation did not conserve {', '.join(pollutants)} mass",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--impacts-image",
        required=True,
        help="IMPACTS image with generated-fleet manifest support.",
    )
    args = parser.parse_args()
    try:
        with smoke_run_directory("impacts-live-sf") as directory:
            run_live_smoke(directory, impacts_image=args.impacts_image)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"IMPACTS live Docker smoke test failed: {exc}") from exc
    print("IMPACTS live San Francisco Docker smoke test passed.")


if __name__ == "__main__":
    main()
