"""Run the PILATES IMPACTS smoke test using the bundled BEAM fixture."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import logging
import os
import shutil
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import zarr

from pilates.config import ImpactsConfig, InfrastructureConfig
from pilates.generic.runner import GenericRunner
from pilates.impacts import ImpactsPostsimRunner, ImpactsPresimRunner
from pilates.impacts.settings_bridge import (
    BEAM_INPUT_CONTAINER_DIR,
    WORKSPACE_CONTAINER_DIR,
)
from pilates.workspace import Workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests/fixtures/impacts_smoke"
IMAGE = "ghcr.io/lbnl-ucb-sti/impacts:v0.2.3"


@contextmanager
def smoke_run_directory(suite: str):
    parent = PROJECT_ROOT / "test-output" / suite
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-")
    directory = Path(tempfile.mkdtemp(prefix=stamp, dir=parent))
    scratch = directory / "tmp"
    scratch.mkdir()
    handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    logger = logging.getLogger()
    logger.addHandler(handler)
    print(f"Test outputs: {directory}", flush=True)
    temp_env = dict.fromkeys(("TMPDIR", "TMP", "TEMP"), str(scratch))
    try:
        with patch.dict(os.environ, temp_env):
            with patch.object(tempfile, "tempdir", str(scratch)):
                yield directory
    except Exception:
        logger.exception("Smoke test failed")
        raise
    else:
        with (directory / "run.log").open("a", encoding="utf-8") as log:
            log.write("Smoke test passed.\n")
    finally:
        logger.removeHandler(handler)
        handler.close()


def run_logged_container(command: list[str], directory: Path) -> None:
    with (directory / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            if status := process.wait():
                raise subprocess.CalledProcessError(status, command)


def check_generated_fleet(workspace) -> None:
    fleet_output = Path(workspace.get_impacts_input_dir()) / "emfac"
    for model, assignment_name in (("atlas", "vehicles"), ("frism", "carriers")):
        (type_file,) = fleet_output.glob(f"vehicleTypes--{model}--*--EM.csv")
        (assignment_file,) = fleet_output.glob(f"{assignment_name}--*--EM.parquet")
        check_generated_fleet_files(model, type_file, assignment_file)


def check_generated_fleet_files(
    model: str, type_file: Path, assignment_file: Path
) -> pd.DataFrame:
    types = pd.read_csv(type_file)
    assignments = pd.read_parquet(assignment_file)
    type_ids = types["vehicleTypeId"]
    assignment_types = assignments["vehicleTypeId"]
    if (
        type_ids.isna().any()
        or type_ids.duplicated().any()
        or assignments.empty
        or assignment_types.isna().any()
        or not set(assignment_types) <= set(type_ids)
    ):
        raise RuntimeError(
            f"Presim {model} vehicle assignments do not match the generated BEAM vehicle types."
        )
    used = types.loc[type_ids.isin(assignment_types)]
    for rate in (
        used[["emfacId", "emissionsRatesFile"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        if not isinstance(rate.emissionsRatesFile, str) or not rate.emissionsRatesFile:
            raise RuntimeError(f"Presim {model} has no rate file for {rate.emfacId}.")
        reference = Path(rate.emissionsRatesFile)
        if reference.is_absolute() or reference.stem != rate.emfacId:
            raise RuntimeError(
                f"Presim {model} has an invalid rate reference for {rate.emfacId}."
            )
        rate_path = type_file.parent / reference
        try:
            rates = pq.ParquetFile(rate_path).read()
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Presim {model} could not read rate file {rate_path}: {exc}"
            ) from exc
        if rates.num_rows == 0:
            raise RuntimeError(
                f"Presim generated empty {model} rates for {rate.emfacId}."
            )
    return assignments


def stage_presim_rates(workspace, presim) -> None:
    activity = yaml.safe_load(
        (presim.output_dir / "activities/activities_manifest.yaml").read_text()
    )
    rates = Path(workspace.full_path) / Path(
        activity["outputs"]["emissions_store_root"]
    ).relative_to(WORKSPACE_CONTAINER_DIR)
    beam_input = Path(workspace.get_beam_mutable_data_dir())
    vehicle_dir = beam_input / "vehicle-tech"
    shutil.copytree(rates, vehicle_dir / "emissions/activities/2018-Baseline/rates")

    skims = pd.read_parquet(
        Path(workspace.get_beam_output_dir()) / "ITERS/it.0/0.skimsEmissions.parquet",
        columns=["vehicleTypeId", "process"],
    )
    types = pd.concat(
        [pd.read_csv(path) for path in vehicle_dir.glob("vehicleTypes--*--EM.csv")],
        ignore_index=True,
    ).set_index("vehicleTypeId", verify_integrity=True)
    for vehicle_type, observed in skims.groupby("vehicleTypeId"):
        vehicle = types.loc[vehicle_type]
        reference = Path(vehicle["emissionsRatesFile"])
        if (
            reference.is_absolute()
            or ".." in reference.parts
            or reference.stem != vehicle["emfacId"]
        ):
            raise RuntimeError(
                f"Invalid BEAM emissions rate reference for {vehicle_type}."
            )
        target = vehicle_dir / reference
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                rates / "dataset" / reference.parent.name / reference.name, target
            )
        table = pd.read_parquet(target)
        county_rates = table.loc[table.county.str.casefold() == "alameda"]
        missing = set(observed.process) - set(county_rates.process)
        if missing:
            raise RuntimeError(
                f"Presim rates for {vehicle_type} lack observed BEAM processes: {sorted(missing)}."
            )
        if not {"speedMph_timeMin", "roadCategory", "nox_gram", "pm25_gram"} <= set(
            table.columns
        ):
            raise RuntimeError(
                f"Presim rates for {vehicle_type} lack BEAM rate columns."
            )


def check_downstream_outputs(workspace, presim, postsim) -> None:
    def read_yaml(path):
        return yaml.safe_load(path.read_text())

    def workspace_path(path):
        return Path(workspace.full_path) / Path(path).relative_to(
            WORKSPACE_CONTAINER_DIR
        )

    before = read_yaml(presim.settings_path)
    after = read_yaml(postsim.settings_path)
    if (
        before["run"] != after["run"]
        or before["impacts"]["scenario"] != after["impacts"]["scenario"]
    ):
        raise RuntimeError("Presim and postsim describe different runs or scenarios.")
    activity = read_yaml(presim.output_dir / "activities/activities_manifest.yaml")
    if activity["calendar_year"] != after["run"]["start_year"]:
        raise RuntimeError("Presim activity year does not match postsim.")

    manifest = read_yaml(postsim.pipeline_manifest)
    inputs = read_yaml(workspace_path(manifest["preprocess_manifest_path"]))["inputs"]
    if (
        workspace_path(manifest["outputs"]["beam_concentration_distribution"])
        != postsim.exposure_table
    ):
        raise RuntimeError("PILATES and IMPACTS disagree on the exposure output path.")
    if (
        inputs["rates_folder"]["source_path"]
        != f"{BEAM_INPUT_CONTAINER_DIR}/vehicle-tech/emissions/activities/2018-Baseline/rates"
    ):
        raise RuntimeError("Postsim did not resolve the rate store staged from presim.")
    beam_output = Path(workspace.get_beam_output_dir())
    skims_file = beam_output / "ITERS/it.0/0.skimsEmissions.parquet"
    network_file = beam_output / "network.csv.gz"
    for key, expected in (
        ("network", network_file),
        ("emissions_skims_input", skims_file),
    ):
        if workspace_path(inputs[key]["source_path"]) != expected:
            raise RuntimeError(f"Postsim selected the wrong BEAM artifact for {key}.")
    skims = pd.read_parquet(skims_file, columns=["vehicleTypeId", "linkId", "process"])
    if not set(skims.linkId) <= set(
        pd.read_csv(network_file, usecols=["linkId"]).linkId
    ):
        raise RuntimeError(
            "The archived skims reference links absent from the BEAM network."
        )

    beam_input = Path(workspace.get_beam_mutable_data_dir())
    known_vehicle_types = set()
    comparison = pd.read_parquet(
        postsim.output_dir / "postprocess/fleet/step1_fleet_comparison.parquet"
    )
    for group, assignment_name in (("passenger", "vehicles"), ("freight", "carriers")):
        inventory_path = activity["outputs"][
            f"final_activity_by_emfacid_output_{group}"
        ]
        if inputs[f"{group}_inventory_emfacid_file"]["source_path"] != inventory_path:
            raise RuntimeError(
                f"Postsim used a different {group} inventory than presim published."
            )
        type_path = Path(inputs[f"{group}_vehicle_types_input"]["source_path"])
        types = pd.read_csv(
            beam_input / type_path.relative_to(BEAM_INPUT_CONTAINER_DIR)
        )
        assignments = pd.read_parquet(
            next(beam_input.rglob(f"{assignment_name}--sample--EM.parquet")),
            columns=["vehicleTypeId"],
        )
        if types.vehicleTypeId.duplicated().any() or not set(
            assignments.vehicleTypeId
        ) <= set(types.vehicleTypeId):
            raise RuntimeError(
                f"Invalid archived BEAM vehicle types or assignments for {group}."
            )
        known_vehicle_types.update(types.vehicleTypeId)
        compared = comparison.loc[comparison.assignment_group.eq(group)]
        inventory = pd.read_parquet(workspace_path(inventory_path))
        check_inventory_totals(inventory, compared, group)
    if skims.empty or not set(skims.vehicleTypeId) <= known_vehicle_types:
        raise RuntimeError("The archived skims reference unknown BEAM vehicle types.")

    def read_output(key, **kwargs):
        return pd.read_parquet(workspace_path(manifest["outputs"][key]), **kwargs)

    key_columns = ["linkId", "vehicleTypeId", "process"]
    allocated = read_output("beam_emissions_by_county_process", columns=key_columns)
    if set(allocated.itertuples(index=False, name=None)) != set(
        skims[key_columns].itertuples(index=False, name=None)
    ):
        raise RuntimeError(
            "Postsim county emissions do not cover the archived BEAM skim keys."
        )
    check_dispersion_outputs(
        read_output, beam_input, FIXTURE / "inmap/isrm_sample.zarr"
    )


def check_inventory_totals(inventory, compared, group: str) -> None:
    expected_totals = (
        inventory.groupby(["county", "emfacId"])[
            ["total_vmt_vehicle_miles_per_year", "population_vehicles"]
        ]
        .max()
        .groupby("emfacId")
        .sum()
    )
    actual_totals = compared.set_index("emfacId", verify_integrity=True)[
        ["emfac_vmt", "emfac_population"]
    ]
    ids = actual_totals.index.union(expected_totals.index)
    np.testing.assert_allclose(
        actual_totals.reindex(ids, fill_value=0).to_numpy(dtype=float),
        expected_totals.reindex(ids, fill_value=0).to_numpy(dtype=float),
        err_msg=f"{group} fleet comparison does not reproduce the presim inventory totals",
    )
    if not (compared["beam_vmt"].sum() > 0 and compared["emfac_vmt"].sum() > 0):
        raise RuntimeError(f"The {group} comparison needs BEAM and EMFAC VMT.")


def check_dispersion_outputs(read_output, beam_input: Path, inmap_store: Path) -> None:
    inmap = read_output("beam_inmap_concentrations").set_index(
        "inmap_cell_id", verify_integrity=True
    )
    aermod = read_output("beam_aermod_concentrations").set_index(
        "aermod_cell_id", verify_integrity=True
    )
    exposure = read_output("beam_concentration_distribution").set_index(
        "aermod_cell_id", verify_integrity=True
    )
    for label, table in (("InMAP", inmap), ("AERMOD", aermod), ("exposure", exposure)):
        values = table[["TotalPM25", "PrimaryPM25", "BC", "NO2"]].to_numpy()
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not (values > 0).any(axis=0).all()
        ):
            raise RuntimeError(
                f"{label} needs finite nonnegative concentrations and a positive result for each pollutant."
            )

    # Independently project the emitted PM2.5 through the fixture's real source-receptor coefficients.
    sr = zarr.open_group(inmap_store, mode="r")
    emitted = read_output("beam_emissions_for_inmap")
    masses = emitted.groupby("inmap_cell_id")[
        "tons_per_year_PM25_inmap_allocated"
    ].sum()
    if set(inmap.index) != set(masses.index):
        raise RuntimeError(
            "InMAP must return a receptor result for each cell with emissions."
        )
    coefficients = sr["PrimaryPM25"].oindex[
        0, masses.index.to_numpy(), inmap.index.to_numpy()
    ]
    short_tons_per_year_to_micrograms_per_second = 907_184.74 * 1e6 / (365 * 24 * 3600)
    np.testing.assert_allclose(
        inmap.PrimaryPM25,
        masses.to_numpy() @ coefficients * short_tons_per_year_to_micrograms_per_second,
        rtol=1e-6,
    )
    nox = emitted.groupby("inmap_cell_id")["tons_per_year_NOx_inmap_allocated"].sum()
    with np.load(
        beam_input / "vehicle-tech/dispersions/nox_to_no2_sample.npz"
    ) as transfer:
        expected_no2 = np.bincount(
            transfer["receptor_ids"],
            weights=transfer["values"]
            * nox.reindex(transfer["source_ids"], fill_value=0).to_numpy(),
            minlength=int(transfer["receptor_dim"]),
        )
    np.testing.assert_allclose(inmap.NO2, expected_no2[inmap.index], rtol=1e-6)
    if set(aermod.index) != set(exposure.index):
        raise RuntimeError("The exposure grid and AERMOD grid have different cell IDs.")
    grid = pd.read_parquet(
        beam_input / "vehicle-tech/dispersions/inmap_cells.parquet", columns=["isrm"]
    )
    if not exposure.inmap_cell_id.isin(grid["isrm"]).all():
        raise RuntimeError("Exposure references an unknown InMAP cell.")
    # Exposure uses zero for InMAP contributions outside its emitted-cell receptor set.
    expected_inmap = inmap.reindex(exposure.inmap_cell_id, fill_value=0)
    expected_aermod = aermod.loc[exposure.index]
    np.testing.assert_allclose(
        exposure.SecondaryPM25, expected_inmap.TotalPM25 - expected_inmap.PrimaryPM25
    )
    np.testing.assert_allclose(
        exposure.TotalPM25, exposure.PrimaryPM25 + exposure.SecondaryPM25
    )
    for column, mask_column in (
        ("PrimaryPM25", "has_aermod_primarypm25"),
        ("BC", "has_aermod_bc"),
        ("NO2", "has_aermod_no2"),
    ):
        mask = expected_aermod[mask_column].to_numpy()
        if not mask.any():
            raise RuntimeError(f"Expected AERMOD coverage for {column}.")
        np.testing.assert_allclose(
            exposure[column],
            np.where(mask, expected_aermod[column], expected_inmap[column]),
        )
    people = read_output("beam_population_distribution")
    counts = read_output("beam_population_counts").set_index(
        "aermod_cell_id", verify_integrity=True
    )
    expected_people = pd.read_parquet(
        beam_input / "urbansim/atlas-2019/persons.parquet"
    )
    if sorted(people.person_id) != sorted(expected_people.person_id) or set(
        counts.index
    ) != set(exposure.index):
        raise RuntimeError(
            "Exposure dropped sampled people or assigned them outside the grid."
        )
    np.testing.assert_array_equal(
        counts.person_count,
        people.groupby("aermod_cell_id").size().reindex(counts.index, fill_value=0),
    )


def require_docker() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker CLI is required.")
    try:
        subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            stdout=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("The Docker daemon is unreachable.") from exc
    return docker


def run_smoke(root: Path, *, fleet: bool = True, image: str = IMAGE) -> None:
    docker = require_docker()

    def run_docker(*, image, volumes, command, working_dir, environment=None, **_):
        # shortcut: launches Docker directly, use a Consist run to test tracking.
        args = [
            docker,
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--workdir",
            working_dir,
            "--env",
            "MPLCONFIGDIR=/tmp/matplotlib",
            # shortcut: v0.2.0 loads an older system libstdc++, remove after the image fixes its library path.
            "--env",
            "LD_LIBRARY_PATH=/opt/conda/lib",
        ]
        for host, target, mode in GenericRunner._extract_mounts(volumes):
            readonly = ",readonly" if mode == "ro" else ""
            args += ["--mount", f"type=bind,src={host},dst={target}{readonly}"]
        for key, value in (environment or {}).items():
            args += ["--env", f"{key}={value}"]
        container_command = GenericRunner._build_full_command(command)
        if container_command[0] == "postsim":
            # shortcut: v0.2.0 puts DuckDB scratch beside the input, remove when it uses writable scratch.
            for directory in duckdb_dirs:
                target = directory.relative_to(beam_input).as_posix()
                args += ["--tmpfs", f"{BEAM_INPUT_CONTAINER_DIR}/{target}:mode=1777"]
        try:
            run_logged_container([*args, image, *container_command], root)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Docker failed while running {container_command[0]} "
                f"(exit {exc.returncode})."
            ) from exc
        return True

    settings = SimpleNamespace(
        infrastructure=InfrastructureConfig(docker_images={"impacts": image}),
        run=SimpleNamespace(
            region="sfbay",
            scenario="smoke",
            start_year=2018,
            output_run_name="smoke",
            output_directory=str(root),
            models=SimpleNamespace(
                land_use=None,
                travel=None,
                activity_demand=None,
                vehicle_ownership=None,
                impacts="impacts",
            ),
        ),
        shared=SimpleNamespace(
            geography=SimpleNamespace(
                FIPS={"state": "06", "counties": ["001"]},
                local_crs="EPSG:26910",
            ),
        ),
        beam=SimpleNamespace(
            local_mutable_data_folder="beam/input",
            local_output_folder="beam/output",
            router_directory="vehicle-tech/sample.osm.pbf",
        ),
        impacts=ImpactsConfig(
            local_input_folder="impacts/input",
            local_output_folder="impacts/output",
            local_inmap_data_folder=str(FIXTURE / "inmap"),
            pipeline_presim={"fleet": fleet},
            population_fallback_to_beam_inputs=False,
            overrides={
                "analysis": {
                    "targets": None,
                    "inventory_targets": None,
                    "basemap": "none",
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
                "activities": {
                    "region_label": "SF",
                    "emissions_raw_archive": "vehicle-tech/emissions_raw.tar.zst",
                },
            },
        ),
    )
    workspace = Workspace(settings, str(root), "run")
    shutil.copytree(FIXTURE / "beam", Path(workspace.full_path) / "beam")
    beam_input = Path(workspace.get_beam_mutable_data_dir())
    duckdb_dirs = [
        assignment.parent / "duckdb"
        for assignment in beam_input.rglob("*--sample--EM.parquet")
    ]
    for directory in duckdb_dirs:
        directory.mkdir()
    output = Path(workspace.get_impacts_output_dir())
    shutil.copytree(
        FIXTURE / "county",
        output / "impacts-postsim--sfbay--smoke_y2018/preprocess",
    )
    state = SimpleNamespace(
        full_settings=settings, set_sub_stage_progress=lambda _: None
    )
    with patch.object(GenericRunner, "run_container", staticmethod(run_docker)):
        presim = ImpactsPresimRunner("impacts", state).run(None, workspace)
        if fleet:
            check_generated_fleet(workspace)
        stage_presim_rates(workspace, presim)
        postsim = ImpactsPostsimRunner("impacts", state).run(None, workspace)

    activity_manifest = presim.output_dir / "activities/activities_manifest.yaml"
    fleet_comparison = (
        postsim.output_dir / "postprocess/fleet/step1_fleet_comparison.parquet"
    )
    for path in (
        activity_manifest,
        postsim.pipeline_manifest,
        postsim.postprocess_manifest,
        fleet_comparison,
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing expected Docker output: {path.name}")
    if yaml.safe_load(activity_manifest.read_text())["calendar_year"] != 2018:
        raise RuntimeError("Expected 2018 in the presim activities manifest.")
    nox = "tons_per_year_NOx_county_allocated"
    county = "county_COUNTYFP"
    emissions = pq.read_table(
        postsim.output_dir / "emissions/beam_emissions_by_county_process.parquet",
        columns=[county, nox],
    )
    if set(emissions[county].to_pylist()) != {"001"}:
        raise RuntimeError("Expected emissions rows for Alameda County only.")
    if not sum(emissions[nox].to_pylist()) > 0:
        raise RuntimeError("Expected positive NOx emissions.")
    check_downstream_outputs(workspace, presim, postsim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", default=IMAGE, help="IMPACTS Docker image to verify."
    )
    parser.add_argument(
        "--replay-without-fleet",
        action="store_true",
        help="Check fresh EMFAC rates and the archived BEAM replay without regenerating the fleet.",
    )
    args = parser.parse_args()
    try:
        with smoke_run_directory("impacts-replay") as directory:
            run_smoke(
                directory,
                fleet=not args.replay_without_fleet,
                image=args.image,
            )
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"IMPACTS Docker smoke test failed: {exc}") from exc
    if args.replay_without_fleet:
        print("IMPACTS Docker replay passed (fleet generation excluded)")
    else:
        print("IMPACTS Docker fleet and replay checks passed")


if __name__ == "__main__":
    main()
