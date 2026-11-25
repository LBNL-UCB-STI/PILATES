"""Prepare the full UrbanSim forecast for residential exposure."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import yaml
from consist import Artifact
from consist.admission import PriorRunExpectation, check_prior_run_expectation
from consist.core.persistence import DatabaseManager
from consist.digest import digest_file

from pilates.utils.usim_h5 import household_home_coordinates
from pilates.workflows.artifact_keys import USIM_DATASTORE_H5, USIM_FORECAST_OUTPUT

logger = logging.getLogger(__name__)


def exposure_population_dir(workspace, year: int) -> Path:
    return (
        Path(workspace.full_path)
        / "impacts/exposure-population"
        / f"urbansim-forecast-{year}"
    )


def validate_exposure_files(files, *, workspace, year: int) -> Path:
    if files is None or any(path is None or not path.is_file() for path in files):
        raise ValueError(
            "IMPACTS exposure requires the prepared UrbanSim population files."
        )
    persons, households, source = (path.resolve() for path in files)
    directory = source.parent
    root = Path(workspace.full_path)
    expected = root.resolve() / exposure_population_dir(workspace, year).relative_to(
        root
    )
    if (persons, households, source) != tuple(
        directory / name
        for name in ("persons.parquet", "households.parquet", "population_source.yaml")
    ) or directory != expected:
        raise ValueError(
            "IMPACTS exposure files must share one directory inside the workspace."
        )
    manifest = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("forecast_year") != year:
        raise ValueError(f"IMPACTS exposure requires the UrbanSim forecast for {year}.")
    return directory


def resolve_archived_forecast(settings, scenario) -> Artifact:
    source = settings.impacts.exposure_population_source
    if source is None or not source.source_tracker_db or not source.expected_bytes_path:
        raise ValueError(
            "IMPACTS exposure requires impacts.exposure_population_source with the "
            "prior UrbanSim step ID, artifact key, archive database and archived HDF5 path."
        )
    if source.artifact_key not in {USIM_FORECAST_OUTPUT, USIM_DATASTORE_H5}:
        raise ValueError("IMPACTS exposure requires a raw UrbanSim forecast output.")
    database = Path(os.path.expandvars(source.source_tracker_db)).expanduser()
    forecast = (
        Path(os.path.expandvars(source.expected_bytes_path)).expanduser().resolve()
    )
    if not database.is_file() or not forecast.is_file():
        raise FileNotFoundError(
            f"IMPACTS exposure archive database or forecast unavailable: {database}, {forecast}"
        )
    db = DatabaseManager(str(database), skip_schema_init=True, read_only=True)
    try:
        run = db.get_run(source.expected_run_id)
        if run is None or not (
            run.model_name == "urbansim_run"
            or (
                run.model_name == "urbansim"
                and (run.phase or run.meta.get("phase")) == "run"
            )
        ):
            raise ValueError(
                "IMPACTS exposure source must be an UrbanSim forecast step."
            )
        artifacts = db.find_linked_artifacts_exact(
            run_id=run.id, key=source.artifact_key, direction="output"
        )
    finally:
        db.close()
    if len(artifacts) != 1:
        raise ValueError("IMPACTS exposure requires exactly one prior forecast output.")
    report = check_prior_run_expectation(
        execution_path=forecast,
        expectation=PriorRunExpectation(
            expected_run_id=run.id,
            artifact_key=source.artifact_key,
            source_tracker_db=database,
            expected_direction="output",
            identity_scheme="container",
        ),
    )
    if report.outcome != "verified":
        raise ValueError(
            f"IMPACTS exposure forecast identity verification failed: "
            f"{report.reason or ', '.join(report.observations)}"
        )
    return scenario.seed_artifact(
        USIM_FORECAST_OUTPUT,
        forecast,
        content_hash=report.observed_artifact_id,
        reuse_if_unchanged=False,
        source_run_id=run.id,
        source_artifact_id=str(artifacts[0].id),
        source_digest=report.observed_artifact_id,
    )


def write_exposure_population(
    forecast: Path, destination: Path, *, year: int, source: dict[str, str]
) -> None:
    with pd.HDFStore(forecast, mode="r") as store:
        tables = [f"/{year}/{name}" for name in ("persons", "households", "blocks")]
        missing = [table for table in tables if table not in store]
        if missing:
            raise ValueError(
                f"IMPACTS exposure forecast is missing exact-year tables: {missing}"
            )
        persons, households, blocks = (store[table] for table in tables)

    for frame, key in (
        (persons, "person_id"),
        (households, "household_id"),
        (blocks, "block_id"),
    ):
        if key in frame.columns:
            frame.set_index(key, inplace=True)
        if frame.index.name != key or frame.index.hasnans or not frame.index.is_unique:
            raise ValueError(
                f"IMPACTS exposure forecast requires unique {key} identifiers."
            )

    coordinates = household_home_coordinates(households, blocks)
    longitude = pd.to_numeric(coordinates["x"], errors="coerce")
    latitude = pd.to_numeric(coordinates["y"], errors="coerce")
    valid = longitude.between(-180, 180) & latitude.between(-90, 90)
    if not valid.any():
        raise ValueError(
            "IMPACTS exposure forecast has no valid longitude/latitude block coordinates."
        )
    households["home_x"] = longitude.where(valid)
    households["home_y"] = latitude.where(valid)
    for column in ("home_x", "home_y"):
        persons[column] = persons["household_id"].map(households[column])
    missing_persons = int(persons["home_x"].isna().sum())
    missing_households = int((~valid).sum())
    if missing_persons or missing_households:
        logger.warning(
            "UrbanSim exposure population for %s has %s persons and %s households without home coordinates.",
            year,
            missing_persons,
            missing_households,
        )

    source_digest = source.get("source_digest")
    digest = digest_file(forecast)
    if source_digest and source_digest != digest:
        raise ValueError("IMPACTS exposure forecast changed after Consist verified it.")
    manifest = {
        "description": "Full UrbanSim forecast before travel-model filtering, used for home-based exposure.",
        "forecast_year": year,
        "source_run_id": source["source_run_id"],
        "source_artifact_id": source["source_artifact_id"],
        "source_digest": digest,
        "person_count": len(persons),
        "household_count": len(households),
        "persons_missing_home_coordinates": missing_persons,
        "households_missing_home_coordinates": missing_households,
        "coordinate_method": "Household block_id joined to UrbanSim block x/y in EPSG:4326.",
    }
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as directory:
        staged = Path(directory)
        persons.reset_index().to_parquet(staged / "persons.parquet", index=False)
        households.reset_index().to_parquet(staged / "households.parquet", index=False)
        (staged / "population_source.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        for name in ("persons.parquet", "households.parquet", "population_source.yaml"):
            (staged / name).replace(destination / name)
