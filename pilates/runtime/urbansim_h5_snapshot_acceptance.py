"""Fail-closed preflight helpers for UrbanSim HDF5 snapshot acceptance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path

import h5py


_COHORT = {"workflow_year": 2017, "forecast_year": 2019, "iteration": 0}
_REQUIRED_ROOT_TABLES = ("households", "persons", "jobs", "blocks")
_EXPECTED_COHORT = "workflow_year=2017, forecast_year=2019, iteration=0"


@dataclass(frozen=True)
class AcceptanceManifest:
    """The one admissible local UrbanSim HDF5 acceptance input."""

    usim_datastore_h5: Path
    workflow_year: int
    forecast_year: int
    iteration: int


def load_manifest(path: Path) -> AcceptanceManifest:
    """Load the exact acceptance cohort and a readable UrbanSim HDF5 source."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read acceptance manifest: {path}") from error

    if not isinstance(loaded, Mapping):
        raise ValueError("acceptance manifest must be a JSON object")
    inputs = loaded.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"usim_datastore_h5"}:
        raise ValueError("acceptance manifest requires only inputs.usim_datastore_h5")
    value = inputs["usim_datastore_h5"]
    if not isinstance(value, str):
        raise ValueError("inputs.usim_datastore_h5 must be a path string")

    cohort = loaded.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError(f"acceptance cohort must be {_EXPECTED_COHORT}")
    _validate_state(cohort)

    source = Path(os.path.expandvars(value)).expanduser()
    if not source.is_file():
        raise ValueError(f"inputs.usim_datastore_h5 must be a readable file: {source}")
    try:
        with source.open("rb"):
            pass
    except OSError as error:
        raise ValueError(
            f"inputs.usim_datastore_h5 must be a readable file: {source}"
        ) from error

    return AcceptanceManifest(
        usim_datastore_h5=source.resolve(),
        workflow_year=_COHORT["workflow_year"],
        forecast_year=_COHORT["forecast_year"],
        iteration=_COHORT["iteration"],
    )


def _validate_state(state: Mapping[str, object]) -> None:
    """Reject every cohort except the one acceptance cohort."""
    if dict(state) != _COHORT:
        raise ValueError(f"acceptance cohort must be {_EXPECTED_COHORT}")


def describe_population_h5(
    path: Path, *, year: int, require_year_aliases: bool
) -> dict[str, object]:
    """Describe required root population tables without changing the source file."""
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        raise ValueError(f"UrbanSim HDF5 input must be a readable file: {resolved_path}")

    try:
        with h5py.File(resolved_path, "r") as handle:
            root_tables = [
                table_name for table_name in _REQUIRED_ROOT_TABLES if table_name in handle
            ]
            missing_root_tables = [
                f"/{table_name}"
                for table_name in _REQUIRED_ROOT_TABLES
                if table_name not in handle
            ]
            if missing_root_tables:
                raise ValueError(
                    "UrbanSim HDF5 input is missing root population tables: "
                    + ", ".join(missing_root_tables)
                )

            descriptor: dict[str, object] = {
                "path": str(resolved_path),
                "size_bytes": resolved_path.stat().st_size,
                "root_tables": root_tables,
            }
            if require_year_aliases:
                aliases = [f"/{year}/{table_name}" for table_name in root_tables]
                missing_aliases = [
                    alias
                    for alias, table_name in zip(aliases, root_tables, strict=True)
                    if alias not in handle
                    or handle[alias].id != handle[table_name].id
                ]
                if missing_aliases:
                    raise ValueError(
                        "UrbanSim HDF5 input is missing exact year population aliases: "
                        + ", ".join(missing_aliases)
                    )
                descriptor["year_aliases"] = aliases
            return descriptor
    except OSError as error:
        raise ValueError(f"could not read UrbanSim HDF5 input: {resolved_path}") from error


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    """Write an acceptance record without creating unrelated directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
