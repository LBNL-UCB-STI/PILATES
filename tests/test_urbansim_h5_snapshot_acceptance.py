"""Contract tests for UrbanSim HDF5 snapshot acceptance preflight."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd
import pytest

from pilates.runtime import urbansim_h5_snapshot_acceptance as acceptance


def _write_population_h5(path: Path, tables: tuple[str, ...]) -> Path:
    """Create a real UrbanSim-style root-population HDF5 fixture."""
    with pd.HDFStore(path, mode="w") as store:
        for table_name in tables:
            store.put(table_name, pd.DataFrame({"value": [1]}))
    return path


def _write_manifest(path: Path, *, source: Path, cohort: dict[str, int]) -> Path:
    path.write_text(
        json.dumps(
            {
                "inputs": {"usim_datastore_h5": str(source)},
                "cohort": cohort,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_manifest_rejects_ambiguous_or_incomplete_cohort(tmp_path: Path) -> None:
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "inputs": {"usim_datastore_h5": str(tmp_path / "source.h5")},
                "cohort": {"year": 2019},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="workflow_year=2017, forecast_year=2019, iteration=0",
    ):
        acceptance.load_manifest(manifest)


def test_load_manifest_requires_a_readable_source_file(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "inputs.json",
        source=tmp_path / "missing.h5",
        cohort={"workflow_year": 2017, "forecast_year": 2019, "iteration": 0},
    )

    with pytest.raises(ValueError, match="readable file"):
        acceptance.load_manifest(manifest)


def test_load_manifest_expands_source_environment_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_population_h5(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    monkeypatch.setenv("URBANSIM_SOURCE_H5", str(source))
    manifest = _write_manifest(
        tmp_path / "inputs.json",
        source=Path("$URBANSIM_SOURCE_H5"),
        cohort={"workflow_year": 2017, "forecast_year": 2019, "iteration": 0},
    )

    loaded = acceptance.load_manifest(manifest)

    assert loaded == acceptance.AcceptanceManifest(
        usim_datastore_h5=source.resolve(),
        workflow_year=2017,
        forecast_year=2019,
        iteration=0,
    )


def test_describe_population_h5_rejects_missing_root_table(tmp_path: Path) -> None:
    source = _write_population_h5(
        tmp_path / "source.h5", ("households", "persons", "jobs")
    )

    with pytest.raises(ValueError, match="missing root population tables"):
        acceptance.describe_population_h5(
            source, year=2019, require_year_aliases=False
        )


def test_describe_population_h5_lists_root_tables_without_writing_year_aliases(
    tmp_path: Path,
) -> None:
    source = _write_population_h5(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )

    descriptor = acceptance.describe_population_h5(
        source, year=2019, require_year_aliases=False
    )

    assert descriptor["path"] == str(source.resolve())
    assert descriptor["size_bytes"] == source.stat().st_size
    assert descriptor["root_tables"] == ["households", "persons", "jobs", "blocks"]
    assert "year_aliases" not in descriptor
    with h5py.File(source, "r") as handle:
        assert "2019" not in handle


def test_describe_population_h5_requires_exact_year_aliases(tmp_path: Path) -> None:
    source = _write_population_h5(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )

    with pytest.raises(ValueError, match="missing exact year population aliases"):
        acceptance.describe_population_h5(source, year=2019, require_year_aliases=True)


def test_describe_population_h5_records_verified_year_aliases(tmp_path: Path) -> None:
    source = _write_population_h5(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    with h5py.File(source, "r+") as handle:
        aliases = handle.require_group("2019")
        for table_name in ("households", "persons", "jobs", "blocks"):
            aliases[table_name] = handle[table_name]

    descriptor = acceptance.describe_population_h5(
        source, year=2019, require_year_aliases=True
    )

    assert descriptor["year_aliases"] == [
        "/2019/households",
        "/2019/persons",
        "/2019/jobs",
        "/2019/blocks",
    ]


def test_write_json_creates_the_destination_parent_and_round_trips_data(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evidence" / "capture.json"

    acceptance._write_json(destination, {"z": 1, "a": {"value": True}})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "a": {"value": True},
        "z": 1,
    }
