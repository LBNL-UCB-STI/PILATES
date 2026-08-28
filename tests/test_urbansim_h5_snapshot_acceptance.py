"""Contract tests for UrbanSim HDF5 snapshot acceptance preflight."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys

import h5py
import pandas as pd
import pytest

from pilates.runtime import urbansim_h5_snapshot_acceptance as acceptance


_SETTINGS = (
    Path(__file__).parents[1]
    / "scenarios/sfbay/settings-sfbay-consist-usim-hpc-2019-canary.yaml"
)


def _write_population_h5(path: Path, tables: tuple[str, ...]) -> Path:
    """Create a real UrbanSim-style root-population HDF5 fixture."""
    with pd.HDFStore(path, mode="w") as store:
        for table_name in tables:
            store.put(table_name, pd.DataFrame({"value": [1]}))
    return path


def _write_manifest(path: Path, *, source: Path, cohort: Mapping[str, object]) -> Path:
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


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _real_h5_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source = _write_population_h5(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    return source, _write_manifest(
        tmp_path / "inputs.json",
        source=source,
        cohort={"workflow_year": 2017, "forecast_year": 2019, "iteration": 0},
    )


def _run_module(*arguments: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pilates.runtime.urbansim_h5_snapshot_acceptance",
            *arguments,
        ],
        check=True,
        text=True,
        capture_output=True,
    )


def test_capture_then_fresh_process_reconciliation_preserves_h5_identity(
    tmp_path: Path,
) -> None:
    _source, manifest = _real_h5_manifest(tmp_path)
    evidence = tmp_path / "evidence"
    _run_module(
        "capture",
        "--settings",
        str(_SETTINGS),
        "--manifest",
        str(manifest),
        "--evidence-root",
        str(evidence),
    )
    (evidence / "provenance.duckdb").unlink()
    _run_module("reconcile", "--evidence-root", str(evidence))

    validation = _read_json(evidence / "validation.json")
    assert validation == {
        **validation,
        "source_h5_valid": True,
        "strict_binding_valid": True,
        "override_is_unowned": True,
        "snapshot_created": True,
        "fresh_snapshot_readable": True,
        "persisted_identity_unchanged": True,
        "output_h5_aliases_valid": True,
        "valid": True,
    }
    assert _read_json(evidence / "capture.json")["trusted_identity"] == _read_json(
        evidence / "reconciliation.json"
    )["persisted_identity"]


def test_reconciliation_requires_tracker_snapshot_before_opening_tracker(
    tmp_path: Path,
) -> None:
    _source, manifest = _real_h5_manifest(tmp_path)
    evidence = tmp_path / "evidence"
    _run_module(
        "capture",
        "--settings",
        str(_SETTINGS),
        "--manifest",
        str(manifest),
        "--evidence-root",
        str(evidence),
    )
    (evidence / "snapshots/tracker.duckdb").unlink()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilates.runtime.urbansim_h5_snapshot_acceptance",
            "reconcile",
            "--evidence-root",
            str(evidence),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "snapshots/tracker.duckdb" in completed.stderr


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


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("workflow_year", True),
        ("workflow_year", 2017.0),
        ("forecast_year", True),
        ("forecast_year", 2019.0),
        ("iteration", False),
        ("iteration", 0.0),
    ],
)
def test_load_manifest_rejects_boolean_and_float_cohort_values(
    tmp_path: Path, field_name: str, invalid_value: object
) -> None:
    source = _write_population_h5(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    cohort: dict[str, object] = {
        "workflow_year": 2017,
        "forecast_year": 2019,
        "iteration": 0,
    }
    cohort[field_name] = invalid_value
    manifest = _write_manifest(
        tmp_path / "inputs.json", source=source, cohort=cohort
    )

    with pytest.raises(
        ValueError,
        match="workflow_year=2017, forecast_year=2019, iteration=0",
    ):
        acceptance.load_manifest(manifest)


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


def test_describe_population_h5_rejects_copied_year_tables(tmp_path: Path) -> None:
    source = _write_population_h5(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    with h5py.File(source, "r+") as handle:
        aliases = handle.require_group("2019")
        for table_name in ("households", "persons", "jobs", "blocks"):
            handle.copy(table_name, aliases, name=table_name)

    with pytest.raises(ValueError, match="missing exact year population aliases"):
        acceptance.describe_population_h5(source, year=2019, require_year_aliases=True)


def test_write_json_creates_the_destination_parent_and_round_trips_data(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evidence" / "capture.json"

    acceptance._write_json(destination, {"z": 1, "a": {"value": True}})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "a": {"value": True},
        "z": 1,
    }
