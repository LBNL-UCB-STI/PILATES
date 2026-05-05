from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest
import yaml

from pilates.runtime.legacy_archive_doctor import inspect_legacy_archive, main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_gz(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_run_state(run_dir: Path, *, year: int = 2023, forecast_year: int = 2025) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.yaml").write_text(
        yaml.safe_dump(
            {
                "year": year,
                "forecast_year": forecast_year,
                "iteration": 0,
                "stage": "activity_demand",
            }
        ),
        encoding="utf-8",
    )


def _seed_activitysim_inputs(run_dir: Path, *, year: int = 2023) -> Path:
    inputs_dir = run_dir / "activitysim" / "output" / f"inputs-year-{year}-iteration-0"
    _write_csv(
        inputs_dir / "households.csv",
        [{"household_id": 1}, {"household_id": 2}],
    )
    _write_csv(
        inputs_dir / "persons.csv",
        [{"person_id": 10, "household_id": 1}, {"person_id": 11, "household_id": 2}],
    )
    _write_csv(inputs_dir / "land_use.csv", [{"TAZ": 1}])
    return inputs_dir


def test_dry_run_writes_report_and_does_not_create_activitysim_alias(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run_state(run_dir)
    _seed_activitysim_inputs(run_dir)

    result = inspect_legacy_archive(run_dir, apply=False)

    doctor_dir = run_dir / ".workflow" / "legacy_archive_doctor"
    alias_dir = run_dir / "activitysim" / "output" / "inputs-year-2025-iteration-0"
    assert result.mode == "dry-run"
    assert not alias_dir.exists()
    assert (doctor_dir / "report.json").exists()
    assert (doctor_dir / "candidate_actions.jsonl").exists()
    assert not (doctor_dir / "manifest.json").exists()

    report = json.loads((doctor_dir / "report.json").read_text(encoding="utf-8"))
    assert report["run_state"]["forecast_year"] == 2025
    assert {Path(action["destination"]).name for action in report["actions"]} == {
        "households.csv",
        "persons.csv",
        "land_use.csv",
    }


def test_apply_creates_forecast_year_activitysim_copies_and_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run_state(run_dir)
    source_dir = _seed_activitysim_inputs(run_dir)

    result = inspect_legacy_archive(run_dir, apply=True)

    doctor_dir = run_dir / ".workflow" / "legacy_archive_doctor"
    alias_dir = run_dir / "activitysim" / "output" / "inputs-year-2025-iteration-0"
    assert result.mode == "apply"
    assert (alias_dir / "households.csv").read_text(encoding="utf-8") == (
        source_dir / "households.csv"
    ).read_text(encoding="utf-8")
    assert (alias_dir / "persons.csv").exists()
    assert (alias_dir / "land_use.csv").exists()

    manifest = json.loads((doctor_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mutated_consist_db_metadata"] is False
    assert manifest["action_count"] == 3
    actions = [
        json.loads(line)
        for line in (doctor_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {action["status"] for action in actions} == {"applied"}
    assert json.loads((doctor_dir / "conflicts.json").read_text(encoding="utf-8")) == {
        "conflicts": []
    }


def test_apply_reports_conflict_without_overwriting_existing_alias(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run_state(run_dir)
    _seed_activitysim_inputs(run_dir)
    alias_households = (
        run_dir
        / "activitysim"
        / "output"
        / "inputs-year-2025-iteration-0"
        / "households.csv"
    )
    _write_csv(alias_households, [{"household_id": 99}])

    result = inspect_legacy_archive(run_dir, apply=True)

    assert len(result.conflicts) == 1
    assert alias_households.read_text(encoding="utf-8").count("99") == 1
    conflicts = json.loads(
        (
            run_dir / ".workflow" / "legacy_archive_doctor" / "conflicts.json"
        ).read_text(encoding="utf-8")
    )
    assert conflicts["conflicts"][0]["reason"] == "activitysim_forecast_year_alias_conflict"


def test_detects_mixed_population_risk_from_tiny_csv_sources(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run_state(run_dir)
    _seed_activitysim_inputs(run_dir)
    _write_csv_gz(
        run_dir / "beam" / "output" / "inputs-year-2025-iteration-0" / "households.csv.gz",
        [{"household_id": 1}, {"household_id": 3}],
    )
    _write_csv(
        run_dir / "atlas" / "output" / "householdv_2025.csv",
        [{"household_id": 1}, {"household_id": 2}],
    )

    result = inspect_legacy_archive(run_dir, apply=False)

    risk = result.mixed_population_risk
    assert risk["status"] == "risk"
    mismatches = {
        (comparison["left_role"], comparison["right_role"])
        for comparison in risk["comparisons"]
        if comparison["status"] == "mismatch"
    }
    assert ("activitysim", "beam") in mismatches
    assert ("atlas", "beam") in mismatches


def test_infers_forecast_year_from_asim_lifecycle_year_and_samples_vehicles2(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_state.yaml").write_text(
        yaml.safe_dump({"year": 2019, "iteration": 0, "stage": "traffic_assignment"}),
        encoding="utf-8",
    )
    _seed_activitysim_inputs(run_dir, year=2019)
    lifecycle_dir = run_dir / ".workflow" / "diagnostics"
    lifecycle_dir.mkdir(parents=True)
    (lifecycle_dir / "artifact_lifecycle_audit.jsonl").write_text(
        json.dumps(
            {
                "event_type": "artifact_logged",
                "artifact_family": "asim_input_archived",
                "key": "asim_input_households_csv_archived",
                "year": 2021,
                "iteration": 0,
                "path": str(
                    run_dir
                    / "activitysim"
                    / "output"
                    / "inputs-year-2019-iteration-0"
                    / "households.csv"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        run_dir / "atlas" / "atlas_output" / "vehicles2_2021.csv",
        [{"vehicleId": 1, "householdId": 3}],
    )

    result = inspect_legacy_archive(run_dir, apply=False)

    assert {
        Path(action.destination).parent.name for action in result.actions
    } == {"inputs-year-2021-iteration-0"}
    risk = result.mixed_population_risk
    assert risk["status"] == "risk"
    assert "atlas" in risk["sampled_roles"]
    assert any(
        comparison["status"] == "mismatch"
        and {comparison["left_role"], comparison["right_role"]}
        == {"activitysim", "atlas"}
        for comparison in risk["comparisons"]
    )


def test_reports_activitysim_h5_forecast_year_path_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run_state(run_dir, year=2019, forecast_year=2021)
    lifecycle_dir = run_dir / ".workflow" / "diagnostics"
    lifecycle_dir.mkdir(parents=True)
    h5_path = run_dir / "urbansim" / "data" / "model_data_2019.h5"
    h5_path.parent.mkdir(parents=True)
    h5_path.write_text("h5", encoding="utf-8")
    (lifecycle_dir / "artifact_lifecycle_audit.jsonl").write_text(
        json.dumps(
            {
                "event_type": "artifact_logged",
                "key": "usim_datastore_h5",
                "artifact_family": "usim_datastore_h5",
                "year": 2021,
                "path": str(h5_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = inspect_legacy_archive(run_dir, apply=False)

    assert result.h5_path_mismatches == [
        {
            "reason": "h5_forecast_year_path_mismatch",
            "event_type": "artifact_logged",
            "artifact_key": "usim_datastore_h5",
            "artifact_family": "usim_datastore_h5",
            "semantic_year": 2021,
            "path_year": 2019,
            "path": str(h5_path),
            "relative_path": "urbansim/data/model_data_2019.h5",
            "canonical_filename": "model_data_2021.h5",
            "repairable": False,
            "detail": (
                "H5 datastore path year does not match lifecycle semantic year; "
                "the doctor reports this but does not rewrite H5 semantics."
            ),
        }
    ]
    report = json.loads(
        (
            run_dir / ".workflow" / "legacy_archive_doctor" / "report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["h5_path_mismatches"][0]["semantic_year"] == 2021


def test_cli_dry_run_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run"
    _seed_run_state(run_dir)
    _seed_activitysim_inputs(run_dir)

    assert main(["--archive-run-dir", str(run_dir), "--dry-run"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["report"].endswith(".workflow/legacy_archive_doctor/report.json")


def test_detects_mixed_population_risk_from_tiny_parquet_when_available(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    run_dir = tmp_path / "run"
    _seed_run_state(run_dir)
    _seed_activitysim_inputs(run_dir)
    beam_dir = run_dir / "beam" / "output" / "inputs-year-2025-iteration-0"
    beam_dir.mkdir(parents=True)
    pd.DataFrame({"household_id": [1, 4]}).to_parquet(beam_dir / "households.parquet")

    result = inspect_legacy_archive(run_dir, apply=False)

    assert result.mixed_population_risk["status"] == "risk"
