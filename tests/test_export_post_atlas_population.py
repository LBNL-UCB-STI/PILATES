from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pilates.scripts import export_post_atlas_population as export_script


def test_artifact_path_resolves_tracker_uri() -> None:
    tracker = SimpleNamespace(resolve_uri=lambda uri: uri.replace("workspace://", "/tmp/"))
    artifact = SimpleNamespace(container_uri="workspace://atlas/vehicles2_2025.csv")
    assert export_script._artifact_path(artifact, tracker) == "/tmp/atlas/vehicles2_2025.csv"


def test_normalize_export_frame_promotes_expected_index() -> None:
    frame = pd.DataFrame({"cars": [1]}, index=pd.Index([100], name="household_id"))
    normalized = export_script._normalize_export_frame(
        frame,
        expected_index_name="household_id",
    )
    assert list(normalized.columns) == ["household_id", "cars"]
    assert normalized.loc[0, "household_id"] == 100


def test_match_scenario_step_run_id_prefers_atlas_postprocess_name() -> None:
    scenario_run = SimpleNamespace(
        meta={
            "steps": [
                {
                    "id": "step-1",
                    "name": "atlas_postprocess",
                    "model": "atlas",
                    "phase": "postprocess",
                    "year": 2025,
                }
            ]
        }
    )
    assert export_script._match_scenario_step_run_id(scenario_run, year=2025) == "step-1"


def test_build_export_manifest_records_unique_scenario_run_id(tmp_path: Path) -> None:
    manifest = export_script._build_export_manifest(
        run_dir=tmp_path / "run",
        db_path=tmp_path / "run" / ".consist" / "provenance.duckdb",
        years=[2025],
        year_manifests=[
            {
                "year": 2025,
                "step": {"scenario_run_id": "scenario-1"},
            }
        ],
    )
    assert manifest["source"]["scenario_run_id"] == "scenario-1"
    assert manifest["requested_years"] == [2025]
    assert manifest["years"] == [2025]
    assert manifest["year_manifests"]["2025"] == "years/2025/table_manifest.json"


def test_build_export_manifest_records_skipped_years(tmp_path: Path) -> None:
    manifest = export_script._build_export_manifest(
        run_dir=tmp_path / "run",
        db_path=tmp_path / "run" / ".consist" / "provenance.duckdb",
        years=[2023, 2025],
        year_manifests=[
            {
                "year": 2025,
                "step": {"scenario_run_id": "scenario-1"},
            }
        ],
        skipped_years=[
            {
                "year": 2023,
                "reason": "missing file",
                "error_type": "FileNotFoundError",
            }
        ],
    )
    assert manifest["requested_years"] == [2023, 2025]
    assert manifest["years"] == [2025]
    assert manifest["skipped_years"] == [
        {
            "year": 2023,
            "reason": "missing file",
            "error_type": "FileNotFoundError",
        }
    ]
