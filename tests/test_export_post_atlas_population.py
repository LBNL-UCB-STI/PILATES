from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd

from pilates.scripts import export_post_atlas_population as export_script


def test_artifact_path_resolves_tracker_uri() -> None:
    tracker = SimpleNamespace(
        resolve_uri=lambda uri: uri.replace("workspace://", "/tmp/")
    )
    artifact = SimpleNamespace(container_uri="workspace://atlas/vehicles2_2025.csv")
    assert (
        export_script._artifact_path(artifact, tracker)
        == "/tmp/atlas/vehicles2_2025.csv"
    )


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
    assert (
        export_script._match_scenario_step_run_id(
            scenario_run,
            year=2025,
            step_name="atlas_postprocess",
            phase="postprocess",
        )
        == "step-1"
    )


def test_schema_for_artifact_key_uses_registry_mappings() -> None:
    assert (
        export_script._schema_for_artifact_key("atlas_households_csv").__name__
        == "AtlasHousehold"
    )
    assert (
        export_script._schema_for_artifact_key("householdv_2025").__name__
        == "HouseholdVAtlasOut"
    )


def test_build_export_manifest_records_unique_scenario_run_id(tmp_path: Path) -> None:
    manifest = export_script._build_export_manifest(
        run_dir=tmp_path / "run",
        db_path=tmp_path / "run" / ".consist" / "provenance.duckdb",
        years=[2025],
        source_mode="hdf",
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
        source_mode="atlas_csv_sql",
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


def test_build_translation_manifest_records_years_and_source(tmp_path: Path) -> None:
    manifest = export_script._build_translation_manifest(
        source_dir=tmp_path / "source",
        output_dir=tmp_path / "out",
        years=[2025, 2027],
        year_manifests=[
            {"year": 2025},
            {"year": 2027},
        ],
        schema_spec="polaris",
        scenario="zev_mandate",
        source_manifest={
            "export_type": "post_atlas_population_extract",
            "source_mode": "atlas_csv_sql",
            "source": {
                "run_dir": "/tmp/run",
                "db_path": "/tmp/db.duckdb",
                "scenario_run_id": "scenario-1",
                "scenario_run_ids": ["scenario-1"],
            },
            "requested_years": [2023, 2025, 2027],
            "years": [2025, 2027],
            "tables": ["households", "persons", "vehicles"],
            "skipped_years": [{"year": 2023}],
        },
        skipped_years=[
            {"year": 2023, "reason": "missing input", "error_type": "FileNotFoundError"}
        ],
    )
    assert manifest["export_type"] == "polaris_population_translation"
    assert manifest["schema_spec"] == "polaris"
    assert manifest["scenario"] == "zev_mandate"
    assert manifest["requested_years"] == [2025, 2027]
    assert manifest["years"] == [2025, 2027]
    assert manifest["year_manifests"]["2025"] == "years/2025/table_manifest.json"
    assert (
        manifest["source"]["source_export_manifest_copy"]
        == "source_export_manifest.json"
    )
    assert manifest["source"]["source_export_type"] == "post_atlas_population_extract"
    assert manifest["source"]["run_dir"] == "/tmp/run"


def test_extract_year_sql_reconstructs_households_and_writes_parquet(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    input_year_dir = run_dir / "atlas" / "atlas_input" / "year2025"
    input_year_dir.mkdir(parents=True)
    households_csv = input_year_dir / "households.csv"
    persons_csv = input_year_dir / "persons.csv"
    householdv_csv = run_dir / "atlas" / "atlas_output" / "householdv_2025.csv"
    householdv_csv.parent.mkdir(parents=True)
    vehicles2_csv = run_dir / "atlas" / "atlas_output" / "vehicles2_2025.csv"

    households_csv.write_text(
        "household_id,income,cars,hh_cars\n1,100,0,none\n2,200,1,one\n",
        encoding="utf-8",
    )
    persons_csv.write_text(
        "person_id,household_id,age\n10,1,30\n20,2,40\n",
        encoding="utf-8",
    )
    householdv_csv.write_text(
        "household_id,nvehicles\n1,2\n2,0\n",
        encoding="utf-8",
    )
    vehicles2_csv.write_text(
        "vehicle_id,household_id,vehicleTypeId\n1000,1,sedan_gas_2015\n",
        encoding="utf-8",
    )

    step_run = SimpleNamespace(
        id="step-1",
        parent_run_id=None,
        status="completed",
        model_name="atlas_postprocess",
    )

    class _Tracker:
        def find_runs(
            self, *, model=None, phase=None, year=None, status=None, limit=None
        ):
            assert year == 2025
            if model == "atlas_preprocess" and phase == "preprocess":
                return [
                    SimpleNamespace(id="pre-1", parent_run_id=None, status="completed")
                ]
            if model == "atlas_run" and phase == "run":
                return [
                    SimpleNamespace(id="run-1", parent_run_id=None, status="completed")
                ]
            if model == "atlas_postprocess" and phase == "postprocess":
                return [step_run]
            return []

        def get_run_outputs(self, run_id):
            if run_id == "pre-1":
                return {
                    "atlas_households_csv": SimpleNamespace(path=str(households_csv)),
                    "atlas_persons_csv": SimpleNamespace(path=str(persons_csv)),
                }
            if run_id == "run-1":
                return {
                    "householdv_2025": SimpleNamespace(path=str(householdv_csv)),
                }
            if run_id == "step-1":
                return {
                    "atlas_vehicles2_output": SimpleNamespace(path=str(vehicles2_csv)),
                }
            raise AssertionError(run_id)

        def get_artifacts_for_run(self, run_id):
            return SimpleNamespace(inputs={}, outputs={})

        def resolve_uri(self, uri: str) -> str:
            return uri

        def get_run(self, run_id):
            return None

    manifest = export_script._extract_year_sql(
        _Tracker(),
        step_run=step_run,
        run_dir=run_dir,
        year=2025,
        output_dir=tmp_path / "out",
        vehicles_source="auto",
        hash_mode="none",
    )

    assert manifest["source_mode"] == "atlas_csv_sql"
    households_parquet = tmp_path / "out" / "years" / "2025" / "households.parquet"
    persons_parquet = tmp_path / "out" / "years" / "2025" / "persons.parquet"
    vehicles_parquet = tmp_path / "out" / "years" / "2025" / "vehicles.parquet"
    assert households_parquet.exists()
    assert persons_parquet.exists()
    assert vehicles_parquet.exists()

    conn = duckdb.connect()
    try:
        households = conn.sql(
            f"select * from read_parquet('{households_parquet}') order by household_id"
        ).fetchall()
        persons = conn.sql(
            f"select * from read_parquet('{persons_parquet}') order by person_id"
        ).fetchall()
        vehicles = conn.sql(
            f"select * from read_parquet('{vehicles_parquet}')"
        ).fetchall()
    finally:
        conn.close()

    assert households == [(1.0, 100.0, 2, "two or more"), (2.0, 200.0, 0, "none")]
    assert persons == [(10, 30.0, 1.0), (20, 40.0, 2.0)]
    assert vehicles == [(1, 1000, "sedan_gas_2015")]
