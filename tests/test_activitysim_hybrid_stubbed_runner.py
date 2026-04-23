"""
Hybrid ActivitySim integration test:
- real ActivitySim preprocessor
- stubbed ActivitySim runner outputs
- real ActivitySim postprocessor

This complements the golden workflow test by validating model-specific
preprocess/postprocess behavior without needing full model execution.
"""

from pathlib import Path

import pandas as pd

from pilates.activitysim.outputs import (
    ActivitySimRunOutputs,
)
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.generic.records import FileRecord, RecordStore
from pilates.workspace import Workspace
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
)
from tests.test_golden_stub_workflow import _build_settings
from workflow_state import WorkflowState


def _write_file(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_activitysim_pre_post_with_stubbed_runner(monkeypatch, tmp_path: Path) -> None:
    """
    Run real ActivitySim pre/postprocessors around a stubbed runner payload.

    This test is intentionally model-scoped: it validates ActivitySim file
    shaping and archival contracts while keeping runtime fast and deterministic.
    """
    settings = _build_settings(tmp_path)
    settings.land_use_enabled = False
    settings.vehicle_ownership_model_enabled = False
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = False
    settings.replanning_enabled = False
    settings.state_file_loc = str(tmp_path / "state.yaml")

    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)
    state.current_year = settings.run.start_year
    state.forecast_year = settings.run.start_year
    state.current_inner_iter = 0

    beam_skims = (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.shared.skims.fname
    )
    _write_file(beam_skims, b"beam-od-skims")

    def _fake_create_asim_data_from_h5(_settings, _state, ws, usim_store_path):
        asim_data_dir = Path(ws.get_asim_mutable_data_dir())
        assert usim_store_path == str(tmp_path / "bound-usim.h5")
        land_use = asim_data_dir / "land_use.csv"
        households = asim_data_dir / "households.csv"
        persons = asim_data_dir / "persons.csv"

        _write_csv(
            land_use,
            pd.DataFrame(
                {
                    "TAZ": [1, 2],
                    "TOTPOP": [120, 80],
                    "TOTHH": [1, 1],
                }
            ),
        )
        _write_csv(
            households,
            pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "TAZ": [1, 2],
                    "persons": [2, 1],
                }
            ),
        )
        _write_csv(
            persons,
            pd.DataFrame(
                {
                    "person_id": [11, 12, 21],
                    "household_id": [1, 1, 2],
                    "TAZ": [1, 1, 2],
                }
            ),
        )

        return [
            FileRecord(
                file_path=str(land_use),
                short_name=ASIM_LAND_USE_IN,
                content_hash="hash_land_use",
            ),
            FileRecord(
                file_path=str(households),
                short_name=ASIM_HOUSEHOLDS_IN,
                content_hash="hash_households_in",
            ),
            FileRecord(
                file_path=str(persons),
                short_name=ASIM_PERSONS_IN,
                content_hash="hash_persons_in",
            ),
        ]

    monkeypatch.setattr(
        "pilates.activitysim.preprocessor.create_asim_data_from_h5",
        _fake_create_asim_data_from_h5,
    )
    bound_usim_h5 = tmp_path / "bound-usim.h5"
    bound_usim_h5.write_text("h5", encoding="utf-8")

    preprocessor = ActivitysimPreprocessor("activitysim", state)
    preprocess_outputs = preprocessor.preprocess(
        workspace,
        usim_datastore_h5=str(bound_usim_h5),
    )

    preprocess_keys = {
        short_name
        for short_name, _path, _description in preprocess_outputs._iter_record_items()
    }
    assert {ASIM_LAND_USE_IN, ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_OMX_SKIMS} <= preprocess_keys
    assert preprocess_outputs.input_hashes[ASIM_LAND_USE_IN] == "hash_land_use"
    assert preprocess_outputs.input_hashes[ASIM_HOUSEHOLDS_IN] == "hash_households_in"
    assert preprocess_outputs.input_hashes[ASIM_PERSONS_IN] == "hash_persons_in"

    staged_omx = Path(workspace.get_asim_mutable_data_dir()) / "skims.omx"
    assert staged_omx.exists()
    assert staged_omx.read_bytes() == b"beam-od-skims"

    asim_output_dir = Path(workspace.get_asim_output_dir())
    raw_households = asim_output_dir / "stub_raw" / "households.parquet"
    raw_persons = asim_output_dir / "stub_raw" / "persons.parquet"
    raw_beam_plans = asim_output_dir / "stub_raw" / "beam_plans.parquet"

    _write_parquet(
        raw_households,
        pd.DataFrame({"household_id": [1, 2], "auto_ownership": [1, 2]}),
    )
    _write_parquet(
        raw_persons,
        pd.DataFrame({"person_id": [11, 12, 21], "is_worker": [True, True, False]}),
    )
    _write_parquet(
        raw_beam_plans,
        pd.DataFrame({"trip_id": [1, 2], "ActivityType": ["home", "work"]}),
    )

    zarr_cache = asim_output_dir / "cache" / "skims.zarr"
    _write_file(zarr_cache, b"zarr-cache")

    runner_outputs = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(raw_households),
                short_name="households_asim_out_temp",
                content_hash="hash_households_out",
            ),
            FileRecord(
                file_path=str(raw_persons),
                short_name="persons_asim_out_temp",
                content_hash="hash_persons_out",
            ),
            FileRecord(
                file_path=str(raw_beam_plans),
                short_name="beam_plans_asim_out_temp",
                content_hash="hash_beam_plans_out",
            ),
        ]
    )

    run_outputs = ActivitySimRunOutputs.from_record_store(runner_outputs, workspace)
    run_outputs.source_input_paths = {
        short_name: path
        for short_name, path, _description in preprocess_outputs._iter_record_items()
    }
    run_outputs.source_input_hashes = dict(preprocess_outputs.input_hashes)
    run_outputs.source_input_paths["zarr_skims"] = zarr_cache
    run_outputs.source_input_hashes["zarr_skims"] = "hash_zarr_skims"

    postprocessor = ActivitysimPostprocessor("activitysim", state)
    postprocess_outputs = postprocessor.postprocess(run_outputs, workspace)

    output_map = {
        short_name: path
        for short_name, path, _description in postprocess_outputs._iter_record_items()
    }
    assert "households_asim_out" in output_map
    assert "persons_asim_out" in output_map
    assert "beam_plans_asim_out" in output_map
    assert "asim_input_households_csv_archived" in output_map
    assert "asim_input_persons_csv_archived" in output_map
    assert "asim_input_land_use_csv_archived" in output_map
    assert "asim_input_skims_omx_archived" not in output_map
    assert "asim_input_skims_zarr_archived" in output_map

    iteration_dir = asim_output_dir / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    assert (iteration_dir / "households.parquet").exists()
    assert (iteration_dir / "persons.parquet").exists()
    assert (iteration_dir / "beam_plans.parquet").exists()

    assert not raw_households.exists()
    assert not raw_persons.exists()
    assert not raw_beam_plans.exists()

    assert postprocess_outputs.processed_output_hashes["households_asim_out"] == "hash_households_out"
    assert postprocess_outputs.processed_output_hashes["persons_asim_out"] == "hash_persons_out"
    assert postprocess_outputs.processed_output_hashes["beam_plans_asim_out"] == "hash_beam_plans_out"
    assert (
        postprocess_outputs.processed_output_hashes["asim_input_households_csv_archived"]
        == "hash_households_in"
    )
    assert (
        postprocess_outputs.processed_output_hashes["asim_input_persons_csv_archived"]
        == "hash_persons_in"
    )
    assert postprocess_outputs.processed_output_hashes["asim_input_land_use_csv_archived"] == "hash_land_use"
    assert (
        postprocess_outputs.processed_output_hashes["asim_input_skims_zarr_archived"]
        == "hash_zarr_skims"
    )
    assert state.sub_stage_progress == "postprocessor"


def test_activitysim_postprocess_archives_omx_when_zarr_was_not_a_run_input(
    monkeypatch, tmp_path: Path
) -> None:
    settings = _build_settings(tmp_path)
    settings.land_use_enabled = False
    settings.vehicle_ownership_model_enabled = False
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = False
    settings.replanning_enabled = False
    settings.state_file_loc = str(tmp_path / "state.yaml")

    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)
    state.current_year = settings.run.start_year
    state.forecast_year = settings.run.start_year
    state.current_inner_iter = 0

    asim_data_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_csv(asim_data_dir / "households.csv", pd.DataFrame({"household_id": [1]}))
    _write_csv(asim_data_dir / "persons.csv", pd.DataFrame({"person_id": [11]}))
    _write_csv(asim_data_dir / "land_use.csv", pd.DataFrame({"TAZ": [1]}))
    _write_file(asim_data_dir / "skims.omx", b"omx-skims")

    asim_output_dir = Path(workspace.get_asim_output_dir())
    raw_households = asim_output_dir / "stub_raw" / "households.parquet"
    raw_persons = asim_output_dir / "stub_raw" / "persons.parquet"
    raw_beam_plans = asim_output_dir / "stub_raw" / "beam_plans.parquet"
    _write_parquet(raw_households, pd.DataFrame({"household_id": [1]}))
    _write_parquet(raw_persons, pd.DataFrame({"person_id": [11]}))
    _write_parquet(raw_beam_plans, pd.DataFrame({"trip_id": [1]}))

    zarr_cache = asim_output_dir / "cache" / "skims.zarr"
    _write_file(zarr_cache, b"zarr-cache")

    runner_outputs = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(raw_households),
                short_name="households_asim_out_temp",
                content_hash="hash_households_out",
            ),
            FileRecord(
                file_path=str(raw_persons),
                short_name="persons_asim_out_temp",
                content_hash="hash_persons_out",
            ),
            FileRecord(
                file_path=str(raw_beam_plans),
                short_name="beam_plans_asim_out_temp",
                content_hash="hash_beam_plans_out",
            ),
        ]
    )

    run_outputs = ActivitySimRunOutputs.from_record_store(runner_outputs, workspace)
    run_outputs.source_input_paths = {
        ASIM_HOUSEHOLDS_IN: asim_data_dir / "households.csv",
        ASIM_PERSONS_IN: asim_data_dir / "persons.csv",
        ASIM_LAND_USE_IN: asim_data_dir / "land_use.csv",
        ASIM_OMX_SKIMS: asim_data_dir / "skims.omx",
    }
    run_outputs.source_input_hashes = {
        ASIM_HOUSEHOLDS_IN: "hash_households_in",
        ASIM_PERSONS_IN: "hash_persons_in",
        ASIM_LAND_USE_IN: "hash_land_use",
        ASIM_OMX_SKIMS: "hash_omx_skims",
    }

    monkeypatch.setattr(
        "pilates.activitysim.postprocessor.create_usim_input_data",
        lambda *args, **kwargs: (None, None),
    )

    postprocessor = ActivitysimPostprocessor("activitysim", state)
    postprocess_outputs = postprocessor.postprocess(run_outputs, workspace)

    output_map = {
        short_name: path
        for short_name, path, _description in postprocess_outputs._iter_record_items()
    }
    assert "asim_input_skims_omx_archived" in output_map
    assert "asim_input_skims_zarr_archived" in output_map
    assert (
        postprocess_outputs.processed_output_hashes["asim_input_skims_omx_archived"]
        == "hash_omx_skims"
    )


def test_activitysim_postprocess_rehydrates_existing_iteration_outputs(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    settings.land_use_enabled = False
    settings.vehicle_ownership_model_enabled = False
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = False
    settings.replanning_enabled = False
    settings.state_file_loc = str(tmp_path / "state.yaml")

    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)
    state.current_year = settings.run.start_year
    state.forecast_year = settings.run.start_year
    state.current_inner_iter = 0

    iteration_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    archived_households = iteration_dir / "households.parquet"
    archived_persons = iteration_dir / "persons.parquet"
    archived_beam_plans = iteration_dir / "beam_plans.parquet"

    _write_parquet(
        archived_households,
        pd.DataFrame({"household_id": [1], "auto_ownership": [1]}),
    )
    _write_parquet(
        archived_persons,
        pd.DataFrame({"person_id": [11], "is_worker": [True]}),
    )
    _write_parquet(
        archived_beam_plans,
        pd.DataFrame({"trip_id": [1], "ActivityType": ["home"]}),
    )

    stale_stub_dir = Path(workspace.get_asim_output_dir()) / "stale_raw"
    run_outputs = ActivitySimRunOutputs(
        output_dir=Path(workspace.get_asim_output_dir()),
        raw_outputs={
            "households_asim_out_temp": stale_stub_dir / "households.parquet",
            "persons_asim_out_temp": stale_stub_dir / "persons.parquet",
            "beam_plans_asim_out_temp": stale_stub_dir / "beam_plans.parquet",
        },
        raw_output_hashes={
            "households_asim_out_temp": "hash_households_out",
            "persons_asim_out_temp": "hash_persons_out",
            "beam_plans_asim_out_temp": "hash_beam_plans_out",
        },
    )

    postprocessor = ActivitysimPostprocessor("activitysim", state)
    postprocess_outputs = postprocessor.postprocess(run_outputs, workspace)

    assert postprocess_outputs.processed_outputs["households_asim_out"] == archived_households
    assert postprocess_outputs.processed_outputs["persons_asim_out"] == archived_persons
    assert (
        postprocess_outputs.processed_outputs["beam_plans_asim_out"]
        == archived_beam_plans
    )
    assert (
        postprocess_outputs.processed_output_hashes["households_asim_out"]
        == "hash_households_out"
    )
    assert (
        postprocess_outputs.processed_output_hashes["persons_asim_out"]
        == "hash_persons_out"
    )
    assert (
        postprocess_outputs.processed_output_hashes["beam_plans_asim_out"]
        == "hash_beam_plans_out"
    )
