"""
Minimal Consist-first dummy workflow test.

Purpose:
- Provide a small, runnable end-to-end example of a Consist-enabled PILATES workflow.
- Demonstrate the generic preprocessor/runner/postprocessor pattern without adapter plumbing.
- Validate basic lineage behavior (inputs/outputs captured per step).

This test intentionally avoids adapter-era features and heavy integration checks.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import h5py
import pandas as pd
import pytest
from consist.types import CacheOptions, ExecutionOptions

from pilates.generic.model import provenance_logging
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import FileRecord, RecordStore
from pilates.generic.runner import GenericRunner
from pilates.utils import consist_runtime as cr


def get_record_by_short_name(store: RecordStore, short_name: str):
    """Find a record in a RecordStore by short_name."""
    for record in store.all_records():
        if record.short_name == short_name:
            return record
    return None


def make_unique_id(file_path: str) -> str:
    """Generate a stable, test-only ID from a file path."""
    import hashlib

    return hashlib.md5(file_path.encode()).hexdigest()[:16]


class DummyWorkflowState:
    """Minimal workflow state for tests."""

    def __init__(self, current_year):
        self.current_year = current_year
        self.forecast_year = current_year
        self.current_major_stage = None
        self.current_inner_iter = 0
        self.current_sub_stage = None
        self.full_settings = type(
            "Settings",
            (),
            {
                "shared": type(
                    "Shared", (), {"database": type("DB", (), {"path": None})()}
                )()
            },
        )()

    def set_sub_stage_progress(self, progress):
        if progress is None:
            self.current_sub_stage = None
        else:
            self.current_sub_stage = type("Stage", (), {"name": str(progress)})()


class DummyWorkspace:
    """Simplified workspace for testing."""

    def __init__(self, output_dir):
        self.output_dir = output_dir

    @property
    def full_path(self):
        return self.output_dir


# ============================================================================
# MODEL A
# ============================================================================


class DummyModelAPreprocessor(GenericPreprocessor):
    def __init__(self, model_name, config, state):
        super().__init__(model_name, state)
        self.config = config

    def copy_data_to_mutable_location(
        self, settings: dict, output_dir: str
    ) -> Tuple[RecordStore, RecordStore]:
        input_dir = self.config["input_dir"]
        shutil.copy(os.path.join(input_dir, "data.csv"), output_dir)
        shutil.copy(os.path.join(input_dir, "data.h5"), output_dir)

        input_csv_path = os.path.join(input_dir, "data.csv")
        input_h5_path = os.path.join(input_dir, "data.h5")

        output_csv_path = os.path.join(output_dir, "data.csv")
        output_h5_path = os.path.join(output_dir, "data.h5")

        input_records = RecordStore(
            recordList=[
                FileRecord(
                    file_path=input_csv_path,
                    short_name="data.csv",
                    unique_id=make_unique_id(input_csv_path),
                ),
                FileRecord(
                    file_path=input_h5_path,
                    short_name="data.h5",
                    unique_id=make_unique_id(input_h5_path),
                ),
            ]
        )

        output_records = RecordStore(
            recordList=[
                FileRecord(
                    file_path=output_csv_path,
                    short_name="data.csv",
                    unique_id=make_unique_id(output_csv_path),
                ),
                FileRecord(
                    file_path=output_h5_path,
                    short_name="data.h5",
                    unique_id=make_unique_id(output_h5_path),
                ),
            ]
        )

        return input_records, output_records

    @provenance_logging
    def _preprocess(
        self, workspace: DummyWorkspace, previous_records: RecordStore = RecordStore()
    ) -> RecordStore:
        csv_record = get_record_by_short_name(previous_records, "data.csv")
        h5_record = get_record_by_short_name(previous_records, "data.h5")
        if not csv_record or not h5_record:
            raise ValueError("Expected data.csv and data.h5 in previous_records")

        pre_csv_path = os.path.join(workspace.output_dir, "preprocessed_data.csv")
        pre_h5_path = os.path.join(workspace.output_dir, "preprocessed_data.h5")
        shutil.copy(csv_record.file_path, pre_csv_path)
        shutil.copy(h5_record.file_path, pre_h5_path)

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=pre_csv_path,
                    short_name="preprocessed_data.csv",
                    unique_id=make_unique_id(pre_csv_path),
                ),
                FileRecord(
                    file_path=pre_h5_path,
                    short_name="preprocessed_data.h5",
                    unique_id=make_unique_id(pre_h5_path),
                ),
            ]
        )


class DummyModelARunner(GenericRunner):
    def __init__(self, model_name, config, state):
        super().__init__(model_name, state)
        self.config = config

    @provenance_logging
    def _run(self, store: RecordStore, workspace: DummyWorkspace) -> RecordStore:
        csv_record = get_record_by_short_name(store, "preprocessed_data.csv")
        h5_record = get_record_by_short_name(store, "preprocessed_data.h5")
        if not csv_record or not h5_record:
            raise ValueError("Expected preprocessed inputs for Model A")

        year = self.state.current_year
        df_csv = pd.read_csv(csv_record.file_path)
        df_csv["a_doubled"] = df_csv["a"] * 2
        output_csv_path = os.path.join(
            workspace.output_dir, f"model_a_output_{year}.csv"
        )
        df_csv.to_csv(output_csv_path, index=False)

        with h5py.File(h5_record.file_path, "r") as f_in:
            table1_data = f_in["table1"][()]
        df_table1 = pd.DataFrame(table1_data)
        df_table1_modified = df_table1 + 1
        output_h5_path = os.path.join(workspace.output_dir, f"model_a_output_{year}.h5")
        with h5py.File(output_h5_path, "w") as f_out:
            f_out.create_dataset(
                "table1_modified", data=df_table1_modified.to_records(index=False)
            )

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=output_csv_path,
                    short_name=f"model_a_output_{year}.csv",
                    unique_id=make_unique_id(output_csv_path),
                    year=year,
                ),
                FileRecord(
                    file_path=output_h5_path,
                    short_name=f"model_a_output_{year}.h5",
                    unique_id=make_unique_id(output_h5_path),
                    year=year,
                ),
            ]
        )


class DummyModelAPostprocessor(GenericPostprocessor):
    def __init__(self, model_name, config, state):
        super().__init__(model_name, state)
        self.config = config

    @provenance_logging
    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: DummyWorkspace,
        runInfo=None,
        model_run_hash=None,
    ) -> RecordStore:
        year = self.state.current_year
        output_csv_record = get_record_by_short_name(
            raw_outputs, f"model_a_output_{year}.csv"
        )
        output_h5_record = get_record_by_short_name(
            raw_outputs, f"model_a_output_{year}.h5"
        )
        if not output_csv_record or not output_h5_record:
            raise ValueError("Expected Model A runner outputs")

        df_csv = pd.read_csv(output_csv_record.file_path)
        df_csv_filtered = df_csv[df_csv["a_doubled"] > 10]
        final_csv_path = os.path.join(
            workspace.output_dir, f"model_a_final_output_{year}.csv"
        )
        df_csv_filtered.to_csv(final_csv_path, index=False)

        with h5py.File(output_h5_record.file_path, "r") as f_in:
            table_data = f_in["table1_modified"][()]
        df_table = pd.DataFrame(table_data)
        df_table_final = df_table * 2
        final_h5_path = os.path.join(
            workspace.output_dir, f"model_a_final_output_{year}.h5"
        )
        with h5py.File(final_h5_path, "w") as f_out:
            f_out.create_dataset(
                "table1_final", data=df_table_final.to_records(index=False)
            )

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=final_csv_path,
                    short_name=f"model_a_final_output_{year}.csv",
                    unique_id=make_unique_id(final_csv_path),
                    year=year,
                ),
                FileRecord(
                    file_path=final_h5_path,
                    short_name=f"model_a_final_output_{year}.h5",
                    unique_id=make_unique_id(final_h5_path),
                    year=year,
                ),
            ]
        )


# ============================================================================
# MODEL B
# ============================================================================


class DummyModelBPreprocessor(GenericPreprocessor):
    def __init__(self, model_name, config, state):
        super().__init__(model_name, state)
        self.config = config

    def copy_data_to_mutable_location(
        self, settings: dict, output_dir: str
    ) -> Tuple[RecordStore, RecordStore]:
        return RecordStore(), RecordStore()

    @provenance_logging
    def _preprocess(
        self, workspace: DummyWorkspace, previous_records: RecordStore = RecordStore()
    ) -> RecordStore:
        csv_record = get_record_by_short_name(
            previous_records, f"model_a_final_output_{self.state.current_year}.csv"
        )
        h5_record = get_record_by_short_name(
            previous_records, f"model_a_final_output_{self.state.current_year}.h5"
        )
        if not csv_record or not h5_record:
            raise ValueError("Expected Model A final outputs for Model B")

        return previous_records


class DummyModelBRunner(GenericRunner):
    def __init__(self, model_name, config, state):
        super().__init__(model_name, state)
        self.config = config

    @provenance_logging
    def _run(self, store: RecordStore, workspace: DummyWorkspace) -> RecordStore:
        year = self.state.current_year
        csv_record = get_record_by_short_name(store, f"model_a_final_output_{year}.csv")
        h5_record = get_record_by_short_name(store, f"model_a_final_output_{year}.h5")
        if not csv_record or not h5_record:
            raise ValueError("Expected Model A final outputs for Model B runner")

        df_csv = pd.read_csv(csv_record.file_path)
        df_csv["b_value"] = df_csv["a"] * 3
        output_csv_path = os.path.join(
            workspace.output_dir, f"model_b_output_{year}.csv"
        )
        df_csv.to_csv(output_csv_path, index=False)

        with h5py.File(h5_record.file_path, "r") as f_in:
            table_data = f_in["table1_final"][()]
        df_table = pd.DataFrame(table_data)
        df_table_modified = df_table - 1
        output_h5_path = os.path.join(workspace.output_dir, f"model_b_output_{year}.h5")
        with h5py.File(output_h5_path, "w") as f_out:
            f_out.create_dataset(
                "table_b_modified", data=df_table_modified.to_records(index=False)
            )

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=output_csv_path,
                    short_name=f"model_b_output_{year}.csv",
                    unique_id=make_unique_id(output_csv_path),
                    year=year,
                ),
                FileRecord(
                    file_path=output_h5_path,
                    short_name=f"model_b_output_{year}.h5",
                    unique_id=make_unique_id(output_h5_path),
                    year=year,
                ),
            ]
        )


class DummyModelBPostprocessor(GenericPostprocessor):
    def __init__(self, model_name, config, state):
        super().__init__(model_name, state)
        self.config = config

    @provenance_logging
    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: DummyWorkspace,
        runInfo=None,
        model_run_hash=None,
    ) -> RecordStore:
        year = self.state.current_year
        output_csv_record = get_record_by_short_name(
            raw_outputs, f"model_b_output_{year}.csv"
        )
        output_h5_record = get_record_by_short_name(
            raw_outputs, f"model_b_output_{year}.h5"
        )
        if not output_csv_record or not output_h5_record:
            raise ValueError("Expected Model B runner outputs")

        df_csv = pd.read_csv(output_csv_record.file_path)
        row_count = len(df_csv)
        final_txt_path = os.path.join(
            workspace.output_dir, f"model_b_final_output_{year}.txt"
        )
        with open(final_txt_path, "w") as f:
            f.write(f"Row count from Model B CSV: {row_count}")

        with h5py.File(output_h5_record.file_path, "r") as f_in:
            table_data = f_in["table_b_modified"][()]
        df_table = pd.DataFrame(table_data)
        total_sum = df_table.values.sum()
        final_h5_summary_path = os.path.join(
            workspace.output_dir, f"model_b_final_output_summary_{year}.txt"
        )
        with open(final_h5_summary_path, "w") as f:
            f.write(f"Total sum from Model B H5: {total_sum}")

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=final_txt_path,
                    short_name=f"model_b_final_output_{year}.txt",
                    unique_id=make_unique_id(final_txt_path),
                    year=year,
                ),
                FileRecord(
                    file_path=final_h5_summary_path,
                    short_name=f"model_b_final_output_summary_{year}.txt",
                    unique_id=make_unique_id(final_h5_summary_path),
                    year=year,
                ),
            ]
        )


# ============================================================================
# TESTS
# ============================================================================


@pytest.fixture
def setup_workflow():
    consist = pytest.importorskip("consist")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "workflow_output"
        output_dir.mkdir()
        db_path = Path(tmpdir) / "provenance.duckdb"

        tracker = consist.Tracker(
            run_dir=output_dir,
            db_path=str(db_path),
            mounts={
                "inputs": str(Path.cwd()),
                "workspace": str(output_dir),
            },
            project_root=str(Path.cwd()),
        )
        yield output_dir, db_path, tracker


def _build_model_a(model_config, state):
    return (
        DummyModelAPreprocessor("ModelA", model_config, state),
        DummyModelARunner("ModelA", model_config, state),
        DummyModelAPostprocessor("ModelA", model_config, state),
    )


def _build_model_b(state):
    return (
        DummyModelBPreprocessor("ModelB", {}, state),
        DummyModelBRunner("ModelB", {}, state),
        DummyModelBPostprocessor("ModelB", {}, state),
    )


def _run_model_a_step(
    store: RecordStore,
    workspace: DummyWorkspace,
    runner: DummyModelARunner,
    output_holder: dict,
) -> None:
    output_holder["outputs"] = runner.run(store, workspace)


def test_dummy_workflow_end_to_end(setup_workflow):
    output_dir, db_path, tracker = setup_workflow
    input_data_dir = Path(__file__).resolve().parent / "fixtures" / "dummy_workflow"
    year = 2025

    state = DummyWorkflowState(current_year=year)
    state.full_settings.shared.database.path = str(db_path)
    workspace = DummyWorkspace(output_dir=str(output_dir))

    model_config = {"input_dir": str(input_data_dir)}
    model_a_pre, model_a_run, model_a_post = _build_model_a(model_config, state)
    model_b_pre, model_b_run, model_b_post = _build_model_b(state)

    with cr.use_tracker(tracker):
        with cr.scenario(
            name="dummy_workflow",
            config={"year": year},
            tags=["dummy_workflow"],
            model="test_orchestrator",
        ) as scenario:
            scenario.add_input(input_data_dir / "data.csv", key="data.csv")
            scenario.add_input(input_data_dir / "data.h5", key="data.h5")

            with scenario.trace("initialization"):
                inputs, mutable = model_a_pre.copy_data_to_mutable_location(
                    model_config, str(output_dir)
                )
                cr.log_artifacts(inputs.to_mapping(), direction="input")
                cr.log_artifacts(mutable.to_mapping(), direction="output")

            with scenario.trace("preprocess_a"):
                recs = model_a_pre.preprocess(workspace, previous_records=mutable)

            # Use scenario.run once to show the preferred Consist entrypoint
            # (trace is still useful for inline orchestration steps).
            run_a_holder = {"outputs": None}
            scenario.run(
                fn=_run_model_a_step,
                name="run_a",
                model="modela",
                execution_options=ExecutionOptions(
                    runtime_kwargs={
                        "store": recs,
                        "workspace": workspace,
                        "runner": model_a_run,
                        "output_holder": run_a_holder,
                    }
                ),
                cache_options=CacheOptions(cache_mode="overwrite"),
            )
            recs = run_a_holder["outputs"]

            with scenario.trace("postprocess_a"):
                recs = model_a_post.postprocess(recs, workspace)

            with scenario.trace("preprocess_b"):
                recs = model_b_pre.preprocess(workspace, previous_records=recs)

            with scenario.trace("run_b"):
                recs = model_b_run.run(recs, workspace)

            with scenario.trace("postprocess_b"):
                model_b_post.postprocess(recs, workspace)

    assert (output_dir / f"model_a_final_output_{year}.csv").exists()
    assert (output_dir / f"model_a_final_output_{year}.h5").exists()
    assert (output_dir / f"model_b_final_output_{year}.txt").exists()
    assert (output_dir / f"model_b_final_output_summary_{year}.txt").exists()


def test_dummy_workflow_lineage(setup_workflow):
    output_dir, db_path, tracker = setup_workflow
    input_data_dir = Path(__file__).resolve().parent / "fixtures" / "dummy_workflow"
    year = 2025

    state = DummyWorkflowState(current_year=year)
    state.full_settings.shared.database.path = str(db_path)
    workspace = DummyWorkspace(output_dir=str(output_dir))

    model_config = {"input_dir": str(input_data_dir)}
    model_a_pre, model_a_run, model_a_post = _build_model_a(model_config, state)

    with cr.use_tracker(tracker):
        with cr.scenario(
            name="dummy_lineage",
            config={"year": year},
            tags=["dummy_lineage"],
            model="test_orchestrator",
        ) as scenario:
            scenario.add_input(input_data_dir / "data.csv", key="data.csv")
            scenario.add_input(input_data_dir / "data.h5", key="data.h5")

            with scenario.trace("initialization"):
                inputs, mutable = model_a_pre.copy_data_to_mutable_location(
                    model_config, str(output_dir)
                )
                cr.log_artifacts(inputs.to_mapping(), direction="input")
                cr.log_artifacts(mutable.to_mapping(), direction="output")

            with scenario.trace("preprocess_a"):
                recs = model_a_pre.preprocess(workspace, previous_records=mutable)

            with scenario.trace("run_a"):
                recs = model_a_run.run(recs, workspace)

            with scenario.trace("postprocess_a"):
                model_a_post.postprocess(recs, workspace)

    runs = tracker.find_runs(tags=["dummy_lineage"])
    assert len(runs) >= 1
    scenario_run = runs[0]
    assert scenario_run.status == "completed"
    assert "steps" in scenario_run.meta

    steps = scenario_run.meta["steps"]
    assert steps, "Expected steps recorded in scenario metadata"

    for step in steps:
        step_artifacts = tracker.get_artifacts_for_run(step["id"])
        assert len(step_artifacts.inputs) > 0
        assert len(step_artifacts.outputs) > 0

    scenario_artifacts = tracker.get_artifacts_for_run(scenario_run.id)
    assert len(scenario_artifacts.inputs) > 0
    assert len(scenario_artifacts.outputs) > 0
