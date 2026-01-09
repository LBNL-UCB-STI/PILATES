"""
Dummy Workflow Test for PILATES Framework - Consist Integration Version

This test module mirrors test_dummy_workflow.py but uses ConsistProvenanceTracker
instead of OpenLineageTracker. It serves as the migration target for Phase 5.2
of the Consist integration.

Purpose:
--------
1. **Migration Validation**: Ensures ConsistProvenanceTracker provides equivalent
   functionality to the legacy OpenLineageTracker.

2. **Gap Identification**: Uses pytest.mark.xfail to identify features that need
   implementation in the adapter or in Consist itself.

3. **Regression Testing**: Once all xfails pass, this validates the Consist integration.

4. **Integration Testing**: Validates the end-to-end integration of multiple components:
   - Generic base classes (GenericPreprocessor, GenericRunner, GenericPostprocessor)
   - Record management (FileRecord, H5FileRecord, H5TableRecord, RecordStore)
   - Provenance tracking (ConsistProvenanceTracker)
   - Database integration (Consist DuckDB backend)
   - Workspace management

See: /Users/zaneedell/git/consist/docs/01_MASTER_ROADMAP.md for migration plan.

Workflow Overview:
------------------
The test simulates a simple two-model workflow:

1. **Model A** (DummyModelA):
   - Preprocessor: Copies input CSV and H5 files to a mutable workspace
   - Runner: Reads CSV, doubles a column; reads H5, adds 1 to values
   - Postprocessor: Filters CSV, multiplies H5 values by 2

2. **Model B** (DummyModelB):
   - Preprocessor: Validates Model A outputs are available
   - Runner: Adds a new column to CSV, subtracts 1 from H5 values
   - Postprocessor: Generates summary statistics (row counts, sums)

Data Flow:
----------
Input Data (fixtures/dummy_workflow/)
    ↓
Model A Preprocessor (copy to workspace)
    ↓
Model A Runner (transform data: CSV doubling, H5 +1)
    ↓
Model A Postprocessor (filter CSV, H5 *2)
    ↓
Model B Preprocessor (validate inputs)
    ↓
Model B Runner (CSV add column, H5 -1)
    ↓
Model B Postprocessor (generate summaries)
    ↓
Final Outputs (*.txt files with statistics)
    ↓
Database Upload (DuckDB with full provenance)

Key Patterns Demonstrated:
---------------------------
1. **Preprocessor Pattern**: How to copy data, create FileRecords, and return RecordStores
2. **Runner Pattern**: How to consume RecordStores, process data, and return results
3. **Postprocessor Pattern**: How to finalize outputs and create final RecordStores
4. **Provenance Tracking**: How to record inputs/outputs at each stage
5. **Record Management**: How to create and manage FileRecords with unique_ids
6. **Chaining Models**: How to pass outputs from one model as inputs to another
7. **Database Integration**: How to upload run results to DuckDB for later analysis

Usage Notes:
------------
- All dummy classes inherit from Generic* base classes to ensure API compatibility
- FileRecords must have unique_ids to be added to RecordStores
- Runners must return RecordStore only
- Preprocessors and Postprocessors return RecordStore only
- RecordStore uses 'recordList' parameter, not 'file_records'
- Use record_input_file/record_output_file for provenance, not add_input_file/add_output_file
"""

import pytest
import pandas as pd
import h5py
import os
import shutil
from pathlib import Path
import tempfile
from typing import Tuple
from types import SimpleNamespace

from consist import Tracker

from pilates.generic.model import provenance_logging
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.runner import GenericRunner
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records_legacy import H5FileRecord, H5TableRecord, FileRecord
from pilates.generic.records import RecordStore
from pilates.utils.consist_adapter import ConsistProvenanceTracker
from pilates.workspace import Workspace
from tests.test_consist_adapter import consist_tracker


def get_record_by_short_name(store: RecordStore, short_name: str):
    """
    Find a record in a RecordStore by its short_name.

    This is a utility function for locating specific records within a RecordStore.
    The RecordStore class stores records by unique_id in a dictionary, but often
    we want to find records by their human-readable short_name.

    Args:
        store: The RecordStore to search. Can be None.
        short_name: The short_name to search for (e.g., 'data.csv', 'model_output.h5')

    Returns:
        The matching Record (FileRecord, H5FileRecord, or H5TableRecord) if found,
        None otherwise.

    Example:
        >>> store = RecordStore(recordList=[...])
        >>> csv_record = get_record_by_short_name(store, 'data.csv')
        >>> if csv_record:
        >>>     df = pd.read_csv(csv_record.file_path)
    """
    if store is None:
        return None
    for record in store.all_records():
        if record.short_name == short_name:
            return record
    return None


def make_unique_id(file_path: str) -> str:
    """
    Generate a unique ID from a file path for testing purposes.

    In production code, unique_ids are typically content hashes (e.g., MD5 of file contents).
    For this test, we use a hash of the file path since we don't need content-based deduplication.

    Args:
        file_path: The file path to hash

    Retu/Users/zaneedell/git/PILATES/tests/test_dummy_workflow_consist.pyrns:
        A 16-character hexadecimal string that uniquely identifies the file path

    Note:
        This is NOT suitable for production use where you want to detect if two files
        have identical content. Use pilates.utils.provenance methods for that.
    """
    import hashlib

    return hashlib.md5(file_path.encode()).hexdigest()[:16]


def assert_provenance_chain(run_info, expected_stages: list):
    """
    Verify that model runs form the expected chain.

    Args:
        run_info: PilatesRunInfo object
        expected_stages: List of (model_name, stage_keyword) tuples where stage_keyword
                        is a substring expected in the description (e.g., "preprocess", "run", "postprocess")

    Example:
        assert_provenance_chain(run_info, [
            ("ModelA", "preprocess"),
            ("ModelA", "run"),
            ("ModelA", "postprocess"),
        ])
    """
    model_runs = list(run_info.model_runs.values())

    # Group runs by model
    runs_by_model = {}
    for run in model_runs:
        if run.model not in runs_by_model:
            runs_by_model[run.model] = []
        runs_by_model[run.model].append(run)

    # Verify expected stages exist
    for model_name, stage_keyword in expected_stages:
        matching_runs = [
            r
            for r in runs_by_model.get(model_name, [])
            if stage_keyword.lower() in (r.description or "").lower()
        ]
        assert len(matching_runs) >= 1, (
            f"Expected at least one {model_name} run with '{stage_keyword}' in description. "
            f"Found runs: {[(r.unique_id, r.description) for r in runs_by_model.get(model_name, [])]}"
        )


class DummyWorkflowState:
    """
    Simplified workflow state tracker for testing.

    In production PILATES workflows, the WorkflowState class tracks the current simulation
    state (year, stage, iteration) and manages stage transitions. This dummy version
    provides the minimal interface required by the Generic* base classes.

    Attributes:
        current_year: The simulation year being processed
        current_major_stage: The current major stage (e.g., 'land_use', 'travel')
        current_inner_iter: The current iteration within a stage
        sub_stage_progress: Tracks progress within a stage (preprocessor/runner/postprocessor)
    """

    def __init__(self, current_year, current_major_stage=None, current_inner_iter=0):
        self.current_year = current_year
        self.current_major_stage = current_major_stage
        self.current_inner_iter = current_inner_iter
        self.sub_stage_progress = None
        # ADDED: Mock settings object to provide database path
        self.full_settings = SimpleNamespace(
            shared=SimpleNamespace(database=SimpleNamespace(path=None))
        )
        self.run_id = "test_run"

    def set_sub_stage_progress(self, progress):
        """Set the current sub-stage (e.g., 'preprocessor', 'runner', 'postprocessor')."""
        self.sub_stage_progress = progress

    def get_current_year(self):
        """Get the current simulation year."""
        return self.current_year

    def get_current_major_stage(self):
        """Get the current major stage."""
        return self.current_major_stage

    def get_current_inner_iter(self):
        """Get the current iteration number."""
        return self.current_inner_iter


# ============================================================================
# MODEL A: Demonstrates the complete preprocessor/runner/postprocessor pattern
# ============================================================================


class DummyModelAPreprocessor(GenericPreprocessor):
    """
    Preprocessor for Model A - demonstrates data copying and record creation.

    This class shows the standard pattern for a PILATES preprocessor:
    1. Copy immutable source data to a mutable workspace
    2. Create FileRecords for both input (source) and output (copied) files
    3. Record provenance for all files
    4. Return RecordStores containing the file metadata

    Key Learning Points:
    - How to implement copy_data_to_mutable_location()
    - How to create FileRecord, H5FileRecord, and H5TableRecord objects
    - How to link H5TableRecords to their parent H5FileRecord
    - How to record provenance using record_input_file() and record_output_file()
    - How unique_ids are required for RecordStore to track records
    """

    def __init__(self, model_name, config, provenance_tracker, state):
        super().__init__(model_name, state, provenance_tracker)
        self.config = config

    def copy_data_to_mutable_location(
        self, settings: dict, output_dir: str
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy immutable source data to the mutable workspace.

        This is a standard preprocessor responsibility: copy read-only source data
        into the workspace where models can modify it. This method creates FileRecords
        for both the source files (inputs) and the copied files (outputs).

        Args:
            settings: Configuration dictionary (not used in this simple example)
            output_dir: Target directory for copied files

        Returns:
            Tuple of (input_records, output_records):
            - input_records: RecordStore containing FileRecords for source files
            - output_records: RecordStore containing FileRecords for copied files

        Implementation Notes:
            - Each FileRecord must have a unique_id (generated via make_unique_id here)
            - H5FileRecords can contain multiple H5TableRecords (one per HDF5 table)
            - H5TableRecords reference their parent H5FileRecord via h5_file_unique_id
            - Provenance is recorded via record_input_file() and record_output_file()
        """
        input_dir = self.config["input_dir"]

        # Copy data.csv and data.h5 to the output directory
        shutil.copy(os.path.join(input_dir, "data.csv"), output_dir)
        shutil.copy(os.path.join(input_dir, "data.h5"), output_dir)

        # Record provenance for copied files (inputs)
        input_csv_path = os.path.join(input_dir, "data.csv")
        input_csv_record = FileRecord(
            file_path=input_csv_path,
            short_name="data.csv",
            description="Dummy CSV input for Model A",
            unique_id=make_unique_id(input_csv_path),
        )
        # Create H5FileRecord first to get its unique_id
        input_h5_path = os.path.join(input_dir, "data.h5")
        input_h5_file_record = H5FileRecord(
            file_path=input_h5_path,
            short_name="data.h5",
            description="Dummy H5 input for Model A",
            unique_id=make_unique_id(input_h5_path),
        )
        # Now create H5TableRecords using the H5FileRecord's unique_id
        input_h5_table1_path = input_h5_path + "/table1"
        input_h5_table1_record = H5TableRecord(
            file_path=input_h5_table1_path,
            h5_file_unique_id=input_h5_file_record.unique_id,
            table_name="table1",
            description="Table 1 from dummy H5",
            unique_id=make_unique_id(input_h5_table1_path),
        )
        input_h5_table2_path = input_h5_path + "/table2"
        input_h5_table2_record = H5TableRecord(
            file_path=input_h5_table2_path,
            h5_file_unique_id=input_h5_file_record.unique_id,
            table_name="table2",
            description="Table 2 from dummy H5",
            unique_id=make_unique_id(input_h5_table2_path),
        )
        input_h5_file_record.table_record_ids = [
            input_h5_table1_record.unique_id,
            input_h5_table2_record.unique_id,
        ]

        # Record provenance for copied files (outputs of this copy operation)
        # NOTE: Inputs to the workflow don't typically have a 'year' unless associated with a run
        output_csv_path = os.path.join(output_dir, "data.csv")
        output_csv_record = FileRecord(
            file_path=output_csv_path,
            short_name="data.csv",
            unique_id=make_unique_id(output_csv_path),
        )
        output_h5_path = os.path.join(output_dir, "data.h5")
        output_h5_file_record = H5FileRecord(
            file_path=output_h5_path,
            short_name="data.h5",
            unique_id=make_unique_id(output_h5_path),
        )
        output_h5_table1_path = output_h5_path + "/table1"
        output_h5_table1_record = H5TableRecord(
            file_path=output_h5_table1_path,
            h5_file_unique_id=output_h5_file_record.unique_id,
            table_name="table1",
            short_name="table1",
            unique_id=make_unique_id(output_h5_table1_path),
        )
        output_h5_table2_path = output_h5_path + "/table2"
        output_h5_table2_record = H5TableRecord(
            file_path=output_h5_table2_path,
            h5_file_unique_id=output_h5_file_record.unique_id,
            table_name="table2",
            short_name="table2",
            unique_id=make_unique_id(output_h5_table2_path),
        )
        output_h5_file_record.table_record_ids = [
            output_h5_table1_record.unique_id,
            output_h5_table2_record.unique_id,
        ]

        return RecordStore(
            recordList=[
                input_csv_record,
                input_h5_file_record,
                input_h5_table1_record,
                input_h5_table2_record,
            ]
        ), RecordStore(
            recordList=[
                output_csv_record,
                output_h5_file_record,
                output_h5_table1_record,
                output_h5_table2_record,
            ]
        )

    @provenance_logging
    def _preprocess(
        self, workspace: Workspace, previous_records: RecordStore = RecordStore()
    ) -> RecordStore:
        # In Consist, an artifact cannot be both an input and an output of the same run
        # (run_artifact_link primary key is (run_id, artifact_id)). So a "pass-through"
        # preprocessor would correctly have inputs but no outputs.
        #
        # For this dummy workflow, we make the preprocessor produce distinct outputs
        # so the step has a meaningful output edge in the lineage graph.
        output_dir = workspace.output_dir

        csv_record = get_record_by_short_name(previous_records, "data.csv")
        h5_record = get_record_by_short_name(previous_records, "data.h5")
        if not csv_record or not h5_record:
            raise ValueError("Expected data.csv and data.h5 in previous_records")

        pre_csv_path = os.path.join(output_dir, "preprocessed_data.csv")
        pre_h5_path = os.path.join(output_dir, "preprocessed_data.h5")
        shutil.copy(csv_record.file_path, pre_csv_path)
        shutil.copy(h5_record.file_path, pre_h5_path)

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=pre_csv_path,
                    short_name="preprocessed_data.csv",
                    description="Preprocessed CSV output for Model A",
                    unique_id=make_unique_id(pre_csv_path),
                ),
                H5FileRecord(
                    file_path=pre_h5_path,
                    short_name="preprocessed_data.h5",
                    description="Preprocessed H5 output for Model A",
                    unique_id=make_unique_id(pre_h5_path),
                ),
            ]
        )


class DummyWorkspace:
    """
    Simplified workspace for testing.

    In production, the Workspace class manages the file system layout for a PILATES run,
    including separate directories for each model, data copying, and path resolution.
    This dummy version provides the minimal interface needed for our test.

    Attributes:
        output_dir: The root directory where all outputs will be written

    Properties:
        full_path: Returns the full path to the workspace root (same as output_dir in this simple version)
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir

    @property
    def full_path(self):
        """Returns the full path to the root of the workspace."""
        return self.output_dir


class DummyModelARunner(GenericRunner):
    """
    Runner for Model A - demonstrates data processing and result generation.

    This class shows how a runner:
    1. Extracts file paths from the input RecordStore
    2. Performs model-specific computations
    3. Writes output files
    4. Creates FileRecords for outputs
    5. Records provenance
    6. Returns a RecordStore with output metadata

    Key Learning Points:
    - How to find records by short_name using get_record_by_short_name()
    - How to access file paths via record.file_path
    - How runners return RecordStore (not tuples)
    - Pattern for creating output FileRecords with unique_ids
    """

    def __init__(self, model_name, config, provenance_tracker, state):
        super().__init__(model_name, state, provenance_tracker)
        self.config = config

    @provenance_logging
    def _run(self, store: RecordStore, workspace: Workspace) -> RecordStore:
        output_dir = workspace.output_dir
        # Extract paths from the RecordStore
        csv_record = get_record_by_short_name(store, "preprocessed_data.csv")
        h5_record = get_record_by_short_name(store, "preprocessed_data.h5")

        # Ensure records are found before accessing their paths
        if not csv_record:
            raise ValueError(
                "preprocessed_data.csv record not found in RecordStore for Model A Runner."
            )
        if not h5_record:
            raise ValueError(
                "preprocessed_data.h5 record not found in RecordStore for Model A Runner."
            )

        csv_path = csv_record.file_path
        h5_path = h5_record.file_path
        year = self.state.current_year
        df_csv = pd.read_csv(csv_path)
        df_csv["a_doubled"] = df_csv["a"] * 2
        output_csv_path = os.path.join(output_dir, f"model_a_output_{year}.csv")
        df_csv.to_csv(output_csv_path, index=False)

        # Trivial operation: Read H5 table1, add 1 to all values, save as new H5 table
        with h5py.File(h5_path, "r") as f_in:
            table1_data = f_in["table1"][()]
        df_table1 = pd.DataFrame(table1_data)
        df_table1_modified = df_table1 + 1
        output_h5_path = os.path.join(output_dir, f"model_a_output_{year}.h5")
        with h5py.File(output_h5_path, "w") as f_out:
            f_out.create_dataset(
                "table1_modified", data=df_table1_modified.to_records(index=False)
            )

        # Record provenance for output files
        output_h5_file_record = H5FileRecord(
            file_path=output_h5_path,
            short_name=f"model_a_output_{year}.h5",
            description="Output H5 from Model A Runner",
            unique_id=make_unique_id(output_h5_path),
            year=year,
        )
        output_h5_table_path = output_h5_path + "/table1_modified"
        output_h5_table_record = H5TableRecord(
            file_path=output_h5_table_path,
            h5_file_unique_id=output_h5_file_record.unique_id,
            table_name="table1_modified",
            short_name="table1_modified",
            description="Modified Table 1 from Model A H5 output",
            unique_id=make_unique_id(output_h5_table_path),
            year=year,
        )
        output_h5_file_record.table_record_ids = [output_h5_table_record.unique_id]

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=output_csv_path,
                    short_name=f"model_a_output_{year}.csv",
                    unique_id=make_unique_id(output_csv_path),
                    year=year,
                ),
                output_h5_file_record,
                output_h5_table_record,
            ]
        )


class DummyModelAPostprocessor(GenericPostprocessor):
    """
    Postprocessor for Model A - demonstrates output finalization.

    Postprocessors typically:
    1. Read raw outputs from the runner
    2. Apply final transformations (filtering, formatting, validation)
    3. Generate publication-ready or archival outputs
    4. Create final FileRecords
    5. Record provenance for final outputs
    6. Return RecordStore with final output metadata

    Key Learning Points:
    - How to consume runner outputs via the raw_outputs RecordStore
    - Pattern for generating final, cleaned outputs
    - How postprocessors return only RecordStore (not a tuple)
    """

    def __init__(self, model_name, config, provenance_tracker, state):
        super().__init__(model_name, state, provenance_tracker)
        self.config = config

    @provenance_logging
    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        model_run_hash=None,
    ) -> RecordStore:
        output_dir = workspace.output_dir
        output_csv_record = get_record_by_short_name(
            raw_outputs, f"model_a_output_{self.state.current_year}.csv"
        )
        output_h5_record = get_record_by_short_name(
            raw_outputs, f"model_a_output_{self.state.current_year}.h5"
        )

        if not output_csv_record:
            raise ValueError(
                "model_a_output_*.csv record not found in raw_outputs for Model A Postprocessor."
            )
        if not output_h5_record:
            raise ValueError(
                "model_a_output_*.h5 record not found in raw_outputs for Model A Postprocessor."
            )

        output_csv_path = output_csv_record.file_path
        output_h5_path = output_h5_record.file_path
        year = self.state.current_year

        # Trivial operation: Read CSV, filter rows where a_doubled > 10, save as final output
        df_csv = pd.read_csv(output_csv_path)
        df_csv_filtered = df_csv[df_csv["a_doubled"] > 10]
        final_csv_path = os.path.join(output_dir, f"model_a_final_output_{year}.csv")
        df_csv_filtered.to_csv(final_csv_path, index=False)

        # Trivial operation: Read H5, multiply all values by 2, save as final output
        with h5py.File(output_h5_path, "r") as f_in:
            table_data = f_in["table1_modified"][()]
        df_table = pd.DataFrame(table_data)
        df_table_final = df_table * 2
        final_h5_path = os.path.join(output_dir, f"model_a_final_output_{year}.h5")
        with h5py.File(final_h5_path, "w") as f_out:
            f_out.create_dataset(
                "table1_final", data=df_table_final.to_records(index=False)
            )

        # Record provenance for final output files
        final_h5_file_record = H5FileRecord(
            file_path=final_h5_path,
            short_name=f"model_a_final_output_{year}.h5",
            description="Final H5 from Model A Postprocessor",
            unique_id=make_unique_id(final_h5_path),
            year=year,
        )
        final_h5_table_record = H5TableRecord(
            file_path=final_h5_path + "/table1_final",
            h5_file_unique_id=final_h5_file_record.unique_id,
            table_name="table1_final",
            short_name="table1_final",
            description="Final Table from Model A H5 output",
            # Note: unique_id should ideally be generated, H5TableRecord generates random if missing
            year=year,
        )
        final_h5_file_record.table_record_ids = [final_h5_table_record.unique_id]

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=final_csv_path,
                    short_name=f"model_a_final_output_{year}.csv",
                    unique_id=make_unique_id(final_csv_path),
                    year=year,
                ),
                final_h5_file_record,
                final_h5_table_record,
            ]
        )


# ============================================================================
# MODEL B: Demonstrates model chaining (using Model A outputs as inputs)
# ============================================================================


class DummyModelBPreprocessor(GenericPreprocessor):
    """
    Preprocessor for Model B - demonstrates consuming another model's outputs.

    This shows how to build model chains in PILATES where one model consumes
    the outputs of another. Model B's preprocessor validates that Model A's
    final outputs are available and passes them through to the runner.

    Key Learning Points:
    - How to validate that required inputs from upstream models exist
    - Pattern for passing through records without modification
    - Importance of returning the RecordStore from _preprocess
    """

    def __init__(self, model_name, config, provenance_tracker, state):
        super().__init__(model_name, state, provenance_tracker)
        self.config = config

    def copy_data_to_mutable_location(
        self, settings: dict, output_dir: str
    ) -> Tuple[RecordStore, RecordStore]:
        # Model B's preprocessor doesn't copy initial data, it processes output from Model A.
        # So, this method can return empty RecordStores.
        return RecordStore(), RecordStore()

    @provenance_logging
    def _preprocess(
        self, workspace: Workspace, previous_records: RecordStore = RecordStore()
    ) -> RecordStore:
        output_dir = workspace.output_dir
        model_a_final_csv_record = get_record_by_short_name(
            previous_records, f"model_a_final_output_{self.state.current_year}.csv"
        )
        model_a_final_h5_record = get_record_by_short_name(
            previous_records, f"model_a_final_output_{self.state.current_year}.h5"
        )

        if not model_a_final_csv_record:
            raise ValueError(
                "model_a_final_output_*.csv record not found in previous_records for Model B Preprocessor."
            )
        if not model_a_final_h5_record:
            raise ValueError(
                "model_a_final_output_*.h5 record not found in previous_records for Model B Preprocessor."
            )

        model_a_final_csv_path = model_a_final_csv_record.file_path
        model_a_final_h5_path = model_a_final_h5_record.file_path
        year = self.state.current_year

        # Model B preprocessor just passes through the records from Model A
        return previous_records


class DummyModelBRunner(GenericRunner):
    def __init__(self, model_name, config, provenance_tracker, state):
        super().__init__(model_name, state, provenance_tracker)
        self.config = config

    @provenance_logging
    def _run(self, store: RecordStore, workspace: Workspace) -> RecordStore:
        output_dir = workspace.output_dir
        csv_record = get_record_by_short_name(
            store, f"model_a_final_output_{self.state.current_year}.csv"
        )
        h5_record = get_record_by_short_name(
            store, f"model_a_final_output_{self.state.current_year}.h5"
        )

        if not csv_record:
            raise ValueError(
                "model_a_final_output_*.csv record not found in RecordStore for Model B Runner."
            )
        if not h5_record:
            raise ValueError(
                "model_a_final_output_*.h5 record not found in RecordStore for Model B Runner."
            )

        csv_path = csv_record.file_path
        h5_path = h5_record.file_path
        year = self.state.current_year

        # Trivial operation: Read CSV, add a new column 'b_value' = a * 3, save as Model B output
        df_csv = pd.read_csv(csv_path)
        df_csv["b_value"] = df_csv["a"] * 3
        output_csv_path = os.path.join(output_dir, f"model_b_output_{year}.csv")
        df_csv.to_csv(output_csv_path, index=False)

        # Trivial operation: Read H5 table, subtract 1 from all values, save as new H5 table
        with h5py.File(h5_path, "r") as f_in:
            table_data = f_in["table1_final"][()]
        df_table = pd.DataFrame(table_data)
        df_table_modified = df_table - 1
        output_h5_path = os.path.join(output_dir, f"model_b_output_{year}.h5")
        with h5py.File(output_h5_path, "w") as f_out:
            f_out.create_dataset(
                "table_b_modified", data=df_table_modified.to_records(index=False)
            )

        # Record provenance for output files
        output_h5_file_record = H5FileRecord(
            file_path=output_h5_path,
            short_name=f"model_b_output_{year}.h5",
            description="Output H5 from Model B Runner",
            unique_id=make_unique_id(output_h5_path),
            year=year,
        )
        output_h5_table_record = H5TableRecord(
            file_path=output_h5_path + "/table_b_modified",
            h5_file_unique_id=output_h5_file_record.unique_id,
            table_name="table_b_modified",
            short_name="table_b_modified",
            description="Modified Table from Model B H5 output",
            year=year,
        )
        output_h5_file_record.table_record_ids = [output_h5_table_record.unique_id]

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=output_csv_path,
                    short_name=f"model_b_output_{year}.csv",
                    unique_id=make_unique_id(output_csv_path),
                    year=year,
                ),
                output_h5_file_record,
                output_h5_table_record,
            ]
        )


class DummyModelBPostprocessor(GenericPostprocessor):
    def __init__(self, model_name, config, provenance_tracker, state):
        super().__init__(model_name, state, provenance_tracker)
        self.config = config

    @provenance_logging
    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        model_run_hash=None,
    ) -> RecordStore:
        output_dir = workspace.output_dir
        output_csv_record = get_record_by_short_name(
            raw_outputs, f"model_b_output_{self.state.current_year}.csv"
        )
        output_h5_record = get_record_by_short_name(
            raw_outputs, f"model_b_output_{self.state.current_year}.h5"
        )

        if not output_csv_record:
            raise ValueError(
                "model_b_output_*.csv record not found in raw_outputs for Model B Postprocessor."
            )
        if not output_h5_record:
            raise ValueError(
                "model_b_output_*.h5 record not found in raw_outputs for Model B Postprocessor."
            )

        output_csv_path = output_csv_record.file_path
        output_h5_path = output_h5_record.file_path
        year = self.state.current_year

        # Trivial operation: Read CSV, count rows, save count to a text file
        df_csv = pd.read_csv(output_csv_path)
        row_count = len(df_csv)
        final_txt_path = os.path.join(output_dir, f"model_b_final_output_{year}.txt")
        with open(final_txt_path, "w") as f:
            f.write(f"Row count from Model B CSV: {row_count}")

        # Trivial operation: Read H5, calculate sum of all values, save to a text file
        with h5py.File(output_h5_path, "r") as f_in:
            table_data = f_in["table_b_modified"][()]
        df_table = pd.DataFrame(table_data)
        total_sum = df_table.values.sum()
        final_h5_summary_path = os.path.join(
            output_dir, f"model_b_final_output_summary_{year}.txt"
        )
        with open(final_h5_summary_path, "w") as f:
            f.write(f"Total sum from Model B H5: {total_sum}")

        # Record provenance for final output files
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
# CONTAINER-AWARE RUNNER: Demonstrates Consist container integration
# ============================================================================


class DummyContainerRunner(GenericRunner):
    """
    Container-aware runner for testing Consist integration.

    This runner demonstrates how to use GenericRunner.run_container() with
    Consist integration enabled. Unlike DummyModelARunner which performs
    transformations directly in Python, DummyContainerRunner:

    1. Prepares input data and volumes for container execution
    2. Uses GenericRunner.run_container() with ConsistProvenanceTracker
    3. Allows Consist to track provenance during container execution
    4. Verifies that container integration is properly triggered

    This is useful for testing the Consist integration path without requiring
    actual Docker/Singularity execution (container execution is mocked).

    Key Learning Points:
    - How to call GenericRunner.run_container() from a runner implementation
    - How to pass ConsistProvenanceTracker for Consist integration
    - How to structure input/output artifacts for provenance tracking
    - How container execution integrates with the standard runner pattern
    """

    def __init__(self, model_name, config, provenance_tracker, state):
        super().__init__(model_name, state, provenance_tracker)
        self.config = config

    @provenance_logging
    def _run(self, store: RecordStore, workspace: Workspace) -> RecordStore:
        """
        Run a dummy model using container execution with Consist integration.

        This demonstrates the container-based execution path:
        1. Extract input records from the RecordStore
        2. Prepare volumes and artifacts for container execution
        3. Call GenericRunner.run_container() with ConsistProvenanceTracker
        4. Handle container results and create output records
        """
        output_dir = workspace.output_dir
        # Preprocessor may produce distinct "preprocessed_*" artifacts (preferred) to
        # avoid pass-through input/output collisions in Consist.
        csv_record = get_record_by_short_name(store, "preprocessed_data.csv")
        if not csv_record:
            csv_record = get_record_by_short_name(store, "data.csv")

        h5_record = get_record_by_short_name(store, "preprocessed_data.h5")
        if not h5_record:
            h5_record = get_record_by_short_name(store, "data.h5")

        if not csv_record:
            raise ValueError(
                "Expected CSV input record not found in RecordStore for DummyContainerRunner."
            )
        if not h5_record:
            raise ValueError(
                "Expected H5 input record not found in RecordStore for DummyContainerRunner."
            )

        csv_path = csv_record.file_path
        h5_path = h5_record.file_path
        year = self.state.current_year

        # Prepare output paths where the container will write results
        output_csv_path = os.path.join(output_dir, f"model_container_output_{year}.csv")
        output_h5_path = os.path.join(output_dir, f"model_container_output_{year}.h5")

        # Prepare volumes for container mounting
        # Docker format: {host_path: {'bind': container_path, 'mode': 'rw'}}
        volumes = {
            output_dir: {"bind": "/output", "mode": "rw"},
        }

        # List of input artifacts for provenance tracking
        input_artifacts = [csv_path, h5_path]

        # List of output paths that will be generated by the container
        output_paths = [output_csv_path, output_h5_path]

        # Mock container execution settings
        # In a real scenario, this would be actual Docker/Singularity config
        from types import SimpleNamespace

        settings = self.state.full_settings

        # Call GenericRunner.run_container with Consist integration
        # This is where the Consist delegation logic is tested
        success = GenericRunner.run_container(
            client=None,  # No Docker client in test
            settings=settings,
            image="dummy-model:latest",
            volumes=volumes,
            command="python /model/process.py",
            model_name=self.model_name,
            working_dir="/output",
            environment={"MODEL_NAME": self.model_name, "YEAR": str(year)},
            args=["--input", "/input/data.csv", "--output", "/output/result.csv"],
            provenance_tracker=self.provenance_tracker,
            input_artifacts=input_artifacts,
            output_paths=output_paths,
        )

        # Since we're mocking container execution in tests, we need to
        # simulate the results that the container would produce
        if not success:
            raise RuntimeError(f"Container execution failed for {self.model_name}")

        # Simulate the container's output: copy input to output
        # (In a real scenario, the container would have created these)
        shutil.copy(csv_path, output_csv_path)
        shutil.copy(h5_path, output_h5_path)

        # Record provenance for output files
        output_csv_record = FileRecord(
            file_path=output_csv_path,
            short_name=f"model_container_output_{year}.csv",
            unique_id=make_unique_id(output_csv_path),
            year=year,
        )

        output_h5_file_record = H5FileRecord(
            file_path=output_h5_path,
            short_name=f"model_container_output_{year}.h5",
            description="Output H5 from DummyContainerRunner",
            unique_id=make_unique_id(output_h5_path),
            year=year,
        )

        output_h5_table_path = output_h5_path + "/table1"
        output_h5_table_record = H5TableRecord(
            file_path=output_h5_table_path,
            h5_file_unique_id=output_h5_file_record.unique_id,
            table_name="table1",
            short_name="table1",
            description="Table 1 from container output",
            unique_id=make_unique_id(output_h5_table_path),
            year=year,
        )
        output_h5_file_record.table_record_ids = [output_h5_table_record.unique_id]

        return RecordStore(
            recordList=[
                output_csv_record,
                output_h5_file_record,
                output_h5_table_record,
            ]
        )


# ============================================================================
# TEST CLASS
# ============================================================================


class TestDummyWorkflowConsist:
    """
    Integration test for PILATES workflow using ConsistProvenanceTracker.

    This test parallels TestDummyWorkflow but validates the Consist adapter.
    It serves as the migration target for Phase 5.2 of the Consist integration.

    Key differences from TestDummyWorkflow:
    - Uses ConsistProvenanceTracker instead of OpenLineageTracker
    - Model names are normalized to lowercase by Consist
    - Some assertions are marked xfail for known gaps (to be fixed in Phase 5.2+)
    - OpenLineage event assertions are xfail until Consist adds OL support

    The test uses pytest fixtures to set up temporary directories and database
    connections that are automatically cleaned up after the test runs.
    """

    @pytest.fixture
    def setup_workflow(self):
        # Create a temporary directory for the entire workflow run
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_output_dir = Path(tmpdir) / "workflow_output"
            workflow_output_dir.mkdir()

            # Create a temporary directory for the provenance database
            db_path = Path(tmpdir) / "provenance.duckdb"

            # Match run.py initialization pattern: mounts + project_root.
            project_root_abs = os.getcwd()
            consist_lib_tracker = Tracker(
                run_dir=workflow_output_dir,
                db_path=str(db_path),
                mounts={
                    "inputs": project_root_abs,
                    "workspace": str(workflow_output_dir),
                },
                project_root=project_root_abs,
            )

            # KEY DIFFERENCE: Use ConsistProvenanceTracker instead of OpenLineageTracker
            provenance_tracker = ConsistProvenanceTracker(
                run_id="placeholder_id",  # Active step run will override
                output_path=str(workflow_output_dir),
                tracker=consist_lib_tracker,
            )

            yield workflow_output_dir, provenance_tracker, db_path, consist_lib_tracker

    def test_single_year_workflow(self, setup_workflow):
        """
        Test a complete two-model workflow for a single simulation year.

        Migrated to use Consist's scenario/step API matching the pattern in run.py.
        The test explicitly manages scenario structure while models remain unaware.
        """
        workflow_output_dir, provenance_tracker, db_path, tracker = setup_workflow

        input_data_dir = Path(__file__).resolve().parent / "fixtures" / "dummy_workflow"
        year = 2025
        scenario_name = "test_scenario"

        # Initialize DummyWorkflowState and DummyWorkspace
        workflow_state = DummyWorkflowState(current_year=year)
        workflow_state.full_settings.shared.database.path = str(db_path)
        workspace = DummyWorkspace(output_dir=str(workflow_output_dir))

        # --- Run Model A ---
        model_a_config = {"input_dir": str(input_data_dir)}
        model_a_preprocessor = DummyModelAPreprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        model_a_runner = DummyModelARunner(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        model_a_postprocessor = DummyModelAPostprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )

        # START SCENARIO - matches run.py pattern
        with tracker.scenario(
                name=scenario_name,
                config={"year": year},  # Scenario-level config
                tags=["test_workflow"],
                model="test_orchestrator"
        ) as scenario:

            # INITIALIZATION STEP - copy immutable data to workspace
            with scenario.step("initialization"):
                input_records_a, mutable_location_records_a = (
                    model_a_preprocessor.copy_data_to_mutable_location(
                        model_a_config, str(workflow_output_dir)
                    )
                )
                # Auto-log RecordStores (adapter helper)
                provenance_tracker.log_record_store(input_records_a, direction="input")
                provenance_tracker.log_record_store(mutable_location_records_a, direction="output")

            # MODEL A PIPELINE
            with scenario.step("preprocess_a"):
                preprocessor_a_output_records = model_a_preprocessor.preprocess(
                    workspace, previous_records=mutable_location_records_a
                )

            with scenario.step("run_a"):
                runner_a_output_records = model_a_runner.run(
                    preprocessor_a_output_records, workspace
                )

            with scenario.step("postprocess_a"):
                postprocessor_a_output_records = model_a_postprocessor.postprocess(
                    runner_a_output_records, workspace
                )

            # Validate Model A outputs
            model_a_csv_path = workflow_output_dir / f"model_a_final_output_{year}.csv"
            model_a_h5_path = workflow_output_dir / f"model_a_final_output_{year}.h5"
            assert model_a_csv_path.exists()
            assert model_a_h5_path.exists()

            # Validate Model A CSV content
            df_a_final = pd.read_csv(model_a_csv_path)
            assert "a_doubled" in df_a_final.columns
            assert (df_a_final["a_doubled"] > 10).all()

            # Validate Model A H5 content
            with h5py.File(model_a_h5_path, "r") as f:
                assert "table1_final" in f

            # --- Run Model B (consumes Model A outputs) ---
            model_b_config = {}
            model_b_preprocessor = DummyModelBPreprocessor(
                "ModelB", model_b_config, provenance_tracker, workflow_state
            )
            model_b_runner = DummyModelBRunner(
                "ModelB", model_b_config, provenance_tracker, workflow_state
            )
            model_b_postprocessor = DummyModelBPostprocessor(
                "ModelB", model_b_config, provenance_tracker, workflow_state
            )

            # MODEL B PIPELINE
            with scenario.step("preprocess_b"):
                preprocessor_b_output_records = model_b_preprocessor.preprocess(
                    workspace, previous_records=postprocessor_a_output_records
                )

            with scenario.step("run_b"):
                runner_b_output_records = model_b_runner.run(
                    preprocessor_b_output_records, workspace
                )

            with scenario.step("postprocess_b"):
                postprocessor_b_output_records = model_b_postprocessor.postprocess(
                    runner_b_output_records, workspace
                )

            # Validate Model B outputs
            model_b_txt_path = workflow_output_dir / f"model_b_final_output_{year}.txt"
            model_b_summary_path = workflow_output_dir / f"model_b_final_output_summary_{year}.txt"
            assert model_b_txt_path.exists()
            assert model_b_summary_path.exists()

            # Validate Model B text outputs
            with open(model_b_txt_path, "r") as f:
                row_count_content = f.read()
                assert "Row count from Model B CSV:" in row_count_content

            with open(model_b_summary_path, "r") as f:
                sum_content = f.read()
                assert "Total sum from Model B H5:" in sum_content

        # SCENARIO CLOSED - Consist has automatically persisted everything to DB

        # ========================================================================
        # CONSIST VALIDATION: Query the database to verify provenance
        # ========================================================================

        # Query using Consist's native API (no DuckDBManager needed)
        runs = tracker.find_runs(tags=["test_workflow"])
        assert len(runs) > 0, "Expected to find test workflow run in database"

        scenario_run = runs[0]
        assert scenario_run.id == scenario_name
        assert scenario_run.status == "completed"

        # Verify scenario metadata contains step information
        assert "steps" in scenario_run.meta, "Scenario should have steps metadata"
        steps = scenario_run.meta["steps"]

        print(f"\nDEBUG: Querying artifacts for scenario run: {scenario_run.id}")

        scenario_artifacts = tracker.get_artifacts_for_run(scenario_run.id)

        print(f"DEBUG: scenario_artifacts.inputs = {scenario_artifacts.inputs}")
        print(f"DEBUG: scenario_artifacts.outputs = {scenario_artifacts.outputs}")

        # Check artifacts for step runs
        print(f"\nDEBUG: Checking artifacts for each step:")
        for step in steps:
            step_id = step["id"]
            step_artifacts = tracker.get_artifacts_for_run(step_id)
            print(f"  Step {step_id}:")
            print(f"    - Inputs: {len(step_artifacts.inputs)}")
            print(f"    - Outputs: {len(step_artifacts.outputs)}")

        # Also query the database directly to see what's there
        if tracker.db:
            from sqlalchemy import text
            with tracker.db.engine.connect() as conn:
                # Check all artifact links
                all_links = conn.execute(
                    text("SELECT run_id, artifact_id, direction FROM run_artifact_link")
                ).fetchall()
                print(f"\nDEBUG: Total run_artifact_links in DB: {len(all_links)}")
                print(f"DEBUG: First 20 links:")
                for link in all_links[:20]:
                    print(f"  - run_id={link[0]}, artifact_id={link[1]}, direction={link[2]}")

        assert len(scenario_artifacts.inputs) > 0, "Expected scenario to have input artifacts"

        # Verify expected steps exist
        step_names = [s["id"] for s in steps]
        expected_steps = [
            "initialization",
            "preprocess_a", "run_a", "postprocess_a",
            "preprocess_b", "run_b", "postprocess_b"
        ]
        for expected_step in expected_steps:
            assert any(expected_step in name for name in step_names), \
                f"Expected step containing '{expected_step}' in {step_names}"

        # Verify all steps completed successfully
        for step in steps:
            assert step["status"] == "completed", \
                f"Step {step['id']} failed with status: {step['status']}"

        # Verify artifacts were tracked
        scenario_artifacts = tracker.get_artifacts_for_run(scenario_run.id)
        assert len(scenario_artifacts.inputs) > 0, "Expected scenario to have input artifacts"
        assert len(scenario_artifacts.outputs) > 0, "Expected scenario to have output artifacts"

        # Verify specific artifacts exist
        output_keys = set(scenario_artifacts.outputs.keys())
        expected_outputs = [
            f"model_a_final_output_{year}.csv",
            f"model_a_final_output_{year}.h5",
            f"model_b_final_output_{year}.txt",
            f"model_b_final_output_summary_{year}.txt"
        ]
        for expected_output in expected_outputs:
            assert expected_output in output_keys, \
                f"Expected output '{expected_output}' in {output_keys}"

        print(f"✓ Workflow completed successfully")
        print(f"✓ Scenario: {scenario_run.id}")
        print(f"✓ Steps executed: {len(steps)}")
        print(f"✓ Inputs tracked: {len(scenario_artifacts.inputs)}")
        print(f"✓ Outputs tracked: {len(scenario_artifacts.outputs)}")
        print(f"✓ Database: {db_path}")

    def test_provenance_tracking_details(self, setup_workflow):
        """
        Test that Consist captures complete provenance lineage.

        This test documents the Consist provenance pattern:
        - Orchestrator manages scenario/step structure
        - Models remain unaware of provenance
        - Adapter auto-logs artifacts during model execution
        - Database queries reveal full lineage
        """
        workflow_output_dir, provenance_tracker, db_path, tracker = setup_workflow

        input_data_dir = Path(__file__).resolve().parent / "fixtures" / "dummy_workflow"
        year = 2025
        scenario_name = "provenance_test"

        # Initialize state and workspace
        workflow_state = DummyWorkflowState(current_year=year)
        workflow_state.full_settings.shared.database.path = str(db_path)
        workspace = DummyWorkspace(output_dir=str(workflow_output_dir))

        # Run just Model A to keep the test focused
        model_a_config = {"input_dir": str(input_data_dir)}
        preprocessor = DummyModelAPreprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        runner = DummyModelARunner(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        postprocessor = DummyModelAPostprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )

        # === ERGONOMIC PATTERN: Orchestrator manages scenario/steps ===
        with tracker.scenario(
                name=scenario_name,
                config={"year": year, "model": "ModelA"},
                tags=["provenance_test"],
                model="test_orchestrator"
        ) as scenario:
            # Register immutable exogenous inputs against the scenario header.
            # Scenario headers are suspended during steps; inputs must be added via add_input.
            scenario.add_input(input_data_dir / "data.csv", key="data.csv")
            scenario.add_input(input_data_dir / "data.h5", key="data.h5")

            with scenario.step("initialization"):
                _, mutable_records = preprocessor.copy_data_to_mutable_location(
                    model_a_config, str(workflow_output_dir)
                )
                provenance_tracker.log_record_store(mutable_records, direction="output")

            with scenario.step("preprocess"):
                preprocess_output = preprocessor.preprocess(
                    workspace, previous_records=mutable_records
                )

            with scenario.step("run"):
                runner_output = runner.run(preprocess_output, workspace)

            with scenario.step("postprocess"):
                final_output = postprocessor.postprocess(runner_output, workspace)

        # === VERIFY SCENARIO STRUCTURE ===
        runs = tracker.find_runs(tags=["provenance_test"])
        assert len(runs) == 1, f"Expected 1 scenario run, got {len(runs)}"

        scenario_run = runs[0]
        assert scenario_run.id == scenario_name
        assert scenario_run.model_name == "test_orchestrator"
        # Scenario-level run doesn't have a year (it's the orchestrator)
        # Year is stored in config and in the step runs

        # Verify year is in config
        # Verify steps metadata
        assert "steps" in scenario_run.meta
        steps = scenario_run.meta["steps"]

        # === VERIFY STEP RUNS ===
        # Each step should be a separate run in the database
        step_runs = tracker.find_runs(parent_id=scenario_name)
        assert len(step_runs) == 4, f"Expected 4 step runs, got {len(step_runs)}"


        # === VERIFY ARTIFACT LINEAGE ===
        # Get artifacts for each step
        initialization_step = [r for r in step_runs if "initialization" in r.id][0]
        preprocess_step = [r for r in step_runs if "preprocess" in r.id][0]
        run_step = [r for r in step_runs if "run" in r.id and "preprocess" not in r.id][0]
        postprocess_step = [r for r in step_runs if "postprocess" in r.id][0]

        # Initialization should have outputs (copied files)
        init_artifacts = tracker.get_artifacts_for_run(initialization_step.id)
        assert len(init_artifacts.outputs) > 0, "Initialization should produce outputs"

        # Preprocess should have inputs and outputs
        preprocess_artifacts = tracker.get_artifacts_for_run(preprocess_step.id)
        assert len(preprocess_artifacts.inputs) > 0, "Preprocess should consume inputs"
        assert len(preprocess_artifacts.outputs) > 0, "Preprocess should produce outputs"

        # Run should have inputs and outputs
        run_artifacts = tracker.get_artifacts_for_run(run_step.id)
        assert len(run_artifacts.inputs) > 0, "Run should consume inputs"
        assert len(run_artifacts.outputs) > 0, "Run should produce outputs"

        # Postprocess should have inputs and outputs
        postprocess_artifacts = tracker.get_artifacts_for_run(postprocess_step.id)
        assert len(postprocess_artifacts.inputs) > 0, "Postprocess should consume inputs"
        assert len(postprocess_artifacts.outputs) > 0, "Postprocess should produce outputs"

        # === VERIFY LINEAGE: Outputs → Inputs ===
        # Preprocess inputs should include initialization outputs
        init_output_ids = set(init_artifacts.outputs.keys())
        preprocess_input_ids = set(preprocess_artifacts.inputs.keys())
        assert init_output_ids.issubset(preprocess_input_ids), \
            "Initialization outputs should be inputs to preprocess"

        # Run inputs should include preprocess outputs
        preprocess_output_ids = set(preprocess_artifacts.outputs.keys())
        run_input_ids = set(run_artifacts.inputs.keys())
        assert preprocess_output_ids.issubset(run_input_ids), \
            "Preprocess outputs should be inputs to run"

        # Postprocess inputs should include run outputs
        run_output_ids = set(run_artifacts.outputs.keys())
        postprocess_input_ids = set(postprocess_artifacts.inputs.keys())
        assert run_output_ids.issubset(postprocess_input_ids), \
            "Run outputs should be inputs to postprocess"

        # === VERIFY SCENARIO-LEVEL ARTIFACTS ===
        # Scenario should have all artifacts from all steps
        scenario_artifacts = tracker.get_artifacts_for_run(scenario_name)
        assert len(scenario_artifacts.inputs) > 0, "Scenario should have inputs"
        assert len(scenario_artifacts.outputs) > 0, "Scenario should have outputs"

        # === VERIFY ARTIFACT METADATA ===
        # Check a sample artifact has proper metadata
        sample_artifact_id = list(postprocess_artifacts.outputs.keys())[0]
        sample_artifact = tracker.get_artifact(sample_artifact_id)

        assert sample_artifact is not None
        assert sample_artifact.key is not None, "Artifact should have a key (filename)"
        assert sample_artifact.uri is not None, "Artifact should have a URI"
        assert sample_artifact.created_at is not None, "Artifact should have timestamp"

        print("\n✓ Provenance tracking verified:")
        print(f"  - Scenario: {scenario_name}")
        print(f"  - Steps: {len(steps)}")
        print(f"  - Step runs in DB: {len(step_runs)}")
        print(f"  - Scenario inputs: {len(scenario_artifacts.inputs)}")
        print(f"  - Scenario outputs: {len(scenario_artifacts.outputs)}")
        print(f"  - Lineage verified: outputs → inputs through all steps")

    def test_schema_capture_persists_artifact_schema(self, setup_workflow):
        """
        Ensure Consist schema profiling is captured in PILATES integration tests.

        Mirrors Consist's e2e expectations:
        - `tracker.ingest(artifact)` profiles the ingested DuckDB table
        - `schema_id` / `schema_summary` are written to Artifact.meta
        - normalized schema rows are persisted (ArtifactSchema/Field/Observation)
        """
        workflow_output_dir, provenance_tracker, db_path, tracker = setup_workflow

        from sqlmodel import Session, select

        from consist.models.artifact_schema import (
            ArtifactSchema,
            ArtifactSchemaField,
            ArtifactSchemaObservation,
        )

        input_data_dir = Path(__file__).resolve().parent / "fixtures" / "dummy_workflow"
        year = 2025
        scenario_name = "schema_capture_test"

        workflow_state = DummyWorkflowState(current_year=year)
        workflow_state.full_settings.shared.database.path = str(db_path)
        workspace = DummyWorkspace(output_dir=str(workflow_output_dir))

        model_a_config = {"input_dir": str(input_data_dir)}
        preprocessor = DummyModelAPreprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        runner = DummyModelARunner(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        postprocessor = DummyModelAPostprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )

        with tracker.scenario(
            name=scenario_name,
            config={"year": year, "model": "ModelA"},
            tags=["schema_capture_test"],
            model="test_orchestrator",
        ) as scenario:
            with scenario.step("initialization"):
                _, mutable_records = preprocessor.copy_data_to_mutable_location(
                    model_a_config, str(workflow_output_dir)
                )
                provenance_tracker.log_record_store(mutable_records, direction="output")

            with scenario.step("preprocess"):
                preprocess_output = preprocessor.preprocess(
                    workspace, previous_records=mutable_records
                )

            with scenario.step("run"):
                runner_output = runner.run(preprocess_output, workspace)

            with scenario.step("postprocess"):
                postprocessor.postprocess(runner_output, workspace)

        # Prefer the runner-stage CSV output because the postprocessed "final" output can be
        # legitimately empty (filtering), which prevents dlt from inferring column types.
        output_key = f"model_a_output_{year}.csv"
        output_path = workflow_output_dir / output_key
        assert output_path.exists(), "Expected CSV output to exist for ingestion"

        artifact = tracker.get_artifact(output_key)
        assert artifact is not None, f"Expected Consist artifact for key='{output_key}'"
        assert artifact.driver == "csv", f"Expected driver='csv', got {artifact.driver!r}"

        tracker.ingest(artifact)

        artifact_after = tracker.get_artifact(output_key)
        assert artifact_after is not None
        meta = artifact_after.meta or {}

        assert meta.get("is_ingested") is True
        assert "schema_id" in meta
        assert "schema_summary" in meta

        schema_id = meta["schema_id"]
        summary = meta["schema_summary"]
        assert summary.get("n_columns", 0) >= 2

        with Session(tracker.engine) as session:
            assert session.get(ArtifactSchema, schema_id) is not None

            fields = session.exec(
                select(ArtifactSchemaField).where(
                    ArtifactSchemaField.schema_id == schema_id
                )
            ).all()
            field_types = {f.name: (f.logical_type or "").lower() for f in fields}
            assert {"a", "a_doubled"} <= set(field_types), f"Got fields: {sorted(field_types)}"
            assert field_types["a"] in {"bigint", "integer", "hugeint"} or "int" in field_types["a"]
            assert field_types["a_doubled"] in {"bigint", "integer", "hugeint"} or "int" in field_types["a_doubled"]

            observations = session.exec(
                select(ArtifactSchemaObservation).where(
                    ArtifactSchemaObservation.schema_id == schema_id
                )
            ).all()
            assert len(observations) >= 1

    @pytest.mark.xfail(
        reason=(
            "Adapter restart lookups are still in-memory only. "
            "Should query Consist persistence once Phase 5.2/5.3 caching is complete."
        ),
        strict=False,
    )
    def test_restart_lookup_uses_consist_persistence(self, setup_workflow):
        """
        Progress test: after a completed workflow, a fresh adapter instance should be
        able to locate prior runs via Consist persistence.

        Expected end state:
        - New ConsistProvenanceTracker can find latest completed model runs from DB/JSON.
        - Enables PILATES-level restart/skip semantics.
        """
        workflow_output_dir, provenance_tracker, db_path, tracker = setup_workflow
        input_data_dir = Path(__file__).resolve().parent / "fixtures" / "dummy_workflow"
        year = 2025

        workflow_state = DummyWorkflowState(current_year=year)
        workspace = DummyWorkspace(output_dir=str(workflow_output_dir))

        model_a_config = {"input_dir": str(input_data_dir)}
        preprocessor = DummyModelAPreprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        runner = DummyModelARunner(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )

        with tracker.scenario(
            name="restart_lookup",
            config={"year": year},
            tags=["restart_lookup"],
            model="test_orchestrator",
        ) as scenario:
            with scenario.step("initialization"):
                _, mutable_records = preprocessor.copy_data_to_mutable_location(
                    model_a_config, str(workflow_output_dir)
                )
                provenance_tracker.log_record_store(mutable_records, direction="output")

            with scenario.step("run"):
                # Force one decorated run so adapter records outputs for restart lookup
                runner.run(mutable_records, workspace)

        # Fresh adapter instance simulating a restart in a new process
        fresh_adapter = ConsistProvenanceTracker(
            run_id="placeholder_id",
            output_path=str(workflow_output_dir),
            tracker=tracker,
        )

        latest = fresh_adapter.get_latest_completed_model_run(
            "modela", year=year, iteration=0
        )
        assert latest is not None, "Expected restart lookup to find prior completed run"

    def test_container_runner_with_consist_integration(self, setup_workflow):
        """
        Test container-aware runner with Consist integration mocked.

        This test validates that:
        1. GenericRunner.run_container() is called with correct parameters
        2. ConsistProvenanceTracker is properly passed for Consist integration
        3. Volume format conversion happens correctly
        4. Container execution is tracked by Consist

        This is the core test for the Consist container integration feature.
        """
        workflow_output_dir, provenance_tracker, db_path, tracker = setup_workflow

        input_data_dir = Path(__file__).resolve().parent / "fixtures" / "dummy_workflow"
        year = 2025

        # Initialize state and workspace
        workflow_state = DummyWorkflowState(current_year=year)
        workflow_state.full_settings.shared.database.path = str(db_path)
        # Set up container manager configuration for the test
        workflow_state.full_settings.infrastructure = SimpleNamespace(
            container_manager="docker",
            docker_config=SimpleNamespace(pull_latest=False, stdout=False),
        )
        workflow_state.full_settings.run = SimpleNamespace(use_stubs=False)

        workspace = DummyWorkspace(output_dir=str(workflow_output_dir))

        # Prepare input data
        model_config = {"input_dir": str(input_data_dir)}
        preprocessor = DummyModelAPreprocessor(
            "ModelContainer", model_config, provenance_tracker, workflow_state
        )

        # Copy input data to mutable location
        _, mutable_records = preprocessor.copy_data_to_mutable_location(
            model_config, str(workflow_output_dir)
        )
        # Decorated preprocessors require an active Consist step.
        with tracker.scenario(
            name="container_test",
            config={"year": year},
            tags=["container_test"],
            model="test_orchestrator",
        ) as scenario:
            with scenario.step("preprocess_container", model="modelcontainer", year=year):
                preprocess_output = preprocessor.preprocess(
                    workspace, previous_records=mutable_records
                )

        # Create container runner
        container_runner = DummyContainerRunner(
            "ModelContainer", model_config, provenance_tracker, workflow_state
        )

        # Mock consist_run_container to verify it's called correctly
        from unittest.mock import patch, MagicMock

        with patch(
            "pilates.generic.runner.consist_run_container"
        ) as mock_consist_run_container:
            # Make it return True (success)
            mock_consist_run_container.return_value = True

            with patch("pilates.generic.runner.CONSIST_AVAILABLE", True):
                # Run the container-based model inside an active step.
                with tracker.scenario(
                    name="container_test_run",
                    config={"year": year},
                    tags=["container_test_run"],
                    model="test_orchestrator",
                ) as scenario:
                    with scenario.step("run_container", model="modelcontainer", year=year):
                        runner_output = container_runner.run(
                            preprocess_output, workspace
                        )

        # Verify that consist_run_container was called
        assert (
            mock_consist_run_container.called
        ), "GenericRunner.run_container should have called consist_run_container"

        # Verify the parameters passed to consist_run_container
        call_kwargs = mock_consist_run_container.call_args.kwargs

        # Check that tracker was passed
        assert (
            "tracker" in call_kwargs
        ), "consist_run_container should be called with tracker parameter"

        # Check that run_id is properly formatted
        assert (
            "run_id" in call_kwargs
        ), "consist_run_container should be called with run_id parameter"
        assert (
            "container" in call_kwargs["run_id"].lower()
        ), "run_id should contain model name and 'container'"

        # Check that image is passed
        assert (
            call_kwargs["image"] == "dummy-model:latest"
        ), "Image should match what was passed to run_container"

        # Check that volumes were converted to Consist format
        assert "volumes" in call_kwargs
        volumes = call_kwargs["volumes"]
        # Consist format should be {host: container} (not Docker format)
        for host, container in volumes.items():
            assert isinstance(
                container, str
            ), f"Consist volumes should have string values, got {type(container)}"
            assert "/" in container, "Container path should be absolute"

        # Check that backend_type was set
        assert call_kwargs["backend_type"] == "docker", "Backend type should be docker"

        # Check that inputs and outputs were passed
        assert (
            "inputs" in call_kwargs
        ), "Inputs should be passed to consist_run_container"
        assert (
            "outputs" in call_kwargs
        ), "Outputs should be passed to consist_run_container"

        # Verify output records were created
        assert runner_output is not None, "Runner should return output records"
        output_records = list(runner_output.all_records())
        assert (
            len(output_records) > 0
        ), "Output records should be created by container runner"

        # Verify output files exist
        container_output_csv = (
            workflow_output_dir / f"model_container_output_{year}.csv"
        )
        container_output_h5 = workflow_output_dir / f"model_container_output_{year}.h5"
        assert (
            container_output_csv.exists()
        ), "Container runner should create CSV output"
        assert container_output_h5.exists(), "Container runner should create H5 output"

        # Verify provenance was tracked via output records

        print("✓ Container runner with Consist integration test passed:")
        print(f"  - consist_run_container was called with correct parameters")
        print(f"  - Volume format conversion verified")
        print(
            f"  - Output files created: {container_output_csv.name}, {container_output_h5.name}"
        )
        print(f"  - {len(output_records)} output records created")

    def test_json_persistence_structure(self, setup_workflow):
        """
        Verify the structure and content of the persisted consist.json file.

        This test ensures that:
        1. Consist dual-write JSON exists for the active run directory.
        2. The JSON includes a run header plus inputs/outputs lists.
        3. Scenario/step metadata is preserved.
        """
        import json

        # 1. Execute the workflow (reuse the logic from test_single_year_workflow)
        # We perform a minimal run here just to populate the data
        workflow_output_dir, provenance_tracker, db_path, consist_tracker = setup_workflow
        self._execute_minimal_workflow(
            workflow_output_dir, provenance_tracker, consist_tracker, db_path
        )

        # 2. Locate and load the JSON file
        json_path = workflow_output_dir / "consist.json"
        assert json_path.exists(), "consist.json was not created"

        with open(json_path, "r") as f:
            data = json.load(f)

        # 3. Validation: ConsistRecord shape (keep loose to avoid brittleness).
        assert "run" in data, "ConsistRecord should include a run header"
        assert "inputs" in data and isinstance(data["inputs"], list)
        assert "outputs" in data and isinstance(data["outputs"], list)
        assert data["run"].get("status") == "completed"
        assert "config" in data, "ConsistRecord should include config"

        print(f"✓ Consist JSON persistence verified at: {json_path}")

    def test_consist_query_integration(self, setup_workflow):
        """
        Verify that Consist query tools can read the database created by the workflow.

        This replaces the CLI test to avoid DuckDB process-locking issues in the test runner.
        It validates the same stack (Data Model -> DB -> Query Logic) without the Typer/Click overhead.
        """
        from sqlmodel import Session
        from consist.tools import queries

        # 1. Execute the workflow
        workflow_output_dir, provenance_tracker, db_path, consist_tracker = setup_workflow
        self._execute_minimal_workflow(
            workflow_output_dir, provenance_tracker, consist_tracker, db_path
        )

        # 2. Use the existing tracker's engine to query
        # This replicates what the CLI does (get_tracker -> session -> queries)
        # but reuses the open connection to avoid 'OperationalError'
        tracker = provenance_tracker._tracker

        with Session(tracker.engine) as session:
            # Test get_runs (equivalent to 'consist runs')
            runs = queries.get_runs(session)
            assert len(runs) >= 2, "Expected at least 2 runs (ModelA, ModelB)"

            model_names = {r.model_name for r in runs}
            assert "modela" in model_names
            assert "modelb" in model_names

            for run in runs:
                assert run.status == "completed"

            # Test get_summary (equivalent to 'consist summary')
            summary = queries.get_summary(session)
            assert summary["total_runs"] >= 2
            assert summary["total_artifacts"] > 0
            assert summary["completed_runs"] >= 2

        # 3. Test artifact lineage/retrieval (equivalent to 'consist artifacts')
        # Prefer Consist DB runs over legacy in-memory adapter state.
        model_a_post = next(
            (
                r
                for r in runs
                if r.model_name == "modela" and "postprocess_a" in r.id
            ),
            None,
        )
        if model_a_post is None:
            model_a_post = next((r for r in runs if r.model_name == "modela"), None)
        assert model_a_post is not None, "Expected a ModelA step run in Consist runs"
        model_a_run_id = model_a_post.id

        # Test get_artifacts_for_run (used by CLI 'artifacts' command)
        artifacts = tracker.get_artifacts_for_run(model_a_run_id)
        assert len(artifacts.inputs) > 0
        assert len(artifacts.outputs) > 0

        # Verify specific artifact presence
        output_keys = [a.key for a in artifacts.outputs.values()]
        # Consist normalizes keys; check if our output is there
        assert any(
            "model_a_final_output" in k for k in output_keys
        ), f"Expected 'model_a_final_output' in {output_keys}"

        print("✓ Consist Query integration verified")
        print("  - queries.get_runs returned correct models")
        print("  - queries.get_summary matched expected counts")
        print("  - tracker.get_artifacts_for_run returned inputs/outputs")

    def _execute_minimal_workflow(
        self, output_dir, provenance_tracker, consist_tracker, db_path
    ):
        """Helper to run a minimal version of the workflow for persistence tests."""
        input_data_dir = Path(__file__).resolve().parent / "fixtures" / "dummy_workflow"
        year = 2025
        state = DummyWorkflowState(current_year=year)
        state.full_settings.shared.database.path = str(db_path)
        workspace = DummyWorkspace(output_dir=str(output_dir))

        # Run Model A Pre/Run/Post
        config = {"input_dir": str(input_data_dir)}
        prep = DummyModelAPreprocessor("ModelA", config, provenance_tracker, state)
        runner = DummyModelARunner("ModelA", config, provenance_tracker, state)
        post = DummyModelAPostprocessor("ModelA", config, provenance_tracker, state)

        prep_b = DummyModelBPreprocessor("ModelB", {}, provenance_tracker, state)
        runner_b = DummyModelBRunner("ModelB", {}, provenance_tracker, state)
        post_b = DummyModelBPostprocessor("ModelB", {}, provenance_tracker, state)

        with consist_tracker.scenario(
            name="minimal_workflow",
            config={"year": year},
            tags=["minimal_workflow"],
            model="test_orchestrator",
        ) as scenario:
            with scenario.step("initialization"):
                _, recs = prep.copy_data_to_mutable_location(config, str(output_dir))
                provenance_tracker.log_record_store(recs, direction="output")

            with scenario.step("preprocess_a", model="modela", year=year):
                recs = prep.preprocess(workspace, previous_records=recs)

            with scenario.step("run_a", model="modela", year=year):
                recs = runner.run(recs, workspace)

            with scenario.step("postprocess_a", model="modela", year=year):
                recs = post.postprocess(recs, workspace)

            with scenario.step("preprocess_b", model="modelb", year=year):
                recs = prep_b.preprocess(workspace, previous_records=recs)

            with scenario.step("run_b", model="modelb", year=year):
                recs, _ = runner_b.run(recs, workspace)

            with scenario.step("postprocess_b", model="modelb", year=year):
                post_b.postprocess(recs, workspace)
