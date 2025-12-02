"""
Dummy Workflow Test for PILATES Framework

This test module serves as both a regression test and a comprehensive example of how to use
the PILATES preprocessor/runner/postprocessor pattern with provenance tracking and database
integration.

Purpose:
--------
1. **Documentation**: Provides a complete, working example of the PILATES workflow pattern
   that new users can study and adapt for their own models.

2. **Regression Testing**: Ensures that the core workflow infrastructure (preprocessor,
   runner, postprocessor, RecordStore, provenance tracking) continues to work correctly
   as the codebase evolves.

3. **Integration Testing**: Validates the end-to-end integration of multiple components:
   - Generic base classes (GenericPreprocessor, GenericRunner, GenericPostprocessor)
   - Record management (FileRecord, H5FileRecord, H5TableRecord, RecordStore)
   - Provenance tracking (OpenLineageTracker)
   - Database integration (DuckDBManager, schema initialization, data upload)
   - Workspace management

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
- Runners must return Tuple[RecordStore, ModelRunInfo]
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

from pilates.generic.model import provenance_logging
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.runner import GenericRunner
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import (
    H5FileRecord,
    H5TableRecord,
    FileRecord,
    RecordStore,
    ModelRunInfo,
)
from pilates.utils.provenance import OpenLineageTracker
from pilates.utils.duckdb_manager import DuckDBManager
from pilates.workspace import Workspace


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

    Returns:
        A 16-character hexadecimal string that uniquely identifies the file path

    Note:
        This is NOT suitable for production use where you want to detect if two files
        have identical content. Use pilates.utils.provenance methods for that.
    """
    import hashlib

    return hashlib.md5(file_path.encode()).hexdigest()[:16]


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
        # In this dummy example, the preprocessor simply passes through the records
        # that were copied to the mutable location.
        # The actual copying is handled by copy_data_to_mutable_location,
        # and its outputs are passed as previous_records to this method.
        return previous_records


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
    6. Returns a RecordStore with output metadata AND a ModelRunInfo

    Key Learning Points:
    - How to find records by short_name using get_record_by_short_name()
    - How to access file paths via record.file_path
    - How runners must return Tuple[RecordStore, ModelRunInfo] (not just RecordStore)
    - Pattern for creating output FileRecords with unique_ids
    """

    def __init__(self, model_name, config, provenance_tracker, state):
        super().__init__(model_name, state, provenance_tracker)
        self.config = config

    @provenance_logging
    def _run(
        self, store: RecordStore, workspace: Workspace
    ) -> Tuple[RecordStore, ModelRunInfo]:
        output_dir = workspace.output_dir
        # Extract paths from the RecordStore
        csv_record = get_record_by_short_name(store, "data.csv")
        h5_record = get_record_by_short_name(store, "data.h5")

        # Ensure records are found before accessing their paths
        if not csv_record:
            raise ValueError(
                "data.csv record not found in RecordStore for Model A Runner."
            )
        if not h5_record:
            raise ValueError(
                "data.h5 record not found in RecordStore for Model A Runner."
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
        )
        output_h5_table_path = output_h5_path + "/table1_modified"
        output_h5_table_record = H5TableRecord(
            file_path=output_h5_table_path,
            h5_file_unique_id=output_h5_file_record.unique_id,
            table_name="table1_modified",
            short_name="table1_modified",
            description="Modified Table 1 from Model A H5 output",
            unique_id=make_unique_id(output_h5_table_path),
        )
        output_h5_file_record.table_record_ids = [output_h5_table_record.unique_id]

        return (
            RecordStore(
                recordList=[
                    FileRecord(
                        file_path=output_csv_path,
                        short_name=f"model_a_output_{year}.csv",
                        unique_id=make_unique_id(output_csv_path),
                    ),
                    output_h5_file_record,
                    output_h5_table_record,
                ]
            ),
            ModelRunInfo(model=self.model_name, year=year),
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
        runInfo=None,
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
        )
        final_h5_table_record = H5TableRecord(
            file_path=final_h5_path + "/table1_final",
            h5_file_unique_id=final_h5_file_record.unique_id,
            table_name="table1_final",
            short_name="table1_final",
            description="Final Table from Model A H5 output",
        )
        final_h5_file_record.table_record_ids = [final_h5_table_record.unique_id]

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=final_csv_path,
                    short_name=f"model_a_final_output_{year}.csv",
                    unique_id=make_unique_id(final_csv_path),
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
    def _run(
        self, store: RecordStore, workspace: Workspace
    ) -> Tuple[RecordStore, ModelRunInfo]:
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
        )
        output_h5_table_record = H5TableRecord(
            file_path=output_h5_path + "/table_b_modified",
            h5_file_unique_id=output_h5_file_record.unique_id,
            table_name="table_b_modified",
            short_name="table_b_modified",
            description="Modified Table from Model B H5 output",
        )
        output_h5_file_record.table_record_ids = [output_h5_table_record.unique_id]

        return (
            RecordStore(
                recordList=[
                    FileRecord(
                        file_path=output_csv_path,
                        short_name=f"model_b_output_{year}.csv",
                        unique_id=make_unique_id(output_csv_path),
                    ),
                    output_h5_file_record,
                    output_h5_table_record,
                ]
            ),
            ModelRunInfo(model=self.model_name, year=year),
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
        runInfo=None,
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
                ),
                FileRecord(
                    file_path=final_h5_summary_path,
                    short_name=f"model_b_final_output_summary_{year}.txt",
                    unique_id=make_unique_id(final_h5_summary_path),
                ),
            ]
        )


# ============================================================================
# TEST CLASS
# ============================================================================


class TestDummyWorkflow:
    """
    Integration test for the PILATES preprocessor/runner/postprocessor workflow.

    This test validates the complete workflow infrastructure and serves as
    documentation for how to implement PILATES models. It tests:

    1. Data copying and workspace setup
    2. Record creation and management
    3. Provenance tracking throughout the workflow
    4. Model chaining (passing outputs from Model A to Model B)
    5. File I/O and data transformations
    6. Database integration (commented out for simplicity)

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
            provenance_tracker = OpenLineageTracker(
                run_id="test_run", output_path=str(workflow_output_dir)
            )

            # Instantiate DuckDBManager and initialize the database
            duckdb_manager = DuckDBManager(database_path=str(db_path))
            duckdb_manager.initialize_database()

            yield workflow_output_dir, provenance_tracker, db_path, duckdb_manager

    def test_single_year_workflow(self, setup_workflow):
        """
        Test a complete two-model workflow for a single simulation year.

        This test executes a full PILATES workflow demonstrating:

        1. **Model A Pipeline**:
           - Preprocessor: Copies CSV and H5 files to workspace, creates FileRecords
           - Runner: Doubles CSV column, adds 1 to H5 values
           - Postprocessor: Filters CSV (a_doubled > 10), multiplies H5 by 2

        2. **Model B Pipeline** (consumes Model A outputs):
           - Preprocessor: Validates Model A outputs are available
           - Runner: Adds new CSV column, subtracts 1 from H5 values
           - Postprocessor: Generates summary statistics (row count, sum)

        3. **Validation**:
           - Verifies all expected output files exist
           - Validates data transformations were applied correctly
           - Confirms provenance tracking captured all operations

        4. **Database Integration**:
           - Uploads complete run data to DuckDB
           - Validates model runs and file records were stored
           - Demonstrates how to persist results for later querying and analysis

        The test uses temporary directories and databases that are automatically
        cleaned up, making it safe to run repeatedly without side effects.

        Args:
            setup_workflow: Pytest fixture providing workspace, provenance tracker, and database
        """
        workflow_output_dir, provenance_tracker, db_path, duckdb_manager = (
            setup_workflow
        )

        input_data_dir = Path(
            "/Users/zaneedell/git/PILATES/tests/fixtures/dummy_workflow"
        )
        year = 2025
        scenario_name = "test_scenario"

        # Initialize DummyWorkflowState and DummyWorkspace
        workflow_state = DummyWorkflowState(current_year=year)
        # Set the database path for the dummy state to allow provenance logging
        workflow_state.full_settings.shared.database.path = str(db_path)
        workspace = DummyWorkspace(output_dir=str(workflow_output_dir))

        # --- Run Model A ---
        model_a_config = {"input_dir": str(input_data_dir)}  # Pass input_dir in config
        model_a_preprocessor = DummyModelAPreprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        model_a_runner = DummyModelARunner(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )
        model_a_postprocessor = DummyModelAPostprocessor(
            "ModelA", model_a_config, provenance_tracker, workflow_state
        )

        # Call copy_data_to_mutable_location first
        _, mutable_location_records_a = (
            model_a_preprocessor.copy_data_to_mutable_location(
                model_a_config, str(workflow_output_dir)
            )
        )

        # Call public methods, passing the records from mutable_location
        preprocessor_a_output_records = model_a_preprocessor.preprocess(
            workspace, previous_records=mutable_location_records_a
        )
        runner_a_output_records, _ = model_a_runner.run(
            preprocessor_a_output_records, workspace
        )
        postprocessor_a_output_records = model_a_postprocessor.postprocess(
            runner_a_output_records, workspace
        )

        # Assert Model A outputs exist and validate content
        model_a_csv_path = workflow_output_dir / f"model_a_final_output_{year}.csv"
        model_a_h5_path = workflow_output_dir / f"model_a_final_output_{year}.h5"
        assert model_a_csv_path.exists()
        assert model_a_h5_path.exists()

        # Validate Model A CSV content: should be filtered (a_doubled > 10) and have a_doubled column
        df_a_final = pd.read_csv(model_a_csv_path)
        assert (
            "a_doubled" in df_a_final.columns
        ), "Model A output should have 'a_doubled' column"
        assert (
            df_a_final["a_doubled"] > 10
        ).all(), "Model A output should only contain rows where a_doubled > 10"

        # Validate Model A H5 content: should have table1_final
        with h5py.File(model_a_h5_path, "r") as f:
            assert "table1_final" in f, "Model A H5 should contain 'table1_final' table"

        # --- Run Model B ---
        model_b_config = {}  # Dummy config
        model_b_preprocessor = DummyModelBPreprocessor(
            "ModelB", model_b_config, provenance_tracker, workflow_state
        )
        model_b_runner = DummyModelBRunner(
            "ModelB", model_b_config, provenance_tracker, workflow_state
        )
        model_b_postprocessor = DummyModelBPostprocessor(
            "ModelB", model_b_config, provenance_tracker, workflow_state
        )

        # Call copy_data_to_mutable_location for Model B (returns empty RecordStores)
        _, mutable_location_records_b = (
            model_b_preprocessor.copy_data_to_mutable_location(
                model_b_config, str(workflow_output_dir)
            )
        )

        # Call public methods, passing output from Model A's postprocessor as input
        preprocessor_b_output_records = model_b_preprocessor.preprocess(
            workspace,
            previous_records=postprocessor_a_output_records
            + mutable_location_records_b,
        )
        runner_b_output_records, _ = model_b_runner.run(
            preprocessor_b_output_records, workspace
        )
        postprocessor_b_output_records = model_b_postprocessor.postprocess(
            runner_b_output_records, workspace
        )

        # Assert Model B outputs exist and validate content
        model_b_txt_path = workflow_output_dir / f"model_b_final_output_{year}.txt"
        model_b_summary_path = (
            workflow_output_dir / f"model_b_final_output_summary_{year}.txt"
        )
        assert model_b_txt_path.exists()
        assert model_b_summary_path.exists()

        # Validate Model B text outputs contain expected content
        with open(model_b_txt_path, "r") as f:
            row_count_content = f.read()
            assert (
                "Row count from Model B CSV:" in row_count_content
            ), "Model B output should contain row count"

        with open(model_b_summary_path, "r") as f:
            sum_content = f.read()
            assert (
                "Total sum from Model B H5:" in sum_content
            ), "Model B summary should contain H5 sum"

        # Save the provenance data (the tracker automatically saves run_info.json)
        # The run_info is saved incrementally as operations complete

        # ========================================================================
        # DATABASE INTEGRATION: Upload results to DuckDB
        # ========================================================================

        # Upload the complete run data to the database
        upload_success = duckdb_manager.upload_run_data(provenance_tracker.run_info)
        assert upload_success, "Failed to upload run data to DuckDB"

        # Validate database content
        conn = duckdb_manager._get_connection()

        # Check that we have file records (the core provenance data)
        file_record_count = conn.execute(
            "SELECT COUNT(*) FROM file_records"
        ).fetchone()[0]
        assert (
            file_record_count > 0
        ), "No file records found in database (expected records from CSV and H5 files)"

        # Check that we have runs table entry
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert run_count > 0, "No runs found in database"

        # Verify file records include both models' outputs
        # Query for file records to see what was uploaded
        records = conn.execute(
            "SELECT short_name, description FROM file_records ORDER BY short_name"
        ).fetchall()
        short_names = [r[0] for r in records]

        # We should have at least some of our output files
        print(f"✓ Database contains {file_record_count} file records")
        print(
            f"  Sample records: {short_names[:5] if len(short_names) > 5 else short_names}"
        )

        conn.close()

        print(f"✓ Workflow output in: {workflow_output_dir}")
        print(f"✓ Provenance DB in: {db_path}")
        print(
            f"✓ Database integration complete: {file_record_count} file records uploaded"
        )
