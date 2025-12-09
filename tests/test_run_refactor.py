"""
test_run_refactor.py

Integration tests specifically for the run.py refactor using Consist.
Verifies:
1. Tracker initialization with dual mounts (inputs/workspace).
2. Initialization adapter behavior (bridging legacy copy logic to Consist).
3. Scenario Context grouping.
"""

import os
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Consist Imports
from consist import Tracker
from consist.models.run import Run

# PILATES Imports
from pilates.utils.consist_adapter import ConsistProvenanceTracker
from pilates.generic.initialization import Initialization
from pilates.workspace import Workspace
from pilates.config.models import PilatesConfig
# We mock WorkflowState entirely to avoid dependency hell
# from workflow_state import WorkflowState

# Mock helpers to simulate pilates settings structure without full config parsing
def create_mock_settings(input_dir, output_dir):
    """Creates a mock PilatesConfig object with necessary path attributes."""
    settings = MagicMock(spec=PilatesConfig)

    # Run Config
    settings.run = MagicMock()
    settings.run.output_directory = str(output_dir)
    settings.run.output_run_name = "test_run"
    settings.run.region = "test_region"
    settings.run.start_year = 2010

    # Model Configs
    settings.run.models = MagicMock()
    settings.run.models.travel = "beam"
    settings.run.models.activity_demand = None
    settings.run.models.vehicle_ownership = None
    settings.run.models.land_use = "urbansim"

    # Urbansim Config (Source of immutable inputs in our logic)
    settings.urbansim = MagicMock()
    settings.urbansim.local_data_input_folder = str(input_dir)
    settings.urbansim.local_mutable_data_folder = "urbansim/data"

    # Shared/Database
    settings.shared = MagicMock()
    settings.shared.database = MagicMock()
    settings.shared.database.enabled = False

    # Beam Config
    settings.beam = MagicMock()
    settings.beam.local_mutable_data_folder = "beam/data"

    return settings

class TestRunRefactor:

    @pytest.fixture
    def setup_env(self):
        """Sets up source (immutable) and run (mutable) directories."""
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)

            # 1. Immutable Inputs (simulating the repo/project root)
            input_dir = root_path / "project_root" / "data"
            input_dir.mkdir(parents=True)

            # Create a dummy source file
            (input_dir / "households.csv").write_text("id,persons\n1,2")

            # 2. Mutable Outputs
            output_dir = root_path / "runs"
            output_dir.mkdir(parents=True)

            yield input_dir, output_dir

    def test_tracker_mount_resolution(self, setup_env):
        """
        Verify that the Tracker correctly resolves paths based on the
        'Longest Prefix Match' assumption in run.py.
        """
        input_dir, output_dir = setup_env
        full_run_dir = output_dir / "test_run_1"
        full_run_dir.mkdir()

        # Initialize Tracker exactly as run.py does
        # FIX: Convert Paths to strings
        tracker = Tracker(
            run_dir=full_run_dir,
            mounts={
                "inputs": str(input_dir),       # Immutable source
                "workspace": str(full_run_dir)  # Mutable scratchpad
            },
            project_root=str(full_run_dir)
        )

        # Test 1: Input Resolution
        # Files inside input_dir should virtualize to inputs://
        source_file = input_dir / "households.csv"
        uri = tracker._virtualize_path(str(source_file))
        assert uri == "inputs://households.csv"

        # Test 2: Workspace Resolution
        # Files inside run_dir should virtualize to workspace://
        dest_file = full_run_dir / "urbansim" / "data" / "households.csv"
        # Create dummy file to ensure it exists for resolution if logic checks existence
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text("id,persons\n1,2")

        uri = tracker._virtualize_path(str(dest_file))
        assert uri == "workspace://urbansim/data/households.csv"

    def test_initialization_adapter_lineage(self, setup_env):
        """
        Verify the critical Initialization step:
        The Adapter + Tracker should log a lineage from inputs:// to workspace://
        """
        input_dir, output_dir = setup_env
        full_run_dir = output_dir / "test_run_adapter"
        full_run_dir.mkdir()
        db_path = full_run_dir / "provenance.db"  # Local DB for this run

        # Setup Tracker
        tracker = Tracker(
            run_dir=full_run_dir,
            db_path=str(db_path),  # Persistence required
            mounts={
                "inputs": str(input_dir),
                "workspace": str(full_run_dir)
            },
            project_root=str(full_run_dir)
        )

        # Mock Settings & State
        settings = create_mock_settings(input_dir, output_dir)

        state = MagicMock()
        state.start_year = 2010
        state.current_year = 2010
        state.full_settings = settings

        # Mock Workspace (Legacy object)
        workspace = MagicMock(spec=Workspace)
        workspace.get_beam_mutable_data_dir.return_value = str(full_run_dir / "beam/data")
        workspace.get_usim_mutable_data_dir.return_value = str(full_run_dir / "urbansim/data")
        workspace.input_data = {}
        workspace.output_data = {}

        # Mock paths for other models to prevent mkdir errors
        workspace.get_atlas_mutable_input_dir.return_value = str(full_run_dir / "atlas/in")
        workspace.get_atlas_output_dir.return_value = str(full_run_dir / "atlas/out")
        workspace.get_asim_mutable_data_dir.return_value = str(full_run_dir / "asim/data")
        workspace.get_asim_output_dir.return_value = str(full_run_dir / "asim/out")

        # START SIMULATION
        with tracker.scenario("test_scenario") as scenario:
            with scenario.step("initialization"):
                # 1. Initialize Adapter
                # FIX: Pass 'tracker=tracker' so it uses the configured mounts and DB
                adapter = ConsistProvenanceTracker(
                    run_id=tracker.current_consist.run.id,
                    output_path=str(full_run_dir),
                    folder_name="test_run_adapter",
                    tracker=tracker
                )

                # 2. Mock ModelFactory to return a simple copier instead of real Preprocessors
                with patch("pilates.generic.initialization.ModelFactory") as MockFactory:
                    mock_preprocessor = MagicMock()

                    def mock_copy(settings, dest_dir):
                        # Simulate what UrbansimPreprocessor does:
                        # 1. Copy file physically
                        src = input_dir / "households.csv"
                        dst = Path(dest_dir) / "households.csv"
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy(src, dst)

                        # 2. Record provenance via Adapter
                        rec_in = adapter.record_input_file("urbansim", str(src))
                        rec_out = adapter.record_output_file("urbansim", str(dst))

                        # Return dummy RecordStores
                        from pilates.generic.records import RecordStore
                        return RecordStore(recordList=[rec_in]), RecordStore(recordList=[rec_out])

                    mock_preprocessor.copy_data_to_mutable_location.side_effect = mock_copy
                    MockFactory.return_value.get_preprocessor.return_value = mock_preprocessor

                    # 3. Run Initialization
                    init_model = Initialization("initialization", state, adapter)
                    init_model.run(settings, workspace)

        # ASSERTIONS

        # 1. Verify artifacts were logged in the DB/Tracker
        init_run = tracker.find_run(id=f"test_scenario_initialization")
        assert init_run is not None, "Initialization run not found in DB"

        # Get artifacts using Consist API
        artifacts = tracker.get_artifacts_for_run(init_run.id)

        # Check Inputs
        input_uris = [a.uri for a in artifacts.inputs.values()]
        # This will now pass because 'inputs://' mount is active in the adapter's tracker
        assert "inputs://households.csv" in input_uris

        # Check Outputs
        output_uris = [a.uri for a in artifacts.outputs.values()]
        expected_output = "workspace://urbansim/data/households.csv"
        # Verify if any output URI ends with the expected path components
        # (Exact string match depends on how tracker handles trailing slashes in mounts, checking suffix is safe)
        assert any(uri == expected_output for uri in output_uris), f"Expected {expected_output} in {output_uris}"

        print("\nLineage Verified:")
        print(f"  Source: {input_uris[0]}")
        print(f"  Dest:   {output_uris[0]}")

    def test_scenario_hierarchy(self, setup_env):
        """
        Verify that scenario.step() creates the correct parent/child relationship
        in the database.
        """
        input_dir, output_dir = setup_env
        full_run_dir = output_dir / "test_run_hierarchy"
        full_run_dir.mkdir()
        db_path = full_run_dir / "hierarchy.db"

        # FIX: Need DB path for find_run to work
        tracker = Tracker(
            run_dir=full_run_dir,
            project_root=full_run_dir,
            db_path=str(db_path)
        )

        with tracker.scenario("master_scenario") as scenario:
            with scenario.step("step_1"):
                pass
            with scenario.step("step_2"):
                pass

        # Verify Hierarchy via DB/Tracker
        master = tracker.find_run(id="master_scenario")
        step1 = tracker.find_run(id="master_scenario_step_1")
        step2 = tracker.find_run(id="master_scenario_step_2")

        assert step1.parent_run_id == master.id
        assert step2.parent_run_id == master.id
        # assert master.status == "completed"
        # assert step1.status == "completed"