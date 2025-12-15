"""
test_run_refactor.py

Integration tests specifically for the run.py refactor using Consist.
Verifies:
1. Tracker initialization with dual mounts.
2. Initialization adapter behavior.
3. Scenario Context grouping.
4. Adapter "Wrapper Mode" constraints and object reconstruction.
"""

import os
import pytest
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Optional dependency stubs (for lightweight test environments)
# ---------------------------------------------------------------------------
import sys
import types


def _stub_module(module_name: str, attrs=None):
    mod = sys.modules.get(module_name)
    if mod is None:
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod
    if attrs:
        for key, value in attrs.items():
            setattr(mod, key, value)
    return mod


# OpenLineage is an optional dependency in some lightweight environments.
# Stub it for tests that exercise Consist integration but not OpenLineage payloads.
try:
    import openlineage  # noqa: F401
except ModuleNotFoundError:
    openlineage_mod = _stub_module("openlineage")
    openlineage_client_mod = _stub_module(
        "openlineage.client",
        attrs={"set_producer": lambda *a, **k: None, "OpenLineageClient": object},
    )
    setattr(openlineage_mod, "client", openlineage_client_mod)
    _stub_module(
        "openlineage.client.facet",
        attrs={
            "SchemaField": object,
            "SchemaDatasetFacet": object,
            "DocumentationJobFacet": object,
            "SourceCodeLocationJobFacet": object,
        },
    )
    _stub_module(
        "openlineage.client.run",
        attrs={
            "Dataset": object,
            "InputDataset": object,
            "OutputDataset": object,
            "RunEvent": object,
            "Run": object,
            "Job": object,
            "RunState": object,
        },
    )
    _stub_module(
        "openlineage.client.transport.http",
        attrs={"HttpTransport": object, "HttpConfig": object},
    )
    _stub_module(
        "openlineage.client.transport.file",
        attrs={"FileTransport": object, "FileConfig": object},
    )
    _stub_module(
        "openlineage.client.transport.composite",
        attrs={"CompositeTransport": object, "CompositeConfig": object},
    )

# ActivitySim preprocessor imports heavy optional deps at module import time.
try:
    import openmatrix  # noqa: F401
except ModuleNotFoundError:
    _stub_module("openmatrix", attrs={"File": object, "open_file": lambda *a, **k: None})

try:
    import geopandas  # noqa: F401
except ModuleNotFoundError:
    _stub_module("geopandas", attrs={"GeoDataFrame": object, "GeoSeries": object})

try:
    import shapely  # noqa: F401
except ModuleNotFoundError:
    shapely_mod = _stub_module("shapely")
    shapely_wkt_mod = _stub_module("shapely.wkt")
    shapely_geometry_mod = _stub_module("shapely.geometry", attrs={"Polygon": object})
    setattr(shapely_mod, "wkt", shapely_wkt_mod)
    setattr(shapely_mod, "geometry", shapely_geometry_mod)

try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    _stub_module("tqdm", attrs={"tqdm": lambda x, *a, **k: x})

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    matplotlib_mod = _stub_module("matplotlib")
    matplotlib_pyplot_mod = _stub_module("matplotlib.pyplot")
    setattr(matplotlib_mod, "pyplot", matplotlib_pyplot_mod)

# Consist Imports
from consist import Tracker

# PILATES Imports
from pilates.utils.consist_adapter import ConsistProvenanceTracker
from pilates.generic.initialization import Initialization
from pilates.workspace import Workspace
from pilates.config.models import PilatesConfig
from pilates.generic.records import FileRecord

# Mock helpers to simulate pilates settings structure without full config parsing
def create_mock_settings(input_dir, output_dir):
    """Creates a mock PilatesConfig object with necessary path attributes."""
    settings = MagicMock(spec=PilatesConfig)
    settings.run = MagicMock()
    settings.run.output_directory = str(output_dir)
    settings.run.output_run_name = "test_run"
    settings.run.region = "test_region"
    settings.run.start_year = 2010
    settings.run.models = MagicMock()
    settings.run.models.travel = "beam"
    settings.run.models.activity_demand = None
    settings.run.models.vehicle_ownership = None
    settings.run.models.land_use = "urbansim"
    settings.urbansim = MagicMock()
    settings.urbansim.local_data_input_folder = str(input_dir)
    settings.urbansim.local_mutable_data_folder = "urbansim/data"
    settings.shared = MagicMock()
    settings.shared.database = MagicMock()
    settings.shared.database.enabled = False
    settings.beam = MagicMock()
    settings.beam.local_mutable_data_folder = "beam/data"
    return settings

class TestRunRefactor:

    @pytest.fixture
    def setup_env(self):
        """Sets up source (immutable) and run (mutable) directories."""
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            input_dir = root_path / "project_root" / "data"
            input_dir.mkdir(parents=True)
            (input_dir / "households.csv").write_text("id,persons\n1,2")
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

        tracker = Tracker(
            run_dir=full_run_dir,
            mounts={
                "inputs": str(input_dir),       # Immutable source
                "workspace": str(full_run_dir)  # Mutable scratchpad
            },
            project_root=str(full_run_dir)
        )

        # Test 1: Input Resolution
        source_file = input_dir / "households.csv"
        uri = tracker._virtualize_path(str(source_file))
        assert uri == "inputs://households.csv"

        # Test 2: Workspace Resolution
        dest_file = full_run_dir / "urbansim" / "data" / "households.csv"
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
        db_path = full_run_dir / "provenance.db"

        tracker = Tracker(
            run_dir=full_run_dir,
            db_path=str(db_path),
            mounts={ "inputs": str(input_dir), "workspace": str(full_run_dir) },
            project_root=str(full_run_dir)
        )

        settings = create_mock_settings(input_dir, output_dir)
        state = MagicMock()
        state.start_year = 2010
        state.current_year = 2010
        state.full_settings = settings

        workspace = MagicMock(spec=Workspace)
        workspace.get_usim_mutable_data_dir.return_value = str(full_run_dir / "urbansim/data")
        workspace.input_data = {}
        workspace.output_data = {}
        # Mock other paths to avoid mkdir errors
        workspace.get_beam_mutable_data_dir.return_value = str(full_run_dir / "beam/data")
        workspace.get_atlas_mutable_input_dir.return_value = str(full_run_dir / "atlas/in")
        workspace.get_atlas_output_dir.return_value = str(full_run_dir / "atlas/out")
        workspace.get_asim_mutable_data_dir.return_value = str(full_run_dir / "asim/data")
        workspace.get_asim_output_dir.return_value = str(full_run_dir / "asim/out")

        with tracker.scenario("test_scenario") as scenario:
            with scenario.step("initialization"):
                adapter = ConsistProvenanceTracker(
                    run_id=tracker.current_consist.run.id,
                    output_path=str(full_run_dir),
                    folder_name="test_run_adapter",
                    tracker=tracker
                )

                with patch("pilates.generic.initialization.ModelFactory") as MockFactory:
                    mock_preprocessor = MagicMock()

                    def mock_copy(settings, dest_dir):
                        src = input_dir / "households.csv"
                        dst = Path(dest_dir) / "households.csv"
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy(src, dst)

                        rec_in = adapter.record_input_file("urbansim", str(src))
                        rec_out = adapter.record_output_file("urbansim", str(dst))

                        from pilates.generic.records import RecordStore
                        return RecordStore(recordList=[rec_in]), RecordStore(recordList=[rec_out])

                    mock_preprocessor.copy_data_to_mutable_location.side_effect = mock_copy
                    MockFactory.return_value.get_preprocessor.return_value = mock_preprocessor

                    init_model = Initialization("initialization", state, adapter)
                    init_model.run(settings, workspace)

        # Assertion: Check DB Lineage
        init_run = tracker.find_run(id="test_scenario_initialization")
        assert init_run is not None
        artifacts = tracker.get_artifacts_for_run(init_run.id)

        input_uris = [a.uri for a in artifacts.inputs.values()]
        assert "inputs://households.csv" in input_uris

        output_uris = [a.uri for a in artifacts.outputs.values()]
        assert any("workspace://urbansim/data/households.csv" in uri for uri in output_uris)

    # --- NEW TESTS BELOW ---

    def test_adapter_resolves_inputs_relative_paths_independent_of_cwd(self, setup_env):
        """
        Ensure ConsistProvenanceTracker resolves relative input paths against the configured
        `tracker.project_root` (inputs mount), not only against cwd/workspace.
        """
        input_dir, output_dir = setup_env
        project_root = input_dir.parent  # .../project_root

        full_run_dir = Path(output_dir) / "test_run_relpaths"
        full_run_dir.mkdir()

        tracker = Tracker(
            run_dir=full_run_dir,
            mounts={"inputs": str(project_root), "workspace": str(full_run_dir)},
            project_root=str(project_root),
        )

        adapter = ConsistProvenanceTracker(
            run_id="placeholder_id",
            output_path=str(full_run_dir),
            folder_name="test_run_relpaths",
            tracker=tracker,
        )

        old_cwd = os.getcwd()
        os.chdir(str(output_dir))  # Simulate production: cwd != project root
        try:
            with tracker.scenario("test_scenario_relpaths") as scenario:
                with scenario.step("relpath_step"):
                    rec = adapter.record_input_file(
                        "test_model",
                        "data/households.csv",
                        short_name="households",
                        skip_missing=False,
                    )
                    assert rec is not None
                    assert rec.uri == "inputs://data/households.csv"
        finally:
            os.chdir(old_cwd)

    def test_adapter_context_enforcement(self, setup_env):
        """
        Verify that the new Adapter raises RuntimeError if start_model_run
        is called without an active 'scenario.step' context.
        """
        input_dir, output_dir = setup_env
        full_run_dir = output_dir / "test_enforcement"
        full_run_dir.mkdir()

        tracker = Tracker(run_dir=full_run_dir)
        adapter = ConsistProvenanceTracker(
            run_id="placeholder",
            output_path=str(full_run_dir),
            tracker=tracker
        )

        # Should fail because tracker.current_consist is None
        with pytest.raises(RuntimeError) as excinfo:
            adapter.start_model_run("test_model")

        assert "No active Consist run found" in str(excinfo.value)

    def test_filerecord_fidelity_and_move(self, setup_env):
        """
        Verify that:
        1. record_output_file correctly reconstructs a PILATES FileRecord from an Artifact.
        2. move_file correctly captures lineage (source_file_paths) in metadata.
        """
        input_dir, output_dir = setup_env
        full_run_dir = output_dir / "test_fidelity"
        full_run_dir.mkdir()
        db_path = full_run_dir / "fidelity.db"

        tracker = Tracker(
            run_dir=full_run_dir,
            db_path=str(db_path),
            mounts={"workspace": str(full_run_dir)}
        )

        adapter = ConsistProvenanceTracker(
            run_id="placeholder",
            output_path=str(full_run_dir),
            tracker=tracker
        )

        # Create dummy file to move
        file_a = full_run_dir / "original.csv"
        file_a.write_text("data")
        file_b = full_run_dir / "moved.csv"

        with tracker.scenario("test_scen") as sc:
            with sc.step("test_step"):
                # 1. Test basic record fidelity
                rec_a = adapter.record_output_file(
                    model="test_model",
                    file_path=str(file_a),
                    description="Original File"
                )

                assert isinstance(rec_a, FileRecord)
                # Check path is relative to workspace (legacy behavior check)
                assert rec_a.file_path == "original.csv"
                # Check unique_id is a UUID (Consist uses UUIDs)
                try:
                    uuid.UUID(rec_a.unique_id)
                except ValueError:
                    pytest.fail("FileRecord unique_id is not a valid UUID")

                # 2. Test move_file logic
                rec_b = adapter.move_file(
                    record=rec_a,
                    source_path=str(file_a),
                    destination_path=str(file_b),
                    model="test_model"
                )

                # Check physical move happened
                assert not file_a.exists()
                assert file_b.exists()

                # Check lineage metadata in retrieved record
                # Note: FileRecord.source_file_paths maps to Artifact.meta["source_file_paths"]
                assert str(file_a) in rec_b.source_file_paths

                # Check consistency in DB
                artifact_b = tracker.get_artifact(rec_b.short_name)
                assert artifact_b is not None
                assert artifact_b.meta["source_file_paths"] == [str(file_a)]

    def test_move_file_skips_noisy_activitysim_final_inputs(self, setup_env):
        """
        ActivitySim parquet pipeline outputs are commonly named `final.parquet`.
        If we log these raw sources as inputs without a stable logical key, Consist
        collapses many distinct artifacts under the key "final", creating noisy
        and misleading step inputs.

        For records ending in `_asim_out_temp`, the adapter should skip logging the
        source as an input and only log the moved output (with `source_file_paths`
        for lineage).
        """
        _, output_dir = setup_env
        full_run_dir = output_dir / "test_asim_final_noise"
        full_run_dir.mkdir()
        db_path = full_run_dir / "fidelity.db"

        tracker = Tracker(
            run_dir=full_run_dir,
            db_path=str(db_path),
            mounts={"workspace": str(full_run_dir)},
        )
        adapter = ConsistProvenanceTracker(
            run_id="placeholder",
            output_path=str(full_run_dir),
            tracker=tracker,
        )

        source_file = full_run_dir / "final.parquet"
        source_file.write_text("data")
        dest_file = full_run_dir / "tours.parquet"

        with tracker.scenario("test_scen") as sc:
            with sc.step("test_step"):
                rec_src = FileRecord(
                    file_path=str(source_file),
                    models=["activitysim"],
                    short_name="tours_asim_out_temp",
                    description="Raw ActivitySim output",
                )

                moved = adapter.move_file(
                    record=rec_src,
                    source_path=str(source_file),
                    destination_path=str(dest_file),
                    model="activitysim_postprocessor",
                )

                run_id = tracker.current_consist.run.id
                artifacts = tracker.get_artifacts_for_run(run_id)

                assert "final" not in (artifacts.inputs or {})
                assert "tours" in (artifacts.outputs or {})
                assert str(source_file) in (moved.source_file_paths or [])
                assert getattr(moved, "uri", None), "Expected moved FileRecord to include a Consist uri"

    def test_legacy_run_info_mirroring(self, setup_env):
        """
        Verify that the adapter keeps its internal `run_info` object updated
        mirrored with Consist actions. Downstream tools rely on this structure.
        """
        _, output_dir = setup_env
        full_run_dir = output_dir / "test_mirror"
        full_run_dir.mkdir()

        tracker = Tracker(run_dir=full_run_dir)
        adapter = ConsistProvenanceTracker(
            run_id="legacy_id",
            output_path=str(full_run_dir),
            tracker=tracker
        )

        fpath = full_run_dir / "data.csv"
        fpath.write_text("content")

        with tracker.scenario("mirror_scen") as sc:
            with sc.step("step1"):
                # Start run via adapter wrapper
                adapter.start_model_run("my_model", year=2020)

                # Log file
                rec = adapter.record_output_file("my_model", str(fpath))

                # Check ModelRunInfo
                current_run_id = tracker.current_consist.run.id
                assert current_run_id in adapter.run_info.model_runs
                model_run = adapter.run_info.model_runs[current_run_id]
                assert model_run.year == 2020
                assert model_run.model == "my_model"

                # Check FileRecord mirroring
                assert rec.unique_id in adapter.run_info.file_records
                assert adapter.run_info.file_records[rec.unique_id].file_path == "data.csv"
