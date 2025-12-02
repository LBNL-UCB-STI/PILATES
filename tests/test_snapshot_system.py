import pytest
import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import xarray as xr
import pandas as pd

from pilates.utils.duckdb_manager import DuckDBManager
from pilates.utils.snapshot_manager import SnapshotManager
from pilates.utils.snapshot_analysis import SnapshotAnalysisManager
from pilates.utils.provenance import FileProvenanceTracker
from pilates.utils.data_artifact_utils import (
    compute_zarr_chunk_manifest,
    get_zarr_metadata,
)


# --- Fixtures for Testing ---


@pytest.fixture(scope="function")
def temp_db_path(tmp_path):
    """Provides a temporary path for a DuckDB database."""
    return tmp_path / "test.duckdb"


@pytest.fixture(scope="function")
def temp_archive_root(tmp_path):
    """Provides a temporary root directory for archiving snapshots."""
    path = tmp_path / "archive"
    path.mkdir()
    return path


@pytest.fixture(scope="function")
def duckdb_manager(temp_db_path):
    """Provides an initialized DuckDBManager instance."""
    with DuckDBManager(str(temp_db_path)) as db_manager:
        # Manually create the 'snapshots' table for testing purposes
        # In a real scenario, initialize_database() would be called, which reads SQL files.
        # For isolated unit testing, we define the minimal schema needed.
        schema_sql = """
        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            year INTEGER,
            iteration INTEGER,
            sub_iteration INTEGER,
            snapshot_type VARCHAR,
            model VARCHAR,
            parent_snapshot_id VARCHAR,
            created_at TIMESTAMP,
            format VARCHAR,
            artifact_path VARCHAR,
            n_variables INTEGER,
            n_chunks INTEGER,
            total_size_mb FLOAT,
            partial_skims_path VARCHAR,
            partial_skims_n_variables INTEGER,
            partial_skims_n_chunks INTEGER,
            partial_skims_total_size_mb FLOAT,
            changed_chunks INTEGER,
            chunk_manifest JSON
        );
        """
        db_manager.execute_sql(schema_sql)
        yield db_manager


@pytest.fixture(scope="function")
def snapshot_manager(duckdb_manager, temp_archive_root):
    """Provides a SnapshotManager instance."""
    return SnapshotManager(duckdb_manager, temp_archive_root)


@pytest.fixture(scope="function")
def snapshot_analysis_manager(duckdb_manager, temp_archive_root):
    """Provides a SnapshotAnalysisManager instance."""
    return SnapshotAnalysisManager(duckdb_manager, temp_archive_root)


@pytest.fixture(scope="function")
def dummy_zarr_store(tmp_path):
    """Creates a dummy Zarr store for testing."""
    zarr_path = tmp_path / "dummy_zarr.zarr"
    # Create a simple xarray Dataset and save it as Zarr
    ds = xr.Dataset(
        {
            "SOV_TIME": (
                ("otaz", "dtaz", "time_period"),
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            ),
            "SOV_DIST": (
                ("otaz", "dtaz", "time_period"),
                [[[10, 11], [12, 13]], [[14, 15], [16, 17]]],
            ),
        },
        coords={"otaz": [0, 1], "dtaz": [0, 1], "time_period": ["AM", "PM"]},
    )
    ds.attrs["original_zone_ids"] = [100, 200]
    ds.to_zarr(zarr_path, consolidated=True)
    return zarr_path


@pytest.fixture(scope="function")
def dummy_netcdf_file(tmp_path):
    """Creates a dummy NetCDF file for testing."""
    netcdf_path = tmp_path / "dummy.nc"
    # Create a simple xarray Dataset and save it as NetCDF
    ds = xr.Dataset(
        {
            "temp": (("lat", "lon"), [[20, 25], [30, 35]]),
            "pressure": (("lat", "lon"), [[1000, 1010], [1020, 1030]]),
        },
        coords={
            "lat": [0, 1],
            "lon": [0, 1],
        },
    )
    ds.to_netcdf(netcdf_path)
    ds.close()
    print(f"DEBUG: NetCDF file size after write: {netcdf_path.stat().st_size}")
    return netcdf_path


@pytest.fixture(scope="function")
def mock_provenance_tracker():
    """Mocks a FileProvenanceTracker for testing."""

    class MockRunInfo:
        def __init__(self, run_id):
            self.run_id = run_id
            self.file_records = {}

        # We don't need get_most_recent_record for this test

    class MockProvenanceTracker:
        def __init__(self, run_id="test_run_id"):
            self.run_info = MockRunInfo(run_id)

        def record_output_file(
            self, model, file_path, year, short_name, description, **kwargs
        ):
            # Simulate recording, just return a dummy record
            return {
                "short_name": short_name,
                "file_path": file_path,
                "year": year,
                "model": model,
                **kwargs,
            }

        def record_input_file(
            self,
            model,
            file_path,
            year=None,
            short_name=None,
            description=None,
            **kwargs,
        ):
            return {
                "short_name": short_name,
                "file_path": file_path,
                "year": year,
                "model": model,
                **kwargs,
            }

    return MockProvenanceTracker()


# --- Test Cases for SnapshotManager ---


class TestSnapshotManager:

    def test_create_zarr_snapshot(
        self, snapshot_manager, dummy_zarr_store, mock_provenance_tracker
    ):
        run_id = "run_abc"
        year = 2025
        iteration = 0
        model = "activitysim"
        snapshot_type = "initialization"
        source_path = dummy_zarr_store
        artifact_format = "zarr"

        snapshot_id = snapshot_manager.create_snapshot(
            run_id=run_id,
            year=year,
            iteration=iteration,
            model=model,
            snapshot_type=snapshot_type,
            source_path=source_path,
            artifact_format=artifact_format,
            provenance_tracker=mock_provenance_tracker,
        )

        assert snapshot_id is not None
        assert isinstance(snapshot_id, str)

        # Verify data in DB
        snapshot_info = snapshot_manager.get_snapshot_info(snapshot_id)
        assert snapshot_info is not None
        assert snapshot_info["snapshot_id"] == snapshot_id
        assert snapshot_info["run_id"] == run_id
        assert snapshot_info["year"] == year
        assert snapshot_info["format"] == artifact_format
        assert snapshot_info["n_variables"] > 0
        assert snapshot_info["total_size_mb"] > 0
        assert snapshot_info["n_chunks"] is not None
        assert snapshot_info["chunk_manifest"] is not None  # JSON field

        # Verify artifact copied to archive
        archived_path = snapshot_manager.archive_root_path / Path(
            snapshot_info["artifact_path"]
        )
        assert archived_path.exists()
        assert archived_path.is_dir()

    def test_create_netcdf_snapshot(
        self, snapshot_manager, dummy_netcdf_file, mock_provenance_tracker
    ):
        run_id = "run_xyz"
        year = 2020
        iteration = 1
        model = "custom_model"
        snapshot_type = "output_data"
        source_path = dummy_netcdf_file
        artifact_format = "netcdf"

        snapshot_id = snapshot_manager.create_snapshot(
            run_id=run_id,
            year=year,
            iteration=iteration,
            model=model,
            snapshot_type=snapshot_type,
            source_path=source_path,
            artifact_format=artifact_format,
            provenance_tracker=mock_provenance_tracker,
        )

        assert snapshot_id is not None
        assert isinstance(snapshot_id, str)

        # Verify data in DB
        snapshot_info = snapshot_manager.get_snapshot_info(snapshot_id)
        assert snapshot_info is not None
        assert snapshot_info["snapshot_id"] == snapshot_id
        assert snapshot_info["run_id"] == run_id
        assert snapshot_info["year"] == year
        assert snapshot_info["format"] == artifact_format
        assert snapshot_info["n_variables"] > 0
        assert snapshot_info["total_size_mb"] > 0
        assert (
            snapshot_info.get("n_chunks") is None
        )  # NetCDF is not chunked in the same way
        assert snapshot_info.get("chunk_manifest") is None

        # Verify artifact copied to archive
        archived_path = snapshot_manager.archive_root_path / Path(
            snapshot_info["artifact_path"]
        )
        assert archived_path.exists()
        assert archived_path.is_file()

    def test_restore_zarr_snapshot(
        self, snapshot_manager, dummy_zarr_store, tmp_path, mock_provenance_tracker
    ):
        snapshot_id = snapshot_manager.create_snapshot(
            run_id="run_restore",
            year=2030,
            iteration=0,
            model="test",
            snapshot_type="merged",
            source_path=dummy_zarr_store,
            artifact_format="zarr",
            provenance_tracker=mock_provenance_tracker,
        )

        target_path = tmp_path / "restored_zarr.zarr"
        restored_path = snapshot_manager.restore_snapshot(snapshot_id, target_path)

        assert restored_path == target_path
        assert restored_path.exists()
        assert restored_path.is_dir()
        # Basic check to ensure it's a valid zarr
        with xr.open_zarr(restored_path) as ds:
            assert "SOV_TIME" in ds.data_vars

    def test_restore_netcdf_snapshot(
        self, snapshot_manager, dummy_netcdf_file, tmp_path, mock_provenance_tracker
    ):
        snapshot_id = snapshot_manager.create_snapshot(
            run_id="run_restore_nc",
            year=2031,
            iteration=0,
            model="test",
            snapshot_type="final_output",
            source_path=dummy_netcdf_file,
            artifact_format="netcdf",
            provenance_tracker=mock_provenance_tracker,
        )

        target_path = tmp_path / "restored.nc"
        restored_path = snapshot_manager.restore_snapshot(snapshot_id, target_path)

        assert restored_path == target_path
        assert restored_path.exists()
        assert restored_path.is_file()
        # Basic check to ensure it's a valid netcdf
        with xr.open_dataset(restored_path) as ds:
            assert "temp" in ds.data_vars

    def test_get_latest_snapshot_id_for_run(
        self, snapshot_manager, dummy_zarr_store, mock_provenance_tracker
    ):
        run_id = "run_sequence"
        # Create first snapshot
        snap_id_1 = snapshot_manager.create_snapshot(
            run_id=run_id,
            year=2000,
            iteration=0,
            model="m1",
            snapshot_type="initial",
            source_path=dummy_zarr_store,
            artifact_format="zarr",
            provenance_tracker=mock_provenance_tracker,
        )
        # Simulate some time passing or different iteration
        # Note: In real life, created_at would auto-increment, here we rely on order of insertion

        # Create second snapshot for the same run
        snap_id_2 = snapshot_manager.create_snapshot(
            run_id=run_id,
            year=2000,
            iteration=1,
            model="m2",
            snapshot_type="merged",
            source_path=dummy_zarr_store,
            artifact_format="zarr",
            provenance_tracker=mock_provenance_tracker,
        )

        latest_snap_id = snapshot_manager.get_latest_snapshot_id_for_run(run_id)
        assert latest_snap_id == snap_id_2

        # Test non-existent run
        assert (
            snapshot_manager.get_latest_snapshot_id_for_run("non_existent_run") is None
        )

    def test_create_snapshot_with_partial_skims_path(
        self, snapshot_manager, dummy_zarr_store, mock_provenance_tracker
    ):
        run_id = "run_partial_skims"
        year = 2026
        iteration = 1
        model = "beam_postprocessor"
        snapshot_type = "merged"
        source_path = dummy_zarr_store
        artifact_format = "zarr"

        # Mock a partial skims path relative to archive root
        mock_partial_skims_path = "run_partial_skims/iteration1/partial.zarr"

        snapshot_id = snapshot_manager.create_snapshot(
            run_id=run_id,
            year=year,
            iteration=iteration,
            model=model,
            snapshot_type=snapshot_type,
            source_path=source_path,
            artifact_format=artifact_format,
            provenance_tracker=mock_provenance_tracker,
            partial_skims_path=mock_partial_skims_path,
        )

        snapshot_info = snapshot_manager.get_snapshot_info(snapshot_id)
        assert snapshot_info is not None
        assert snapshot_info["partial_skims_path"] == mock_partial_skims_path


class TestSnapshotAnalysisManager:

    @pytest.fixture
    def populated_snapshots(
        self,
        snapshot_manager,
        dummy_zarr_store,
        dummy_netcdf_file,
        mock_provenance_tracker,
    ):
        # Create Zarr snapshots
        snap_id_zarr_1 = snapshot_manager.create_snapshot(
            run_id="run_A",
            year=2020,
            iteration=0,
            model="activitysim",
            snapshot_type="initial",
            source_path=dummy_zarr_store,
            artifact_format="zarr",
            provenance_tracker=mock_provenance_tracker,
        )
        snap_id_zarr_2 = snapshot_manager.create_snapshot(
            run_id="run_A",
            year=2020,
            iteration=1,
            model="beam_postprocessor",
            snapshot_type="merged",
            source_path=dummy_zarr_store,
            artifact_format="zarr",
            provenance_tracker=mock_provenance_tracker,
        )
        snap_id_zarr_3 = snapshot_manager.create_snapshot(
            run_id="run_B",
            year=2021,
            iteration=0,
            model="activitysim",
            snapshot_type="initial",
            source_path=dummy_zarr_store,
            artifact_format="zarr",
            provenance_tracker=mock_provenance_tracker,
        )
        # Create a NetCDF snapshot
        snap_id_netcdf_1 = snapshot_manager.create_snapshot(
            run_id="run_C",
            year=2020,
            iteration=0,
            model="custom_model",
            snapshot_type="final_output",
            source_path=dummy_netcdf_file,
            artifact_format="netcdf",
            provenance_tracker=mock_provenance_tracker,
        )
        return [snap_id_zarr_1, snap_id_zarr_2, snap_id_zarr_3, snap_id_netcdf_1]

    def test_build_view_all_snapshots(
        self, snapshot_analysis_manager, populated_snapshots
    ):
        view = snapshot_analysis_manager.build_view()
        assert view is not None
        assert isinstance(view, xr.Dataset)
        assert "snapshot" in view.dims
        assert len(view.coords["snapshot"]) == len(populated_snapshots)
        # Check if MultiIndex is created
        assert isinstance(view.indexes["snapshot"], pd.MultiIndex)
        assert "run_id" in view.indexes["snapshot"].names
        assert "year" in view.indexes["snapshot"].names
        assert "iteration" in view.indexes["snapshot"].names
        assert "snapshot_id" in view.indexes["snapshot"].names
        assert "SOV_TIME" in view.data_vars
        assert "temp" in view.data_vars

    def test_build_view_filter_by_run_id(
        self, snapshot_analysis_manager, populated_snapshots
    ):
        view = snapshot_analysis_manager.build_view(run_ids=["run_A"])
        assert len(view.coords["snapshot"]) == 2
        assert all(r_id == "run_A" for r_id in view.coords["run_id"].values)
        assert "SOV_TIME" in view.data_vars
        assert "temp" not in view.data_vars  # NetCDF from run_C should not be there

    def test_build_view_filter_by_year_and_model(
        self, snapshot_analysis_manager, populated_snapshots
    ):
        view = snapshot_analysis_manager.build_view(
            years=[2020], models=["activitysim"]
        )
        assert len(view.coords["snapshot"]) == 1
        assert view.coords["run_id"].values[0] == "run_A"
        assert view.coords["year"].values[0] == 2020
        assert view.coords["model"].values[0] == "activitysim"
        assert "SOV_TIME" in view.data_vars
        assert "temp" not in view.data_vars

    def test_build_view_filter_by_format(
        self, snapshot_analysis_manager, populated_snapshots
    ):
        view = snapshot_analysis_manager.build_view(formats=["netcdf"])
        assert len(view.coords["snapshot"]) == 1
        assert view.coords["run_id"].values[0] == "run_C"
        assert view.coords["format"].values[0] == "netcdf"
        assert "temp" in view.data_vars
        assert "SOV_TIME" not in view.data_vars

    def test_build_view_filter_by_variables(
        self, snapshot_analysis_manager, populated_snapshots
    ):
        view = snapshot_analysis_manager.build_view(variables=["SOV_TIME"])
        assert len(view.data_vars) == 1
        assert "SOV_TIME" in view.data_vars
        assert "SOV_DIST" not in view.data_vars
        assert (
            "temp" not in view.data_vars
        )  # Should exclude netcdf as it doesn't have SOV_TIME

    def test_build_view_no_matching_snapshots(self, snapshot_analysis_manager):
        view = snapshot_analysis_manager.build_view(run_ids=["non_existent_run"])
        assert isinstance(view, xr.Dataset)
        assert len(view.coords.get("snapshot", [])) == 0
        assert not view.data_vars

    def test_build_view_error_handling_invalid_artifact(
        self,
        snapshot_manager,
        dummy_zarr_store,
        monkeypatch,
        mock_provenance_tracker,
        snapshot_analysis_manager,
    ):
        # Create a snapshot with a path to a non-existent file
        corrupt_path = snapshot_manager.archive_root_path / "non_existent.zarr"
        corrupt_path.mkdir()  # Create a dir to make it seem plausible but empty
        snapshot_id = snapshot_manager.create_snapshot(
            run_id="run_corrupt",
            year=2022,
            iteration=0,
            model="test",
            snapshot_type="test",
            source_path=corrupt_path,
            artifact_format="zarr",
            provenance_tracker=mock_provenance_tracker,
        )
        # Need to ensure the metadata stored has the actual source_path
        snapshot_info = snapshot_manager.get_snapshot_info(snapshot_id)
        snapshot_info["artifact_path"] = str(
            corrupt_path.relative_to(snapshot_manager.archive_root_path)
        )
        snapshot_manager._insert_snapshot_record(
            snapshot_info
        )  # Update DB with corrupt path

        # Try to build view, should log warning and skip
        view = snapshot_analysis_manager.build_view(run_ids=["run_corrupt"])
        assert len(view.coords.get("snapshot", [])) == 0

        shutil.rmtree(corrupt_path)  # Clean up the dummy dir
