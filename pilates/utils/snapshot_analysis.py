"""
Manager for building multi-dimensional xarray views from archived data artifact snapshots.

This module implements the Analysis Layer of the data versioning architecture.
It allows users to query and combine multiple independent snapshots into a single,
ergonomic xarray.Dataset for cross-run, cross-year analysis.
"""

import logging
import uuid
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

import xarray as xr
import pandas as pd

from pilates.utils.duckdb_manager import DuckDBManager # Assuming this is the correct path to DuckDBManager

logger = logging.getLogger(__name__)

# Default local path for archiving artifacts. Must match SnapshotManager's.
DEFAULT_ARCHIVE_ROOT = Path("/tmp/pilates_data_archives")


class SnapshotAnalysisManager:
    """
    Manages the creation of unified xarray Dataset views from archived data snapshots.
    """

    def __init__(self, db_manager: DuckDBManager, archive_root_path: Optional[Path] = None):
        """
        Initialize the SnapshotAnalysisManager.

        Args:
            db_manager: An initialized DuckDBManager instance to interact with the database.
            archive_root_path: Optional. The root directory/path where artifacts are archived.
                               Defaults to DEFAULT_ARCHIVE_ROOT. Must match SnapshotManager's root.
        """
        self.db_manager = db_manager
        self.archive_root_path = archive_root_path or DEFAULT_ARCHIVE_ROOT
        logger.info(f"SnapshotAnalysisManager initialized with archive root: {self.archive_root_path}")

    def build_view(
        self,
        run_ids: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        iterations: Optional[List[int]] = None,
        models: Optional[List[str]] = None,
        snapshot_types: Optional[List[str]] = None,
        formats: Optional[List[str]] = None, # Added to filter by artifact format
        variables: Optional[List[str]] = None, # For filtering variables within each dataset
    ) -> xr.Dataset:
        """
        Builds a unified xarray.Dataset view from multiple archived snapshots matching specified criteria.

        Args:
            run_ids: Optional list of PILATES run IDs to include.
            years: Optional list of simulation years to include.
            iterations: Optional list of simulation iterations to include.
            models: Optional list of model names ('activitysim', 'beam_postprocessor', etc.) to include.
            snapshot_types: Optional list of snapshot types ('initialization', 'merged', etc.) to include.
            formats: Optional list of artifact formats ('zarr', 'netcdf') to include.
            variables: Optional list of data variables to load from each snapshot (e.g., ['SOV_TIME']).
                       If None, all variables are loaded.

        Returns:
            An xarray.Dataset with a MultiIndex on the 'snapshot' dimension,
            allowing powerful cross-run/year/iteration analysis. Returns an empty
            Dataset if no matching snapshots are found.
        """
        logger.info("Building multi-snapshot xarray view...")

        # 1. Construct SQL query to fetch snapshot metadata
        query_parts = ["SELECT * FROM snapshots WHERE 1=1"]
        params = []

        if run_ids:
            query_parts.append(f"run_id IN ({', '.join(['?'] * len(run_ids))})")
            params.extend(run_ids)
        if years:
            query_parts.append(f"year IN ({', '.join(['?'] * len(years))})")
            params.extend(years)
        if iterations:
            query_parts.append(f"iteration IN ({', '.join(['?'] * len(iterations))})")
            params.extend(iterations)
        if models:
            query_parts.append(f"model IN ({', '.join(['?'] * len(models))})")
            params.extend(models)
        if snapshot_types:
            query_parts.append(f"snapshot_type IN ({', '.join(['?'] * len(snapshot_types))})")
            params.extend(snapshot_types)
        if formats:
            query_parts.append(f"format IN ({', '.join(['?'] * len(formats))})")
            params.extend(formats)

        query = " AND ".join(query_parts) + " ORDER BY run_id, year, iteration, sub_iteration"
        
        # 2. Fetch snapshot metadata from the database
        try:
            # DuckDB connection.execute().fetchdf() returns a pandas DataFrame
            snapshot_records_df = self.db_manager.connection.execute(query, params).fetchdf()
        except Exception as e:
            logger.error(f"Error querying database for snapshots: {e}")
            return xr.Dataset()

        if snapshot_records_df.empty:
            logger.warning("No snapshots found matching the specified criteria.")
            return xr.Dataset()

        list_of_datasets = []
        for _, record in snapshot_records_df.iterrows():
            snapshot_id = record["snapshot_id"]
            artifact_format = record["format"]
            
            # Reconstruct the absolute path to the archived artifact
            relative_artifact_path = Path(record["artifact_path"])
            archived_artifact_full_path = self.archive_root_path / relative_artifact_path

            if not archived_artifact_full_path.exists():
                logger.warning(
                    f"Archived artifact for snapshot ID {snapshot_id} not found at "
                    f"{archived_artifact_full_path}. Skipping this snapshot."
                )
                continue
            
            try:
                # 3. Lazily open each artifact based on its format
                if artifact_format.lower() == "zarr":
                    ds = xr.open_zarr(archived_artifact_full_path)
                elif artifact_format.lower() == "netcdf":
                    ds = xr.open_dataset(archived_artifact_full_path)
                else:
                    logger.warning(f"Unsupported artifact format '{artifact_format}' for snapshot {snapshot_id}. Skipping.")
                    continue

                # Filter variables if requested
                if variables:
                    available_vars = [v for v in variables if v in ds]
                    if not available_vars:
                        logger.warning(f"None of the requested variables {variables} found in snapshot {snapshot_id}. Skipping.")
                        continue
                    ds = ds[available_vars]
                
                # 4. Assign metadata as new coordinates to each dataset
                # The "snapshot" dimension will be used for concatenation
                ds = ds.assign_coords(
                    snapshot_id=("snapshot", [snapshot_id]),
                    run_id=("snapshot", [record["run_id"]]),
                    year=("snapshot", [record["year"]]),
                    iteration=("snapshot", [record["iteration"]]),
                    model=("snapshot", [record["model"]]),
                    snapshot_type=("snapshot", [record["snapshot_type"]]),
                    format=("snapshot", [record["format"]])
                )
                list_of_datasets.append(ds)

            except Exception as e:
                logger.error(f"Failed to open or process snapshot ID {snapshot_id} from {archived_artifact_full_path}: {e}")
                continue

        if not list_of_datasets:
            logger.warning("No valid datasets were opened to combine.")
            return xr.Dataset()

        # 5. Concatenate all datasets along the new 'snapshot' dimension
        combined_view = xr.concat(list_of_datasets, dim="snapshot", combine_attrs="override", coords="minimal")

        # 6. Apply a MultiIndex for powerful, ergonomic selection
        final_view = combined_view.set_index(
            snapshot=["run_id", "year", "iteration", "snapshot_id"]
        )
        
        logger.info(f"Successfully built xarray view with {len(list_of_datasets)} snapshots.")
        return final_view

