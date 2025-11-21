"""
Manager for creating, storing, and restoring data artifact snapshots in a persistent archive
and recording their metadata in a DuckDB database.

This module implements the Storage Layer of the data versioning architecture.
It ensures that each version of a data artifact (e.g., Zarr store, NetCDF file)
is stored immutably and independently.
"""

import logging
import os
import shutil
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List

from pilates.utils.duckdb_manager import DuckDBManager
from pilates.utils.data_artifact_utils import (
    compute_zarr_chunk_manifest,
    get_artifact_metadata,
    copy_artifact,
)
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_ROOT = Path("/tmp/pilates_data_archives")


class SnapshotManager:
    """
    Manages the creation, archiving, and restoration of data artifact snapshots.
    """

    def __init__(self, db_manager: DuckDBManager, archive_root_path: Optional[Path] = None):
        """
        Initialize the SnapshotManager.

        Args:
            db_manager: An initialized DuckDBManager instance.
            archive_root_path: The root path where artifacts will be archived.
        """
        self.db_manager = db_manager
        self.archive_root_path = archive_root_path or DEFAULT_ARCHIVE_ROOT
        self.archive_root_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"SnapshotManager initialized with archive root: {self.archive_root_path}")

    def create_snapshot(
        self,
        run_id: str,
        year: int,
        iteration: int,
        model: str,
        snapshot_type: str,
        source_path: Path,
        artifact_format: str,
        sub_iteration: Optional[int] = None,
        parent_snapshot_id: Optional[str] = None,
        provenance_tracker: Optional[FileProvenanceTracker] = None,
        **kwargs: Any,
    ) -> str:
        """
        Creates an immutable snapshot of a data artifact.
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source artifact not found at {source_path}")

        snapshot_id = str(uuid.uuid4())
        relative_artifact_dir = Path(snapshot_id)
        artifact_filename = "artifact.zarr" if artifact_format.lower() == "zarr" else f"artifact.{artifact_format.lower()}"
        relative_artifact_path = relative_artifact_dir / artifact_filename
        
        destination_path = self.archive_root_path / relative_artifact_path

        logger.info(f"Creating snapshot {snapshot_id} for {source_path} ({artifact_format})")
        
        copy_artifact(source_path, destination_path, artifact_format)

        artifact_metadata = get_artifact_metadata(destination_path, artifact_format)
        
        chunk_manifest = None
        if artifact_format.lower() == "zarr":
            chunk_manifest = compute_zarr_chunk_manifest(destination_path)

        snapshot_record = {
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "year": year,
            "iteration": iteration,
            "sub_iteration": sub_iteration,
            "model": model,
            "snapshot_type": snapshot_type,
            "parent_snapshot_id": parent_snapshot_id,
            "created_at": datetime.now(),
            "format": artifact_format,
            "artifact_path": str(relative_artifact_path),
            "n_variables": artifact_metadata.get("n_variables"),
            "total_size_mb": artifact_metadata.get("total_size_mb", artifact_metadata.get("file_size_mb")),
            "n_chunks": artifact_metadata.get("n_chunks", len(chunk_manifest) if chunk_manifest else None),
            "chunk_manifest": json.dumps(chunk_manifest) if chunk_manifest else None,
            **kwargs,
        }
        
        self._insert_snapshot_record(snapshot_record)

        logger.info(f"Snapshot {snapshot_id} created and metadata recorded.")

        if provenance_tracker:
            provenance_tracker.record_output_file(
                model=model,
                file_path=str(destination_path),
                year=year,
                short_name=f"{model}_{snapshot_type}_{year}_{iteration}",
                description=f"Data artifact snapshot (ID: {snapshot_id})",
                metadata={"snapshot_id": snapshot_id, "format": artifact_format},
            )

        return snapshot_id

    def restore_snapshot(self, snapshot_id: str, target_path: Path) -> Path:
        """
        Restores a data artifact snapshot from the archive to a target local path.
        """
        snapshot_info = self.get_snapshot_info(snapshot_id)
        if not snapshot_info:
            raise ValueError(f"Snapshot ID '{snapshot_id}' not found in database.")

        relative_artifact_path = Path(snapshot_info["artifact_path"])
        archived_artifact_full_path = self.archive_root_path / relative_artifact_path
        artifact_format = snapshot_info["format"]

        if not archived_artifact_full_path.exists():
            raise FileNotFoundError(f"Archived artifact not found at {archived_artifact_full_path}.")

        logger.info(f"Restoring snapshot {snapshot_id} (format: {artifact_format}) from {archived_artifact_full_path} to {target_path}")
        copy_artifact(archived_artifact_full_path, target_path, artifact_format)
        logger.info(f"Snapshot {snapshot_id} restored to {target_path}")
        return target_path

    def get_snapshot_info(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata for a specific snapshot from the database.
        """
        query = "SELECT * FROM snapshots WHERE snapshot_id = ?"
        try:
            result_df = self.db_manager.execute_sql(query, [snapshot_id]).fetchdf()
            if not result_df.empty:
                record = result_df.iloc[0].to_dict()
                if 'chunk_manifest' in record and isinstance(record['chunk_manifest'], str):
                    try:
                        record['chunk_manifest'] = json.loads(record['chunk_manifest'])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode chunk_manifest for {snapshot_id}")
                return record
            return None
        except Exception as e:
            logger.error(f"Error querying snapshot record for ID {snapshot_id}: {e}")
            return None

    def get_latest_snapshot_id_for_run(self, run_id: str) -> Optional[str]:
        """
        Retrieves the most recent snapshot_id for a given run_id.

        Args:
            run_id: The PILATES run ID to look up.

        Returns:
            The latest snapshot_id as a string, or None if no snapshots are found.
        """
        query = "SELECT snapshot_id FROM snapshots WHERE run_id = ? ORDER BY created_at DESC LIMIT 1"
        try:
            result = self.db_manager.execute_sql(query, [run_id]).fetchone()
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Error querying for latest snapshot for run_id {run_id}: {e}")
            return None

    def _insert_snapshot_record(self, snapshot_data: Dict[str, Any]):
        """Helper to insert a snapshot record into the database."""
        # This list defines the exact order and columns for the INSERT statement
        # It must match the `snapshots` table schema.
        ordered_columns = [
            "snapshot_id", "run_id", "year", "iteration", "sub_iteration",
            "snapshot_type", "model", "parent_snapshot_id", "created_at",
            "format", "artifact_path", "n_variables", "n_chunks", "total_size_mb",
            "partial_skims_path", "partial_skims_n_variables", "partial_skims_n_chunks",
            "partial_skims_total_size_mb", "changed_chunks", "chunk_manifest"
        ]

        # Prepare values in the correct order, substituting None for missing keys
        values_to_insert = [snapshot_data.get(col) for col in ordered_columns]

        # Construct the SQL query with placeholders
        col_names = ", ".join(ordered_columns)
        placeholders = ", ".join(['?'] * len(ordered_columns))
        
        # Build the ON CONFLICT DO UPDATE SET clause
        update_set_clauses = [f"{col} = EXCLUDED.{col}" for col in ordered_columns if col != "snapshot_id"]
        update_clause = ", ".join(update_set_clauses)

        query = f"""
            INSERT INTO snapshots ({col_names}) VALUES ({placeholders})
            ON CONFLICT (snapshot_id) DO UPDATE SET
                {update_clause}
        """

        try:
            self.db_manager.execute_sql(query, values_to_insert)
            logger.debug(f"Inserted snapshot {snapshot_data['snapshot_id']} into database.")
        except Exception as e:
            logger.error(f"Failed to insert snapshot record {snapshot_data.get('snapshot_id', 'N/A')}: {e}")
            raise