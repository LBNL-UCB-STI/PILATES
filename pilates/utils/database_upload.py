"""
Database upload utilities for PILATES.

This module provides utilities for uploading PILATES run data to
the configured database backend after run completion.
"""

import json
import logging
import os
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

from pilates.generic.records import PilatesRunInfo, FileRecord, ModelRunInfo
from pilates.utils.database import DatabaseManager
from pilates.utils.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


def create_database_manager(settings: Dict[str, Any]) -> Optional[DatabaseManager]:
    """
    Create a database manager based on settings configuration.

    Args:
        settings: PILATES settings dictionary

    Returns:
        Configured database manager or None if not configured
    """
    db_config = settings.get("database", {})

    if not db_config or not db_config.get("enabled", False):
        logger.info("Database upload not configured or disabled")
        return None

    db_type = db_config.get("type", "duckdb").lower()
    db_path = db_config.get("path")

    if not db_path:
        logger.warning("Database path not specified in configuration")
        return None

    # Expand relative paths relative to workspace or project root
    if not os.path.isabs(db_path):
        # Try to make it relative to the current working directory
        db_path = os.path.abspath(db_path)

    try:
        if db_type == "duckdb":
            manager = DuckDBManager(db_path, **db_config)
            if manager.initialize_database():
                logger.info(f"Initialized DuckDB database at {db_path}")
                return manager
            else:
                logger.error("Failed to initialize DuckDB database")
                return None
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to create database manager: {e}")
        return None


def _upload_activitysim_csv_data(
    run_info_path: str, run_info: PilatesRunInfo, db_manager: DatabaseManager
) -> bool:
    """
    Upload ActivitySim CSV input files to database if they exist.

    Args:
        run_info_path: Path to run_info.json file
        run_info: Loaded run information
        db_manager: Database manager instance

    Returns:
        bool: True if CSV data uploaded successfully or no CSV data found
    """
    try:
        # Find the run directory
        run_directory = os.path.dirname(run_info_path)

        # Look for ActivitySim input CSV files
        activitysim_data_dir = os.path.join(run_directory, "activitysim", "data")

        if not os.path.exists(activitysim_data_dir):
            logger.info("No activitysim/data directory found - skipping CSV upload")
            return True  # Not an error

        # Standard ActivitySim input files that should be treated as processed data
        activitysim_csv_files = {
            "households.csv": "households",
            "persons.csv": "persons",
            "land_use.csv": "land_use",
        }

        csv_upload_success = True
        uploaded_files = []

        for csv_filename, table_name in activitysim_csv_files.items():
            csv_path = os.path.join(activitysim_data_dir, csv_filename)

            if os.path.exists(csv_path):
                try:
                    # Load CSV data
                    logger.info(f"📊 Loading ActivitySim CSV: {csv_filename}")
                    df = pd.read_csv(csv_path)

                    # Create file record for this CSV
                    file_record = FileRecord(
                        unique_id=f"{run_info.run_id}_{table_name}_csv",
                        openlineage_id=str(uuid.uuid4()),
                        file_path=csv_path,
                        created_at=datetime.now().isoformat(),
                        short_name=f"activitysim_{table_name}_input",
                        description=f"ActivitySim input table '{table_name}' from CSV file",
                        models=["activitysim"],
                        schema=[
                            {"name": col, "type": str(dtype)}
                            for col, dtype in df.dtypes.items()
                        ],
                        metadata={
                            "table_name": f"{table_name}_processed",
                            "data_type": "processed_activitysim",
                            "source_file": csv_filename,
                            "record_count": len(df),
                            "extraction_method": "csv_upload_from_run_info",
                        },
                    )

                    # First, add the file record to the database
                    import json

                    conn = db_manager._get_connection()
                    conn.execute(
                        """
                        INSERT INTO file_records (
                            unique_id, run_id, openlineage_id, file_path, created_at,
                            short_name, description, models, schema, metadata, exists
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        [
                            file_record.unique_id,
                            run_info.run_id,
                            file_record.openlineage_id,
                            file_record.file_path,
                            file_record.created_at,
                            file_record.short_name,
                            file_record.description,
                            file_record.models,
                            json.dumps(file_record.schema),
                            json.dumps(file_record.metadata),
                            file_record.exists,
                        ],
                    )

                    # Then upload to processed ActivitySim table
                    success = db_manager.store_activitysim_data(
                        table_name=table_name,
                        df=df,
                        file_record_id=file_record.unique_id,
                        run_id=run_info.run_id,
                        openlineage_id=file_record.openlineage_id,
                    )

                    if success:
                        uploaded_files.append(f"{table_name} ({len(df)} records)")
                        logger.info(
                            f"  ✅ Successfully uploaded {csv_filename} as processed {table_name} data"
                        )
                    else:
                        logger.error(f"  ❌ Failed to upload {csv_filename}")
                        csv_upload_success = False

                except Exception as e:
                    logger.error(f"Failed to process {csv_filename}: {e}")
                    csv_upload_success = False
            else:
                logger.debug(f"ActivitySim CSV file not found: {csv_path}")

        if uploaded_files:
            logger.info(
                f"📤 Successfully uploaded ActivitySim CSV data: {', '.join(uploaded_files)}"
            )
        else:
            logger.info("No ActivitySim CSV files found to upload")

        return csv_upload_success

    except Exception as e:
        logger.error(f"Failed to upload ActivitySim CSV data: {e}")
        return False

def _upload_h5_data(
    file_path: str, run_info: PilatesRunInfo, db_manager: DatabaseManager, file_record: FileRecord
) -> bool:
    """
    Upload data from an H5 file to the database.
    """
    try:
        import h5py

        with h5py.File(file_path, "r") as f:
            for table_name in f.keys():
                logger.info(f"Reading table '{table_name}' from H5 file: {file_path}")
                try:
                    df = pd.read_hdf(file_path, key=table_name)

                    # Calculate hash of the table data
                    table_hash = hashlib.sha256(df.to_csv().encode()).hexdigest()

                    # Create a FileRecord for the table
                    table_file_record = FileRecord(
                        unique_id=table_hash,
                        openlineage_id=str(uuid.uuid4()),
                        file_path=f"{file_path}/{table_name}",
                        created_at=datetime.now().isoformat(),
                        short_name=f"{file_record.short_name}_{table_name}",
                        description=f"Table '{table_name}' from H5 file '{file_record.short_name}'",
                        models=file_record.models,
                        schema=[
                            {"name": col, "type": str(dtype)}
                            for col, dtype in df.dtypes.items()
                        ],
                        metadata={
                            "table_name": table_name,
                            "source_h5_file": file_record.file_path,
                        },
                        source_file_paths=[file_record.unique_id],
                    )

                    # Check if the table data already exists
                    if db_manager.check_dataset_exists_by_hash(table_file_record.unique_id):
                        logger.info(f"Skipping table upload, dataset already exists: {table_name}")
                        continue

                    # Upload the table's FileRecord
                    db_manager.upload_file_record(table_file_record, run_info.run_id)

                    db_manager.store_urbansim_raw_data(
                            table_name,
                            df,
                            table_file_record.unique_id,
                            run_info.run_id,
                            file_record.openlineage_id,
                            table_file_record.openlineage_id,
                        )
                except Exception as e:
                    logger.warning(f"Could not read table '{table_name}' from H5 file: {e}")
        return True
    except ImportError:
        logger.error("h5py is required to upload H5 files. Please install it.")
        return False
    except Exception as e:
        logger.error(f"Failed to upload H5 file {file_path}: {e}")
        return False

def _upload_data_from_file_records(
    run_info: PilatesRunInfo, db_manager: DatabaseManager
) -> bool:
    """
    Upload data from file records to the database.
    """
    try:
        for file_record in run_info.file_records.values():
            if file_record.file_path.endswith(".h5"):
                logger.info(f"Found H5 file: {file_record.file_path}")
                _upload_h5_data(file_record.file_path, run_info, db_manager, file_record)
            elif file_record.file_path.endswith(".csv"):
                # This is a CSV file.
                logger.info(f"Found CSV file: {file_record.file_path}")
                # TODO: Implement generic CSV upload

        return True
    except Exception as e:
        logger.error(f"Failed to upload data from file records: {e}")
        return False


def upload_run_info_to_database(run_info_path: str, settings: Dict[str, Any]) -> bool:
    """
    Upload run information from run_info.json to the configured database.

    Args:
        run_info_path: Path to run_info.json file
        settings: PILATES settings dictionary

    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        # Load run_info.json
        if not os.path.exists(run_info_path):
            logger.error(f"run_info.json not found at {run_info_path}")
            return False

        with open(run_info_path, "r") as f:
            run_data = json.load(f)

        # Convert file records from dict format to FileRecord objects
        file_records = {}
        for unique_id, record_data in run_data.get("file_records", {}).items():
            if isinstance(record_data, dict):
                # Convert dict to FileRecord object
                file_records[unique_id] = FileRecord(
                    unique_id=record_data.get("unique_id", unique_id),
                    openlineage_id=record_data.get("openlineage_id", ""),
                    file_path=record_data.get("file_path", ""),
                    created_at=record_data.get("created_at", ""),
                    short_name=record_data.get("short_name", ""),
                    description=record_data.get("description", ""),
                    models=record_data.get("models", []),
                    schema=record_data.get("schema", []),
                    metadata=record_data.get("metadata", {}),
                    year=record_data.get("year"),
                    producing_run_id=record_data.get("producing_run_id"),
                    consuming_run_ids=record_data.get("consuming_run_ids", []),
                    source_file_paths=record_data.get("source_file_paths", []),
                    exists=record_data.get("exists", True),
                )
            else:
                file_records[unique_id] = record_data

        # Convert model runs from dict format to ModelRunInfo objects
        model_runs = {}
        for unique_id, model_data in run_data.get("model_runs", {}).items():
            if isinstance(model_data, dict):
                # Convert dict to ModelRunInfo object
                model_runs[unique_id] = ModelRunInfo(
                    unique_id=model_data.get("unique_id", unique_id),
                    openlineage_id=model_data.get("openlineage_id", ""),
                    model=model_data.get("model", ""),
                    year=model_data.get("year"),
                    iteration=model_data.get("iteration"),
                    description=model_data.get("description", ""),
                    created_at=model_data.get("created_at", ""),
                    completed_at=model_data.get("completed_at"),
                    status=model_data.get("status", "unknown"),
                    input_record_hashes=model_data.get("input_record_hashes", []),
                    output_record_hashes=model_data.get("output_record_hashes", []),
                )
            else:
                model_runs[unique_id] = model_data

        # Convert to PilatesRunInfo object for type safety
        run_info = PilatesRunInfo(
            run_id=run_data.get("run_id"),
            created_at=run_data.get("created_at"),
            start_year=run_data.get("start_year"),
            end_year=run_data.get("end_year"),
            models_used=run_data.get("models_used", []),
            settings_hash=run_data.get("settings_hash"),
            code_version=run_data.get("code_version"),
            hostname=run_data.get("hostname"),
            file_records=file_records,
            repo_records=run_data.get("repo_records", {}),
            model_runs=model_runs,
            config_snapshot=run_data.get("config_snapshot"),
            openlineage_event_metadata=run_data.get("openlineage_event_metadata", []),
        )

        # Create database manager
        db_manager = create_database_manager(settings)
        if not db_manager:
            logger.info("Database upload skipped - not configured")
            return True  # Not an error if not configured

        # Upload metadata first
        with db_manager:
            metadata_success = db_manager.upload_run_data(run_info)
            if not metadata_success:
                logger.error(f"Failed to upload run metadata for {run_info.run_id}")
                return False

            logger.info(f"Successfully uploaded run metadata for {run_info.run_id}")

            # Try to upload ActivitySim CSV data if present
            data_success = _upload_activitysim_csv_data(
                run_info_path, run_info, db_manager
            )

            # Upload data from file records
            record_data_success = _upload_data_from_file_records(run_info, db_manager)


            if metadata_success and data_success and record_data_success:
                logger.info(
                    f"Successfully uploaded ALL data (metadata + CSV files) for run {run_info.run_id}"
                )
                return True
            elif metadata_success:
                logger.info(
                    f"Successfully uploaded run metadata for {run_info.run_id} (no CSV data found)"
                )
                return True
            else:
                return False

    except Exception as e:
        logger.error(f"Error uploading run data to database: {e}")
        return False


def upload_run_directory_to_database(
    run_directory: str, settings: Dict[str, Any]
) -> bool:
    """
    Upload run information from a run directory to the database.

    Args:
        run_directory: Path to run directory containing run_info.json
        settings: PILATES settings dictionary

    Returns:
        bool: True if upload successful, False otherwise
    """
    run_info_path = os.path.join(run_directory, "run_info.json")
    return upload_run_info_to_database(run_info_path, settings)


def batch_upload_runs_to_database(
    run_directories: list, settings: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Upload multiple runs to the database in batch.

    Args:
        run_directories: List of run directory paths
        settings: PILATES settings dictionary

    Returns:
        Dict mapping run directory to success status
    """
    results = {}

    db_manager = create_database_manager(settings)
    if not db_manager:
        logger.info("Database upload skipped - not configured")
        return {run_dir: True for run_dir in run_directories}

    with db_manager:
        for run_dir in run_directories:
            try:
                run_info_path = os.path.join(run_dir, "run_info.json")
                if not os.path.exists(run_info_path):
                    logger.warning(f"run_info.json not found in {run_dir}")
                    results[run_dir] = False
                    continue

                with open(run_info_path, "r") as f:
                    run_data = json.load(f)

                run_info = PilatesRunInfo(
                    run_id=run_data.get("run_id"),
                    created_at=run_data.get("created_at"),
                    start_year=run_data.get("start_year"),
                    end_year=run_data.get("end_year"),
                    models_used=run_data.get("models_used", []),
                    settings_hash=run_data.get("settings_hash"),
                    code_version=run_data.get("code_version"),
                    hostname=run_data.get("hostname"),
                    file_records=run_data.get("file_records", {}),
                    repo_records=run_data.get("repo_records", {}),
                    model_runs=run_data.get("model_runs", {}),
                    config_snapshot=run_data.get("config_snapshot"),
                    openlineage_event_metadata=run_data.get(
                        "openlineage_event_metadata", []
                    ),
                )

                success = db_manager.upload_run_data(run_info)
                results[run_dir] = success

                if success:
                    logger.info(f"Uploaded {run_info.run_id} from {run_dir}")
                else:
                    logger.error(f"Failed to upload run from {run_dir}")

            except Exception as e:
                logger.error(f"Error processing {run_dir}: {e}")
                results[run_dir] = False

    successful = sum(1 for success in results.values() if success)
    total = len(results)
    logger.info(
        f"Batch upload completed: {successful}/{total} runs uploaded successfully"
    )

    return results