#!/usr/bin/env python3
"""
Extract ActivitySim inputs from UrbanSim H5 datastore and upload to database.

This utility extracts the preprocessed ActivitySim input data from an UrbanSim H5 file
and uploads it to the database with proper OpenLineage tracking. This allows bypassing
the complex ActivitySimPreprocessor when UrbanSim is turned off.
"""

import argparse
import hashlib
import logging
import os
import sys
import uuid
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import h5py


# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pilates.activitysim.preprocessor import (
    read_zone_geoms,
    process_raw_h5_files,
    _create_land_use_table,
)
from pilates.utils.database_upload import create_database_manager
from pilates.generic.records import (
    FileRecord,
    PilatesRunInfo,
    ModelRunInfo,
    OpenLineageEventMetadata,
)
from pilates.utils.config_snapshot import ConfigSnapshotManager

logger = logging.getLogger(__name__)


class H5ActivitySimExtractor:
    """
    Extracts ActivitySim inputs from UrbanSim H5 files and uploads to database.
    """

    # Standard ActivitySim input tables expected in H5 files (processed data)
    ACTIVITYSIM_TABLES = [
        "households",
        "persons",
        "land_use",
        "accessibility",
        "buildings",
        "parcels",
        "zones",
        "maz",
        "taz",
        "regional_demographic_controls",
        "employment_controls",
        "household_controls",
    ]

    # Raw UrbanSim tables that should be preserved
    RAW_URBANSIM_TABLES = [
        "households",  # Raw household data before ActivitySim processing
        "persons",  # Raw person data before ActivitySim processing
        "jobs",  # Raw job data
        "blocks",  # Raw block data
        "buildings",  # Raw building data
        "parcels",  # Raw parcel data
    ]

    # Additional tables that might be present and useful
    OPTIONAL_TABLES = [
        "establishments",
        "travel_data",
        "poi",
        "network",
        "transit_stops",
    ]

    def __init__(
        self,
        h5_path: str,
        settings: Dict[str, Any],
        run_id: Optional[str] = None,
        parallel_uploads: bool = True,
        max_workers: int = 1,
    ):
        """
        Initialize H5 extractor.

        Args:
            h5_path: Path to UrbanSim H5 file
            settings: PILATES settings for database configuration
            run_id: Optional run ID, will generate if not provided
            parallel_uploads: Whether to enable parallel table uploads
            max_workers: Maximum number of parallel upload workers
        """
        self.h5_path = os.path.abspath(h5_path)
        self.settings = settings
        self.run_id = run_id or f"h5-extract-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.parallel_uploads = parallel_uploads
        self.max_workers = max_workers
        self.db_manager = None

        # Track extracted data for upload
        self.extracted_tables = {}  # Processed ActivitySim tables
        self.raw_tables = {}  # Raw UrbanSim tables
        self.file_records = {}

        if self.max_workers > 1:
            raise NotImplementedError(
                "Parallel uploads with more than 1 worker are not yet implemented, sorry!"
            )

    def _get_h5_table_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze H5 file structure to identify available tables.

        Returns:
            Dictionary mapping table names to metadata
        """
        table_info = {}

        try:
            with h5py.File(self.h5_path, "r") as h5_file:

                def visit_func(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # Try to get basic info about the dataset
                        table_info[name] = {
                            "shape": obj.shape,
                            "dtype": str(obj.dtype),
                            "size_mb": obj.size * obj.dtype.itemsize / (1024 * 1024),
                        }

                h5_file.visititems(visit_func)

        except Exception as e:
            logger.error(f"Failed to analyze H5 file structure: {e}")

        return table_info

    def _detect_h5_structure(self, year: Optional[int] = None) -> str:
        """
        Detect H5 file structure and determine the appropriate table prefix.

        This follows the same logic as pilates.utils.io.read_datastore():
        - Base year/input data: tables at root level (table_prefix_yr = "")
        - Forecast years/output data: tables with year prefix (table_prefix_yr = str(year))

        Returns:
            str: Table prefix to use (empty string or year)
        """
        try:
            with pd.HDFStore(self.h5_path, mode="r") as store:
                # First try to find households table at root level
                if "/households" in store or "households" in store:
                    logger.info("Detected base year/input format: tables at root level")
                    return ""

                # If year is provided, try year-prefixed format
                if year:
                    year_prefix = str(year)
                    if f"/{year_prefix}/households" in store:
                        logger.info(
                            f"Detected forecast year format: tables under /{year_prefix}/"
                        )
                        return year_prefix

                # Fall back to empty prefix
                logger.warning("Could not detect H5 structure, using root level")
                return ""

        except Exception as e:
            logger.error(f"Failed to detect H5 structure: {e}")
            return ""

    def _extract_table_from_h5(
        self, table_name: str, year: Optional[int] = None, table_type: str = "processed"
    ) -> Optional[pd.DataFrame]:
        """
        Extract a specific table from the H5 file.

        Args:
            table_name: Name of table to extract
            year: Optional year filter for time-series data
            table_type: Type of table ('processed' for ActivitySim, 'raw' for UrbanSim)

        Returns:
            DataFrame with extracted data or None if not found
        """
        try:
            with pd.HDFStore(self.h5_path, mode="r") as store:
                # Detect H5 structure inline to avoid circular calls
                table_prefix = ""

                # First try to find households table at root level
                if "/households" in store or "households" in store:
                    table_prefix = ""
                # If year is provided, try year-prefixed format
                elif year and f"/{year}/households" in store:
                    table_prefix = str(year)
                # Default to empty prefix
                else:
                    table_prefix = ""

                # Build possible paths based on detected structure
                possible_paths = []

                if table_prefix:
                    # Year-prefixed format (forecast year data)
                    possible_paths.extend(
                        [
                            f"/{table_prefix}/{table_name}",
                            f"/base/{table_prefix}/{table_name}",
                        ]
                    )
                else:
                    # Root level format (base year/input data)
                    possible_paths.extend(
                        [
                            f"/{table_name}",
                            f"/base/{table_name}",
                            f"/data/{table_name}",
                            table_name,  # No leading slash
                        ]
                    )

                # Also try alternative paths as fallback
                if year and not table_prefix:
                    possible_paths.extend(
                        [
                            f"/{year}/{table_name}",
                            f"/base/{year}/{table_name}",
                        ]
                    )
                elif not table_prefix:
                    possible_paths.extend(
                        [
                            f"/{table_name}",
                            f"/base/{table_name}",
                            f"/data/{table_name}",
                            table_name,
                        ]
                    )

                for path in possible_paths:
                    if path and path in store:
                        logger.info(
                            f"Found {table_type} table {table_name} at path {path}"
                        )
                        df = store.select(path)

                        # Add metadata columns
                        df["_table_name"] = table_name
                        df["_table_type"] = table_type
                        df["_h5_path"] = path  # Store the actual H5 path found
                        df["_table_prefix"] = table_prefix  # Store detected prefix
                        df["_extracted_at"] = datetime.now().isoformat()
                        df["_source_h5"] = os.path.basename(self.h5_path)
                        if year:
                            df["_year"] = year

                        return df

                # Log available tables for debugging
                available_tables = list(store.keys())
                logger.warning(
                    f"Table {table_name} not found in H5 file. Available tables: {available_tables[:10]}..."
                )
                return None

        except Exception as e:
            logger.error(f"Failed to extract table {table_name}: {e}")
            return None

    def _create_file_record(
        self, table_name: str, df: pd.DataFrame, output_path: Optional[str] = None
    ) -> FileRecord:
        """
        Create a FileRecord for an extracted table.

        Args:
            table_name: Name of the extracted table
            df: DataFrame containing the data
            output_path: Optional path where data was saved

        Returns:
            FileRecord for database upload
        """
        # Create content hash based on data (convert dtypes to strings)
        dtypes_str = {col: str(dtype) for col, dtype in df.dtypes.items()}
        content_str = f"{table_name}_{len(df)}_{list(df.columns)}_{dtypes_str}"
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        # Generate schema information
        schema = [{"name": col, "type": str(dtype)} for col, dtype in df.dtypes.items()]

        file_record = FileRecord(
            unique_id=content_hash,
            openlineage_id=str(uuid.uuid4()),
            file_path=output_path or f"memory://{table_name}",
            created_at=datetime.now().isoformat(),
            short_name=f"activitysim_input_{table_name}",
            description=f"ActivitySim input table '{table_name}' extracted from UrbanSim H5 file",
            models=["activitysim"],
            schema=schema,
            metadata={
                "table_name": table_name,
                "record_count": len(df),
                "source_h5_file": self.h5_path,
                "extraction_method": "h5_to_database",
                "data_types": dtypes_str,  # Use string-converted dtypes
            },
        )

        return file_record

    def extract_raw_urbansim_data(
        self,
        year: Optional[int] = None,
        save_csv: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract raw UrbanSim data tables from H5 file.

        Args:
            year: Optional year to extract (for time-series data)
            save_csv: Whether to save extracted data as CSV files
            output_dir: Directory to save CSV files (if save_csv=True)

        Returns:
            Dictionary mapping table names to DataFrames
        """
        logger.info(f"Extracting raw UrbanSim data from {self.h5_path}")
        if year:
            logger.info(f"Target year: {year}")

        # Auto-detect structure
        table_prefix = self._detect_h5_structure(year)
        if table_prefix:
            logger.info(f"Using year-prefixed format: /{table_prefix}/[table]")
        else:
            logger.info("Using root-level format: /[table]")

        raw_data = {}

        # Extract raw UrbanSim tables
        for table_name in self.RAW_URBANSIM_TABLES:
            df = self._extract_table_from_h5(table_name, year, table_type="raw")
            if df is not None:
                raw_data[table_name] = df
                logger.info(f"Extracted raw {table_name}: {len(df)} records")

                # Optionally save as CSV
                if save_csv and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    csv_path = os.path.join(output_dir, f"{table_name}_raw.csv")
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved raw {table_name} to {csv_path}")

        self.raw_tables = raw_data
        logger.info(f"Successfully extracted {len(raw_data)} raw UrbanSim tables")

        return raw_data

    def extract_activitysim_inputs(
        self,
        year: Optional[int] = None,
        save_csv: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract all ActivitySim input tables from H5 file.

        Args:
            year: Optional year to extract (for time-series data)
            save_csv: Whether to save extracted data as CSV files
            output_dir: Directory to save CSV files (if save_csv=True)

        Returns:
            Dictionary mapping table names to DataFrames
        """
        logger.info(f"Extracting processed ActivitySim inputs from {self.h5_path}")
        if year:
            logger.info(f"Target year: {year}")

        # Auto-detect structure
        table_prefix = self._detect_h5_structure(year)
        if table_prefix:
            logger.info(f"Using year-prefixed format: /{table_prefix}/[table]")
        else:
            logger.info("Using root-level format: /[table]")

        # First, analyze what's available
        table_info = self._get_h5_table_info()
        logger.info(f"Found {len(table_info)} datasets in H5 file")

        # Log some key tables found for debugging
        key_tables = ["households", "persons", "land_use", "blocks"]
        for key_table in key_tables:
            found_paths = [path for path in table_info.keys() if key_table in path]
            if found_paths:
                logger.info(f"  {key_table} found at: {found_paths}")

        extracted_data = {}

        # Extract standard ActivitySim tables (processed data)
        for table_name in self.ACTIVITYSIM_TABLES:
            df = self._extract_table_from_h5(table_name, year, table_type="processed")
            if df is not None:
                extracted_data[table_name] = df
                logger.info(f"Extracted processed {table_name}: {len(df)} records")

                # Optionally save as CSV
                if save_csv and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    csv_path = os.path.join(output_dir, f"{table_name}_processed.csv")
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved processed {table_name} to {csv_path}")

        # Extract optional tables that exist
        for table_name in self.OPTIONAL_TABLES:
            df = self._extract_table_from_h5(table_name, year, table_type="processed")
            if df is not None:
                extracted_data[table_name] = df
                logger.info(f"Extracted optional table {table_name}: {len(df)} records")

                if save_csv and output_dir:
                    csv_path = os.path.join(output_dir, f"{table_name}_processed.csv")
                    df.to_csv(csv_path, index=False)

        # Generate land_use table if not present but we have raw data
        if "land_use" not in extracted_data:
            logger.info("land_use table not found in H5, generating from raw data...")
            land_use_df = self._generate_land_use_table(year)
            if land_use_df is not None:
                extracted_data["land_use"] = land_use_df
                logger.info(f"Generated land_use table: {len(land_use_df)} records")

                if save_csv and output_dir:
                    csv_path = os.path.join(output_dir, "land_use_processed.csv")
                    land_use_df.to_csv(csv_path, index=False)
                    logger.info(f"Saved generated land_use to {csv_path}")

        self.extracted_tables = extracted_data
        logger.info(
            f"Successfully extracted {len(extracted_data)} processed ActivitySim tables"
        )

        return extracted_data

    def _generate_land_use_table(
        self, year: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Generate land_use table using ActivitySim preprocessor logic."""
        try:
            region = self.settings["region"]
            FIPS = self.settings["FIPS"][region]
            asim_zone_id_col = "TAZ"
            # TODO: Generalize this or add it to settings.yaml
            input_zone_id_col = self.settings.get("geoms_index_col", "zone_id")
            zones = read_zone_geoms(
                self.settings,
                year or 2017,
                asim_zone_id_col=asim_zone_id_col,
                default_zone_id_col=input_zone_id_col,
            )

            # Extract raw data needed for land_use generation
            raw_households = self._extract_table_from_h5(
                "households", year, table_type="raw"
            )
            raw_persons = self._extract_table_from_h5("persons", year, table_type="raw")
            raw_jobs = self._extract_table_from_h5("jobs", year, table_type="raw")
            raw_blocks = self._extract_table_from_h5("blocks", year, table_type="raw")

            if raw_households is None:
                logger.warning("Cannot generate land_use: households table not found")
                return None

            # Call ActivitySim preprocessor function to generate land_use
            logger.info(
                "Calling ActivitySim preprocessor to generate land_use table..."
            )

            (
                raw_blocks,
                raw_persons,
                raw_households,
                raw_jobs,
                num_reassigned,
                blocks_to_taz_mapping_updated,
            ) = process_raw_h5_files(
                raw_blocks,
                raw_persons,
                raw_households,
                raw_jobs,
                self.settings,
                year,
                asim_zone_id_col,
            )
            land_use_df = _create_land_use_table(
                self.settings,
                zones,
                raw_households,
                raw_persons,
                raw_jobs,
                raw_blocks,
            )
            return land_use_df

        except Exception as e:
            logger.error(f"Failed to generate land_use table: {e}")
            return None

    def extract_all_data(
        self,
        year: Optional[int] = None,
        save_csv: bool = False,
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Extract both raw UrbanSim data and processed ActivitySim data from H5 file.

        This implements the dual storage strategy to preserve both original UrbanSim
        data and expensive ActivitySim processing results.

        Args:
            year: Optional year to extract (for time-series data)
            save_csv: Whether to save extracted data as CSV files
            output_dir: Directory to save CSV files (if save_csv=True)

        Returns:
            Tuple of (raw_data_dict, processed_data_dict)
        """
        logger.info(f"🔍 Starting dual extraction from {self.h5_path}")
        logger.info("📊 Extracting raw UrbanSim data...")

        # Extract raw UrbanSim data
        raw_data = self.extract_raw_urbansim_data(year, save_csv, output_dir)

        logger.info("⚙️ Extracting processed ActivitySim data...")

        # Extract processed ActivitySim data
        processed_data = self.extract_activitysim_inputs(year, save_csv, output_dir)

        logger.info(f"✅ Dual extraction complete:")
        logger.info(f"   - Raw UrbanSim tables: {len(raw_data)}")
        logger.info(f"   - Processed ActivitySim tables: {len(processed_data)}")

        return raw_data, processed_data

    def _upload_table_worker(self, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Worker function for parallel table uploads.

        Args:
            table_info: Dictionary with table upload information

        Returns:
            Dictionary with upload results
        """
        table_name = table_info["table_name"]
        table_type = table_info["table_type"]  # 'raw' or 'processed'
        df = table_info["df"]
        file_record = table_info["file_record"]

        try:
            if table_type == "raw":
                logger.info(
                    f"  🏗️ Uploading raw {table_name} data ({len(df)} records)..."
                )
                success = self.db_manager.store_urbansim_raw_data(
                    table_name=table_name,
                    df=df,
                    file_record_id=file_record.unique_id,
                    run_id=self.run_id,
                    openlineage_id=file_record.openlineage_id,
                )
            else:  # processed
                logger.info(
                    f"  📊 Uploading processed {table_name} data ({len(df)} records)..."
                )
                success = self.db_manager.store_activitysim_data(
                    table_name=table_name,
                    df=df,
                    file_record_id=file_record.unique_id,
                    run_id=self.run_id,
                    openlineage_id=file_record.openlineage_id,
                )

            return {
                "table_name": table_name,
                "table_type": table_type,
                "success": success,
                "record_count": len(df),
            }

        except Exception as e:
            logger.error(f"Failed to upload {table_type} {table_name}: {e}")
            return {
                "table_name": table_name,
                "table_type": table_type,
                "success": False,
                "error": str(e),
                "record_count": len(df),
            }

    def upload_to_database(self) -> bool:
        """
        Upload extracted data to database.

        This method uploads both raw UrbanSim data and processed ActivitySim data
        along with metadata (file records) to implement dual storage strategy.

        Returns:
            bool: True if upload successful
        """
        if not self.extracted_tables and not self.raw_tables:
            logger.error(
                "No extracted tables to upload. Run extract_all_data() or extract methods first."
            )
            return False

        try:
            # Create database manager
            self.db_manager = create_database_manager(self.settings)
            if not self.db_manager:
                logger.error("Failed to create database manager")
                return False

            # Create file records for each extracted table (both raw and processed)
            file_records = {}

            # File records for processed ActivitySim tables
            for table_name, df in self.extracted_tables.items():
                file_record = self._create_file_record(f"{table_name}_processed", df)
                file_record.metadata["data_type"] = "processed_activitysim"
                file_records[file_record.unique_id] = file_record

            # File records for raw UrbanSim tables
            for table_name, df in self.raw_tables.items():
                file_record = self._create_file_record(f"{table_name}_raw", df)
                file_record.metadata["data_type"] = "raw_urbansim"
                file_records[file_record.unique_id] = file_record

            # Create a synthetic run info for this extraction
            run_info = PilatesRunInfo(
                run_id=self.run_id,
                created_at=datetime.now().isoformat(),
                models_used=["activitysim"],
                code_version="h5_extraction_utility",
                hostname=os.uname().nodename if hasattr(os, "uname") else "unknown",
                file_records=file_records,
                repo_records={},
                model_runs={},
                config_snapshot={
                    "snapshot_id": str(uuid.uuid4()),
                    "created_timestamp": datetime.now().isoformat(),
                    "extraction_source": self.h5_path,
                    "extraction_method": "h5_to_database_utility_dual_storage",
                    "extracted_activitysim_tables": list(self.extracted_tables.keys()),
                    "extracted_raw_tables": list(self.raw_tables.keys()),
                    "total_tables": len(self.extracted_tables) + len(self.raw_tables),
                },
            )

            # Upload to database
            with self.db_manager:
                # First upload the run metadata
                metadata_success = self.db_manager.upload_run_data(run_info)
                if not metadata_success:
                    logger.error("Failed to upload run metadata")
                    return False

                # Upload both raw UrbanSim data and processed ActivitySim data in parallel
                data_upload_success = True

                # Prepare upload tasks for parallel execution
                upload_tasks = []

                # Add processed ActivitySim data upload tasks
                for table_name, df in self.extracted_tables.items():
                    file_record = None
                    for record in file_records.values():
                        if (
                            record.metadata.get("table_name")
                            == f"{table_name}_processed"
                            and record.metadata.get("data_type")
                            == "processed_activitysim"
                        ):
                            file_record = record
                            break

                    if file_record:
                        upload_tasks.append(
                            {
                                "table_name": table_name,
                                "table_type": "processed",
                                "df": df,
                                "file_record": file_record,
                            }
                        )
                    else:
                        logger.error(
                            f"Could not find file record for processed table {table_name}"
                        )
                        data_upload_success = False

                # Add raw UrbanSim data upload tasks
                for table_name, df in self.raw_tables.items():
                    file_record = None
                    for record in file_records.values():
                        if (
                            record.metadata.get("table_name") == f"{table_name}_raw"
                            and record.metadata.get("data_type") == "raw_urbansim"
                        ):
                            file_record = record
                            break

                    if file_record:
                        upload_tasks.append(
                            {
                                "table_name": table_name,
                                "table_type": "raw",
                                "df": df,
                                "file_record": file_record,
                            }
                        )
                    else:
                        logger.error(
                            f"Could not find file record for raw table {table_name}"
                        )
                        data_upload_success = False

                # Execute uploads (parallel or sequential based on settings)
                if upload_tasks:
                    if self.parallel_uploads and len(upload_tasks) > 1:
                        logger.info(
                            f"📤 Uploading {len(upload_tasks)} tables in parallel (max workers: {self.max_workers})..."
                        )
                        max_workers = min(len(upload_tasks), self.max_workers)

                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            # Submit all upload tasks
                            future_to_task = {
                                executor.submit(self._upload_table_worker, task): task
                                for task in upload_tasks
                            }

                            # Process completed uploads
                            for future in as_completed(future_to_task):
                                result = future.result()

                                if result["success"]:
                                    logger.info(
                                        f"  ✅ Successfully stored {result['table_type']} {result['table_name']} data ({result['record_count']} records)"
                                    )
                                else:
                                    logger.error(
                                        f"  ❌ Failed to store {result['table_type']} {result['table_name']} data"
                                    )
                                    if "error" in result:
                                        logger.error(f"     Error: {result['error']}")
                                    data_upload_success = False
                    else:
                        # Sequential upload (original behavior)
                        logger.info(
                            f"📤 Uploading {len(upload_tasks)} tables sequentially..."
                        )
                        for task in upload_tasks:
                            result = self._upload_table_worker(task)

                            if result["success"]:
                                logger.info(
                                    f"  ✅ Successfully stored {result['table_type']} {result['table_name']} data ({result['record_count']} records)"
                                )
                            else:
                                logger.error(
                                    f"  ❌ Failed to store {result['table_type']} {result['table_name']} data"
                                )
                                if "error" in result:
                                    logger.error(f"     Error: {result['error']}")
                                data_upload_success = False

                if metadata_success and data_upload_success:
                    logger.info(
                        f"🎉 Successfully uploaded ALL data to database (dual storage)!"
                    )
                    logger.info(f"Run ID: {self.run_id}")

                    # Log OpenLineage IDs for easy access
                    logger.info("\n📋 OpenLineage IDs for extracted tables:")
                    logger.info("📊 Processed ActivitySim tables:")
                    for table_name in self.extracted_tables.keys():
                        for record in file_records.values():
                            if (
                                record.metadata.get("table_name")
                                == f"{table_name}_processed"
                                and record.metadata.get("data_type")
                                == "processed_activitysim"
                            ):
                                logger.info(
                                    f"  {table_name}_processed: {record.openlineage_id}"
                                )
                                break

                    logger.info("🏗️ Raw UrbanSim tables:")
                    for table_name in self.raw_tables.keys():
                        for record in file_records.values():
                            if (
                                record.metadata.get("table_name") == f"{table_name}_raw"
                                and record.metadata.get("data_type") == "raw_urbansim"
                            ):
                                logger.info(
                                    f"  {table_name}_raw: {record.openlineage_id}"
                                )
                                break

                    logger.info("\n📋 Next Steps:")
                    logger.info("1. Configure ActivitySim to use database input mode")
                    logger.info(
                        "2. Set land_use_model: 'off' and activitysim_database.enabled: true"
                    )
                    logger.info(
                        "3. Run PILATES - ActivitySim will query processed data from database"
                    )
                    logger.info(
                        "4. Optional: Use raw data for reprocessing or debugging"
                    )

                    return True
                else:
                    logger.error("Upload partially failed - some data may be missing")
                    return False

        except Exception as e:
            logger.error(f"Failed to upload extracted data: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract ActivitySim inputs from UrbanSim H5 file and upload to database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and upload both raw and processed data (dual storage - recommended)
  python h5_to_database.py --h5-file /path/to/urbansim_data.h5 --settings settings.yaml
  
  # Extract only raw UrbanSim data
  python h5_to_database.py --h5-file /path/to/urbansim_data.h5 --settings settings.yaml --raw-only
  
  # Extract only processed ActivitySim data
  python h5_to_database.py --h5-file /path/to/urbansim_data.h5 --settings settings.yaml --processed-only
  
  # Extract for specific year and save CSV files
  python h5_to_database.py --h5-file /path/to/urbansim_data.h5 --settings settings.yaml --year 2020 --save-csv --output-dir ./extracted_data
  
  # Use parallel uploads with custom worker count for faster processing
  python h5_to_database.py --h5-file /path/to/urbansim_data.h5 --settings settings.yaml --max-workers 8
  
  # Just analyze H5 structure without uploading
  python h5_to_database.py --h5-file /path/to/urbansim_data.h5 --settings settings.yaml --analyze-only
        """,
    )

    parser.add_argument("--h5-file", required=True, help="Path to UrbanSim H5 file")
    parser.add_argument(
        "--settings", required=True, help="Path to PILATES settings YAML file"
    )
    parser.add_argument(
        "--year", type=int, help="Year to extract (for time-series data)"
    )
    parser.add_argument("--run-id", help="Custom run ID for this extraction")
    parser.add_argument(
        "--save-csv", action="store_true", help="Save extracted data as CSV files"
    )
    parser.add_argument("--output-dir", help="Directory to save CSV files")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze H5 structure, do not upload",
    )
    parser.add_argument(
        "--raw-only", action="store_true", help="Extract only raw UrbanSim data"
    )
    parser.add_argument(
        "--processed-only",
        action="store_true",
        help="Extract only processed ActivitySim data",
    )
    parser.add_argument(
        "--parallel-uploads",
        action="store_true",
        default=True,
        help="Enable parallel table uploads (default: True)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel upload workers (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Validate inputs
    if not os.path.exists(args.h5_file):
        logger.error(f"H5 file not found: {args.h5_file}")
        return 1

    if not os.path.exists(args.settings):
        logger.error(f"Settings file not found: {args.settings}")
        return 1

    # Load settings
    try:
        with open(args.settings, "r") as f:
            settings = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return 1

    try:
        # Create extractor
        extractor = H5ActivitySimExtractor(
            args.h5_file,
            settings,
            args.run_id,
            parallel_uploads=args.parallel_uploads,
            max_workers=args.max_workers,
        )

        if args.analyze_only:
            # Just analyze structure
            table_info = extractor._get_h5_table_info()
            print(f"\nH5 file structure ({len(table_info)} datasets):")
            for name, info in sorted(table_info.items()):
                size_mb = info["size_mb"]
                print(f"  {name}: {info['shape']} ({size_mb:.1f} MB)")
            return 0

        # Extract data based on command line options
        if args.raw_only:
            logger.info("🏗️ Extracting raw UrbanSim data only...")
            raw_data = extractor.extract_raw_urbansim_data(
                year=args.year, save_csv=args.save_csv, output_dir=args.output_dir
            )
            processed_data = {}
        elif args.processed_only:
            logger.info("📊 Extracting processed ActivitySim data only...")
            processed_data = extractor.extract_activitysim_inputs(
                year=args.year, save_csv=args.save_csv, output_dir=args.output_dir
            )
            raw_data = {}
        else:
            logger.info("🔍 Using dual storage approach (recommended)...")
            raw_data, processed_data = extractor.extract_all_data(
                year=args.year, save_csv=args.save_csv, output_dir=args.output_dir
            )

        if not raw_data and not processed_data:
            logger.warning("No data tables found in H5 file")
            return 1

        logger.info(f"\n📋 Extraction Summary:")
        logger.info(f"  Raw UrbanSim tables: {len(raw_data)}")
        logger.info(f"  Processed ActivitySim tables: {len(processed_data)}")
        logger.info(f"  Total tables extracted: {len(raw_data) + len(processed_data)}")

        # Upload to database
        success = extractor.upload_to_database()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Extraction cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
