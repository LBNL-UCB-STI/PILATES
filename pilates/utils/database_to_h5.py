#!/usr/bin/env python3
"""
Reconstruct UrbanSim H5 datastore from database records.

This utility queries the database for UrbanSim and ActivitySim data and reconstructs
an H5 file that can be used as input to UrbanSim. This is complex because H5 files
can contain arbitrary data structures that may not all be captured in the database.
"""

import argparse
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import h5py
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pilates.utils.database_upload import create_database_manager

logger = logging.getLogger(__name__)


class DatabaseToH5Reconstructor:
    """
    Reconstructs UrbanSim H5 files from database records.

    NOTE: This is a complex process because H5 files can contain arbitrary
    data structures. This utility handles common cases but may need extension
    for specific UrbanSim configurations.
    """

    # Core tables expected in UrbanSim H5 files
    CORE_URBANSIM_TABLES = [
        "buildings",
        "parcels",
        "households",
        "persons",
        "jobs",
        "establishments",
        "zones",
        "accessibility",
    ]

    # ActivitySim tables that might be needed
    ACTIVITYSIM_TABLES = [
        "land_use",
        "maz",
        "taz",
        "regional_demographic_controls",
        "employment_controls",
        "household_controls",
    ]

    # Optional tables that enhance the model
    OPTIONAL_TABLES = ["poi", "transit_stops", "network", "travel_data"]

    def __init__(self, settings: Dict[str, Any], output_h5_path: str):
        """
        Initialize H5 reconstructor.

        Args:
            settings: PILATES settings for database configuration
            output_h5_path: Path where reconstructed H5 file will be saved
        """
        self.settings = settings
        self.output_h5_path = os.path.abspath(output_h5_path)
        self.db_manager = None

        # Track what data we've found and reconstructed
        self.available_tables = {}
        self.reconstructed_tables = {}
        self.missing_tables = set()

    def _query_available_activitysim_data(self) -> Dict[str, pd.DataFrame]:
        """
        Query database for available ActivitySim input data.

        Returns:
            Dictionary mapping table names to DataFrames
        """
        available_data = {}

        try:
            conn = self.db_manager._get_connection()

            # Query for ActivitySim input tables
            activitysim_records = conn.execute(
                """
                SELECT unique_id, openlineage_id, short_name, metadata, schema
                FROM file_records 
                WHERE 'activitysim' = ANY(models)
                AND short_name LIKE 'activitysim_input_%'
                ORDER BY created_at DESC
            """
            ).fetchall()

            logger.info(
                f"Found {len(activitysim_records)} ActivitySim input records in database"
            )

            for record in activitysim_records:
                # Extract table name from short_name
                short_name = record[2]  # short_name column
                if short_name.startswith("activitysim_input_"):
                    table_name = short_name.replace("activitysim_input_", "")

                    # TODO: Actually retrieve the data from storage
                    # For now, we just track that it exists
                    # In a real implementation, you'd need to:
                    # 1. Query the actual data storage (could be files, could be in database)
                    # 2. Load the data into a DataFrame
                    # 3. Handle different storage formats

                    logger.info(f"Found ActivitySim input table: {table_name}")
                    available_data[table_name] = None  # Placeholder

                    # TODO: Implement actual data retrieval based on your storage strategy
                    # This might involve:
                    # - Reading from CSV files referenced in file_path
                    # - Querying data from separate data tables
                    # - Reading from cloud storage

        except Exception as e:
            logger.error(f"Failed to query ActivitySim data: {e}")

        return available_data

    def _query_urbansim_output_data(
        self, config_hash: Optional[str] = None, year: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Query database for UrbanSim output data from previous runs.

        Args:
            config_hash: Optional configuration hash to match
            year: Optional year to filter data

        Returns:
            Dictionary mapping table names to DataFrames
        """
        urbansim_data = {}

        try:
            conn = self.db_manager._get_connection()

            # Build query conditions
            conditions = ["'urbansim' = ANY(models)"]
            params = []

            if config_hash:
                conditions.append(
                    "run_id IN (SELECT run_id FROM runs WHERE config_content_hash = ?)"
                )
                params.append(config_hash)

            if year:
                conditions.append("year = ?")
                params.append(year)

            where_clause = " AND ".join(conditions)

            urbansim_records = conn.execute(
                f"""
                SELECT unique_id, openlineage_id, short_name, metadata, year, file_path
                FROM file_records 
                WHERE {where_clause}
                ORDER BY created_at DESC
            """,
                params,
            ).fetchall()

            logger.info(f"Found {len(urbansim_records)} UrbanSim records in database")

            for record in urbansim_records:
                # Extract useful information
                short_name = record[2]
                metadata = record[3] if record[3] else "{}"
                file_path = record[5]

                # TODO: Map short_name to H5 table structure
                # This requires understanding your UrbanSim output naming conventions
                # Examples:
                # - "buildings_2020" -> buildings table for year 2020
                # - "households_urbansim_out" -> households table

                logger.info(f"Found UrbanSim record: {short_name}")

                # TODO: Implement data loading based on file_path or other storage method
                urbansim_data[short_name] = None  # Placeholder

        except Exception as e:
            logger.error(f"Failed to query UrbanSim data: {e}")

        return urbansim_data

    def _create_base_h5_structure(self, h5_file) -> None:
        """
        Create the basic H5 file structure expected by UrbanSim.

        Args:
            h5_file: Open h5py File object
        """
        # TODO: This depends heavily on your specific UrbanSim configuration
        # Common structures include:

        # Create base groups
        if "base" not in h5_file:
            base_group = h5_file.create_group("base")
            logger.info("Created 'base' group for baseline data")

        # Create year-specific groups if needed
        # for year in range(start_year, end_year + 1):
        #     if str(year) not in h5_file:
        #         year_group = h5_file.create_group(str(year))
        #         logger.info(f"Created '{year}' group")

        # TODO: Add metadata attributes that UrbanSim expects
        # h5_file.attrs['created_by'] = 'PILATES database reconstruction'
        # h5_file.attrs['created_at'] = datetime.now().isoformat()

    def _write_table_to_h5(
        self, h5_file, table_name: str, df: pd.DataFrame, group_path: str = "/base"
    ) -> None:
        """
        Write a DataFrame to the H5 file in the correct location.

        Args:
            h5_file: Open h5py File object
            table_name: Name of the table
            df: DataFrame to write
            group_path: H5 group path to write to
        """
        try:
            full_path = f"{group_path}/{table_name}"

            # Remove existing table if it exists
            if full_path in h5_file:
                del h5_file[full_path]

            # Write DataFrame using pandas HDFStore format
            # Note: This creates a temporary HDFStore to write properly formatted data
            with pd.HDFStore(self.output_h5_path, mode="a") as store:
                store.put(full_path, df, format="table", data_columns=True)

            logger.info(f"Wrote table {table_name} to {full_path} ({len(df)} records)")

        except Exception as e:
            logger.error(f"Failed to write table {table_name}: {e}")

    def _generate_synthetic_data_if_missing(
        self, table_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Generate synthetic/stub data for missing required tables.

        Args:
            table_name: Name of missing table

        Returns:
            DataFrame with minimal synthetic data or None
        """
        logger.warning(f"Generating synthetic data for missing table: {table_name}")

        # TODO: Create minimal synthetic data for each table type
        # This is highly dependent on your UrbanSim configuration

        synthetic_data = {
            "buildings": pd.DataFrame(
                {
                    "building_id": [1],
                    "parcel_id": [1],
                    "building_type_id": [1],
                    "sqft": [1000],
                    "year_built": [2000],
                    "note": ["synthetic_placeholder"],
                }
            ),
            "parcels": pd.DataFrame(
                {
                    "parcel_id": [1],
                    "zone_id": [1],
                    "land_value": [100000],
                    "total_sqft": [5000],
                    "note": ["synthetic_placeholder"],
                }
            ),
            "households": pd.DataFrame(
                {
                    "household_id": [1],
                    "building_id": [1],
                    "persons": [2],
                    "income": [50000],
                    "note": ["synthetic_placeholder"],
                }
            ),
            "persons": pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "household_id": [1, 1],
                    "age": [35, 32],
                    "worker": [1, 1],
                    "note": ["synthetic_placeholder", "synthetic_placeholder"],
                }
            ),
            "jobs": pd.DataFrame(
                {
                    "job_id": [1],
                    "building_id": [1],
                    "sector_id": [1],
                    "note": ["synthetic_placeholder"],
                }
            ),
            "zones": pd.DataFrame(
                {
                    "zone_id": [1],
                    "taz": [1],
                    "county": [1],
                    "note": ["synthetic_placeholder"],
                }
            ),
        }

        if table_name in synthetic_data:
            logger.warning(
                f"Using synthetic data for {table_name} - "
                f"REPLACE WITH REAL DATA BEFORE PRODUCTION USE"
            )
            return synthetic_data[table_name]

        return None

    def reconstruct_h5_file(
        self,
        config_hash: Optional[str] = None,
        year: Optional[int] = None,
        include_synthetic: bool = False,
    ) -> bool:
        """
        Reconstruct H5 file from database records.

        Args:
            config_hash: Optional configuration hash to match
            year: Optional year to reconstruct for
            include_synthetic: Whether to generate synthetic data for missing tables

        Returns:
            bool: True if reconstruction successful
        """
        try:
            # Create database manager
            self.db_manager = create_database_manager(self.settings)
            if not self.db_manager:
                logger.error("Failed to create database manager")
                return False

            logger.info(f"Reconstructing H5 file: {self.output_h5_path}")

            with self.db_manager:
                # Query available data
                activitysim_data = self._query_available_activitysim_data()
                urbansim_data = self._query_urbansim_output_data(config_hash, year)

                # Combine available data
                all_available_data = {**activitysim_data, **urbansim_data}

                logger.info(f"Found data for {len(all_available_data)} tables")

                # Create H5 file
                os.makedirs(os.path.dirname(self.output_h5_path), exist_ok=True)

                with h5py.File(self.output_h5_path, "w") as h5_file:
                    # Create base structure
                    self._create_base_h5_structure(h5_file)

                    # Track what we've written and what's missing
                    written_tables = set()

                    # Write available data
                    for table_name, df in all_available_data.items():
                        if df is not None:
                            self._write_table_to_h5(h5_file, table_name, df)
                            written_tables.add(table_name)
                        else:
                            logger.warning(
                                f"Data for table {table_name} is not available (placeholder)"
                            )

                    # Check for missing core tables
                    all_expected_tables = set(
                        self.CORE_URBANSIM_TABLES + self.ACTIVITYSIM_TABLES
                    )
                    missing_tables = all_expected_tables - written_tables

                    if missing_tables:
                        logger.warning(f"Missing core tables: {missing_tables}")
                        self.missing_tables = missing_tables

                        if include_synthetic:
                            for table_name in missing_tables:
                                synthetic_df = self._generate_synthetic_data_if_missing(
                                    table_name
                                )
                                if synthetic_df is not None:
                                    self._write_table_to_h5(
                                        h5_file, table_name, synthetic_df
                                    )
                                    written_tables.add(table_name)

                    # Add reconstruction metadata
                    h5_file.attrs["reconstructed_by"] = "PILATES database_to_h5 utility"
                    h5_file.attrs["reconstructed_at"] = datetime.now().isoformat()
                    h5_file.attrs["source_config_hash"] = config_hash or "unspecified"
                    h5_file.attrs["source_year"] = year or "unspecified"
                    h5_file.attrs["written_tables"] = list(written_tables)
                    h5_file.attrs["missing_tables"] = list(missing_tables)

                logger.info(
                    f"Successfully reconstructed H5 file with {len(written_tables)} tables"
                )

                if missing_tables:
                    logger.warning(
                        "IMPORTANT: Some tables are missing or contain synthetic data."
                    )
                    logger.warning(
                        "Review the H5 file before using in production UrbanSim runs."
                    )
                    logger.warning(f"Missing tables: {missing_tables}")

                return True

        except Exception as e:
            logger.error(f"Failed to reconstruct H5 file: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct UrbanSim H5 file from database records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reconstruct H5 file from latest data
  python database_to_h5.py --output urbansim_data.h5 --settings settings.yaml
  
  # Reconstruct for specific configuration
  python database_to_h5.py --output urbansim_data.h5 --settings settings.yaml --config-hash abc123
  
  # Reconstruct for specific year with synthetic data for missing tables
  python database_to_h5.py --output urbansim_data.h5 --settings settings.yaml --year 2020 --include-synthetic
  
  # Just analyze what data is available
  python database_to_h5.py --settings settings.yaml --analyze-only
        """,
    )

    parser.add_argument("--output", help="Path for output H5 file")
    parser.add_argument(
        "--settings", required=True, help="Path to PILATES settings YAML file"
    )
    parser.add_argument(
        "--config-hash", help="Configuration hash to match for data selection"
    )
    parser.add_argument("--year", type=int, help="Year to reconstruct data for")
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Generate synthetic data for missing required tables",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze available data, do not create H5 file",
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
    if not args.analyze_only and not args.output:
        logger.error("--output is required unless using --analyze-only")
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
        if args.analyze_only:
            # Just analyze available data
            reconstructor = DatabaseToH5Reconstructor(settings, "/tmp/dummy.h5")
            reconstructor.db_manager = create_database_manager(settings)

            if not reconstructor.db_manager:
                logger.error("Failed to create database manager")
                return 1

            with reconstructor.db_manager:
                activitysim_data = reconstructor._query_available_activitysim_data()
                urbansim_data = reconstructor._query_urbansim_output_data(
                    args.config_hash, args.year
                )

                print(
                    f"\nAvailable ActivitySim input tables ({len(activitysim_data)}):"
                )
                for table_name in sorted(activitysim_data.keys()):
                    print(f"  {table_name}")

                print(f"\nAvailable UrbanSim output tables ({len(urbansim_data)}):")
                for table_name in sorted(urbansim_data.keys()):
                    print(f"  {table_name}")

                all_available = set(activitysim_data.keys()) | set(urbansim_data.keys())
                all_expected = set(
                    reconstructor.CORE_URBANSIM_TABLES
                    + reconstructor.ACTIVITYSIM_TABLES
                )
                missing = all_expected - all_available

                if missing:
                    print(f"\nMissing expected tables ({len(missing)}):")
                    for table_name in sorted(missing):
                        print(f"  {table_name}")

            return 0

        # Reconstruct H5 file
        reconstructor = DatabaseToH5Reconstructor(settings, args.output)
        success = reconstructor.reconstruct_h5_file(
            config_hash=args.config_hash,
            year=args.year,
            include_synthetic=args.include_synthetic,
        )

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Reconstruction cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
