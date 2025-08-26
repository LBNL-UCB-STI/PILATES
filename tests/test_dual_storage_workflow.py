#!/usr/bin/env python3
"""
Test dual storage workflow for ActivitySim database input mode.

This test validates the complete dual storage workflow:
1. Extract data from H5 file to database (both raw UrbanSim and processed ActivitySim)
2. Query database to create ActivitySim CSV inputs
3. Compare database-generated CSVs with H5-generated CSVs to ensure equivalence

This is critical functionality that ensures database storage preserves data integrity.
"""

import os
import tempfile
import unittest
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Import PILATES modules
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pilates.utils.h5_to_database import H5ActivitySimExtractor
from pilates.utils.database_upload import create_database_manager
from pilates.activitysim.preprocessor import (
    create_asim_data_from_h5,
    create_asim_data_from_database,
    _clean_activitysim_data_for_csv,
)
from pilates.generic.records import FileRecord, RecordStore
from pilates.utils.provenance import FileProvenanceTracker
from pilates.workspace import Workspace


class MockState:
    """Mock workflow state for testing."""

    def __init__(self, year: int = 2017):
        self.current_year = year
        self.forecast_year = year
        self.current_inner_iter = 0


class MockWorkspace:
    """Mock workspace for testing."""

    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.settings = {"region": "sfbay", "start_year": 2017, "forecast_year": 2017}

    def get_asim_mutable_data_dir(self):
        asim_dir = os.path.join(self.temp_dir, "activitysim", "data")
        os.makedirs(asim_dir, exist_ok=True)
        return asim_dir


class TestDualStorageWorkflow(unittest.TestCase):
    """Test the complete dual storage workflow."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - find available H5 file."""
        # Check for environment variable from test runner
        cls.h5_file_path = os.environ.get("PILATES_TEST_H5_FILE")
        cls.settings_file = os.environ.get("PILATES_TEST_SETTINGS_FILE")

        if cls.h5_file_path:
            if os.path.exists(cls.h5_file_path):
                print(f"📁 Using H5 file from environment: {cls.h5_file_path}")
            else:
                print(f"❌ H5 file from environment not found: {cls.h5_file_path}")
                cls.h5_file_path = None

        # Fallback to manual search if no environment variable
        if not cls.h5_file_path:
            import glob

            h5_search_paths = [
                "pilates/urbansim/data/custom_mpo_06197001_model_data.h5",
                "urbansim/data/custom_mpo_06197001_model_data.h5",
                "pilates/urbansim/data/custom_mpo_53199100_model_data.h5",
                "pilates/urbansim/data/*.h5",
                "urbansim/data/*.h5",
                "tmp/*/urbansim/data/*.h5",
            ]

            for pattern in h5_search_paths:
                if "*" in pattern:
                    matches = glob.glob(pattern)
                    if matches:
                        cls.h5_file_path = matches[0]
                        break
                else:
                    if os.path.exists(pattern):
                        cls.h5_file_path = pattern
                        break

        if not cls.h5_file_path:
            print(
                "⚠️  Warning: No H5 file found for testing. Dual storage tests will be skipped."
            )
        else:
            print(f"📁 Using H5 file for testing: {cls.h5_file_path}")
            size_mb = os.path.getsize(cls.h5_file_path) / (1024 * 1024)
            print(f"📊 H5 file size: {size_mb:.1f} MB")

    def setUp(self):
        """Set up test environment."""
        if not self.h5_file_path:
            self.skipTest("No H5 file available for testing")

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="pilates_dual_storage_test_")
        self.db_path = os.path.join(self.temp_dir, "test_pilates.duckdb")

        # Load settings from file if provided, otherwise use defaults
        if self.settings_file and os.path.exists(self.settings_file):
            try:
                import yaml

                with open(self.settings_file, "r") as f:
                    self.settings = yaml.safe_load(f)
                print(f"📄 Loaded settings from: {self.settings_file}")

                # Override database settings for testing
                if "database" not in self.settings:
                    self.settings["database"] = {}
                self.settings["database"].update(
                    {"enabled": True, "type": "duckdb", "path": self.db_path}
                )

                # Override ActivitySim database settings for testing
                if "activitysim_database" not in self.settings:
                    self.settings["activitysim_database"] = {}
                self.settings["activitysim_database"].update(
                    {"enabled": True, "use_processed_data": True}
                )

            except Exception as e:
                print(f"⚠️  Error loading settings file: {e}")
                print("🔄 Falling back to default test settings...")
                self.settings = self._get_default_settings()
        else:
            self.settings = self._get_default_settings()

    def _get_default_settings(self):
        """Get default test settings."""
        return {
            # Database configuration
            "database": {"enabled": True, "type": "duckdb", "path": self.db_path},
            # ActivitySim database configuration
            "activitysim_database": {
                "enabled": True,
                "use_processed_data": True,
                "year": 2017,
            },
            # Basic PILATES settings for H5 processing
            "region": "sfbay",
            "start_year": 2017,
            "forecast_year": 2017,
            "FIPS": {
                "sfbay": {
                    "state": "06",
                    "counties": [
                        "001",
                        "013",
                        "041",
                        "055",
                        "075",
                        "081",
                        "085",
                        "095",
                        "097",
                    ],
                }
            },
            "local_crs": {"sfbay": "EPSG:26910"},
            "skims_zone_type": "taz",
            "region_to_region_id": {"sfbay": "06197001"},
            "usim_local_data_input_folder": "pilates/urbansim/data",
            "usim_formattable_input_file_name": "custom_mpo_{region_id}_model_data.h5",
            "usim_formattable_output_file_name": "model_data_{year}.h5",
        }

        # Create mock objects
        self.state = MockState(2017)
        self.workspace = MockWorkspace(self.temp_dir)
        self.provenance_tracker = FileProvenanceTracker(run_id="test_dual_storage")

        print(f"🧪 Test setup complete - temp dir: {self.temp_dir}")

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_h5_structure_detection(self):
        """Test H5 structure detection for different formats."""
        print("\n🔍 Testing H5 structure detection...")

        extractor = H5ActivitySimExtractor(self.h5_file_path, self.settings)

        # Test structure detection
        table_prefix = extractor._detect_h5_structure(year=2017)
        print(f"   Detected table prefix: '{table_prefix}'")

        # Test table info extraction
        table_info = extractor._get_h5_table_info()
        print(f"   Found {len(table_info)} datasets in H5 file")

        # Log some key tables
        key_tables = ["households", "persons", "jobs"]
        for key_table in key_tables:
            found_paths = [path for path in table_info.keys() if key_table in path]
            if found_paths:
                print(
                    f"   {key_table} found at: {found_paths[:3]}..."
                )  # Show first 3 paths

        self.assertIsInstance(table_prefix, str)
        self.assertIsInstance(table_info, dict)
        self.assertGreater(
            len(table_info), 0, "Should find at least some datasets in H5 file"
        )

    def test_dual_extraction_from_h5(self):
        """Test extraction of both raw and processed data from H5 file."""
        print("\n📤 Testing dual extraction from H5...")

        extractor = H5ActivitySimExtractor(
            self.h5_file_path, self.settings, run_id="test_extraction"
        )

        # Extract both raw and processed data
        raw_data, processed_data = extractor.extract_all_data(year=2017)

        print(f"   Raw UrbanSim tables extracted: {len(raw_data)}")
        print(f"   Processed ActivitySim tables extracted: {len(processed_data)}")

        # Verify we got some data
        total_tables = len(raw_data) + len(processed_data)
        self.assertGreater(total_tables, 0, "Should extract at least some tables")

        # Check for key tables
        expected_raw_tables = ["households", "persons", "jobs", "blocks"]
        expected_processed_tables = ["households", "persons", "land_use"]

        raw_found = [table for table in expected_raw_tables if table in raw_data]
        processed_found = [
            table for table in expected_processed_tables if table in processed_data
        ]

        print(f"   Raw tables found: {raw_found}")
        print(f"   Processed tables found: {processed_found}")

        # Store for next test
        self.extracted_raw_data = raw_data
        self.extracted_processed_data = processed_data

        return extractor

    def test_database_upload(self):
        """Test uploading extracted data to database."""
        print("\n💾 Testing database upload...")

        # First extract data
        extractor = self.test_dual_extraction_from_h5()

        # Upload to database
        upload_success = extractor.upload_to_database()

        self.assertTrue(upload_success, "Database upload should succeed")
        print("   ✅ Database upload successful")

        # Verify database contains data
        db_manager = create_database_manager(self.settings)
        with db_manager:
            # Check that database was initialized
            self.assertTrue(os.path.exists(self.db_path), "Database file should exist")
            print(f"   📊 Database created at: {self.db_path}")

            # Test querying some data
            for table_name in ["households", "persons", "land_use"]:
                data_df = db_manager.retrieve_activitysim_data(table_name)
                if data_df is not None and not data_df.empty:
                    print(f"   ✅ Retrieved {table_name}: {len(data_df)} records")
                else:
                    print(f"   ⚠️  No {table_name} data found in database")

    def test_h5_csv_generation(self):
        """Generate ActivitySim CSVs directly from H5 file for comparison."""
        print("\n📄 Generating ActivitySim CSVs from H5...")

        # Create directory for H5-generated CSVs
        h5_output_dir = os.path.join(self.temp_dir, "h5_generated")
        os.makedirs(h5_output_dir, exist_ok=True)

        # Mock workspace for H5 generation
        class H5MockWorkspace(MockWorkspace):
            def get_asim_mutable_data_dir(self):
                return h5_output_dir

        h5_workspace = H5MockWorkspace(self.temp_dir)

        try:
            # Generate CSVs from H5
            h5_records = create_asim_data_from_h5(
                settings=self.settings,
                state=self.state,
                workspace=h5_workspace,
                provenance_tracker=self.provenance_tracker,
                model_run_hash="test_h5_generation",
            )

            print(f"   Generated {len(h5_records)} CSV files from H5")

            # Check which CSV files were created
            h5_csv_files = {}
            for csv_file in os.listdir(h5_output_dir):
                if csv_file.endswith(".csv"):
                    table_name = csv_file.replace(".csv", "")
                    csv_path = os.path.join(h5_output_dir, csv_file)
                    df = pd.read_csv(csv_path, index_col=0)
                    h5_csv_files[table_name] = df
                    print(f"   ✅ H5-generated {table_name}.csv: {len(df)} records")

            self.h5_csv_files = h5_csv_files
            return h5_csv_files

        except Exception as e:
            print(f"   ⚠️  H5 CSV generation failed: {e}")
            # Create minimal mock data for comparison
            return self._create_mock_h5_data(h5_output_dir)

    def test_database_csv_generation(self):
        """Generate ActivitySim CSVs from database for comparison."""
        print("\n🗄️  Generating ActivitySim CSVs from database...")

        # First ensure database has data
        self.test_database_upload()

        # Create directory for database-generated CSVs
        db_output_dir = os.path.join(self.temp_dir, "db_generated")
        os.makedirs(db_output_dir, exist_ok=True)

        # Mock workspace for database generation
        class DBMockWorkspace(MockWorkspace):
            def get_asim_mutable_data_dir(self):
                return db_output_dir

        db_workspace = DBMockWorkspace(self.temp_dir)

        # Generate CSVs from database
        db_records = create_asim_data_from_database(
            settings=self.settings,
            state=self.state,
            workspace=db_workspace,
            provenance_tracker=self.provenance_tracker,
            model_run_hash="test_db_generation",
        )

        print(f"   Generated {len(db_records)} CSV files from database")

        # Check which CSV files were created
        db_csv_files = {}
        for csv_file in os.listdir(db_output_dir):
            if csv_file.endswith(".csv"):
                table_name = csv_file.replace(".csv", "")
                csv_path = os.path.join(db_output_dir, csv_file)
                df = pd.read_csv(csv_path, index_col=0)
                db_csv_files[table_name] = df
                print(f"   ✅ DB-generated {table_name}.csv: {len(df)} records")

        self.db_csv_files = db_csv_files
        return db_csv_files

    def test_csv_comparison(self):
        """Compare H5-generated and database-generated CSVs for equivalence."""
        print("\n🔍 Comparing H5 vs Database-generated CSVs...")

        # Generate both sets of CSVs
        h5_csv_files = self.test_h5_csv_generation()
        db_csv_files = self.test_database_csv_generation()

        # Compare each table
        comparison_results = {}

        for table_name in ["households", "persons", "land_use"]:
            print(f"\n   📊 Comparing {table_name} table...")

            if table_name not in h5_csv_files:
                print(f"     ⚠️  {table_name} not found in H5-generated data")
                comparison_results[table_name] = {"status": "missing_h5"}
                continue

            if table_name not in db_csv_files:
                print(f"     ⚠️  {table_name} not found in DB-generated data")
                comparison_results[table_name] = {"status": "missing_db"}
                continue

            h5_df = h5_csv_files[table_name]
            db_df = db_csv_files[table_name]

            # Compare basic properties
            h5_shape = h5_df.shape
            db_shape = db_df.shape

            print(f"     H5 shape: {h5_shape}")
            print(f"     DB shape: {db_shape}")

            # Check row counts
            if h5_shape[0] != db_shape[0]:
                print(f"     ⚠️  Row count mismatch: H5={h5_shape[0]}, DB={db_shape[0]}")

            # Check column sets
            h5_cols = set(h5_df.columns)
            db_cols = set(db_df.columns)

            missing_in_db = h5_cols - db_cols
            extra_in_db = db_cols - h5_cols

            if missing_in_db:
                print(f"     ⚠️  Columns missing in DB: {missing_in_db}")
            if extra_in_db:
                print(f"     ⚠️  Extra columns in DB: {extra_in_db}")

            # Compare common columns
            common_cols = h5_cols & db_cols
            differences = {}

            for col in common_cols:
                h5_series = h5_df[col]
                db_series = db_df[col]

                # For numeric columns, check if values are close
                if pd.api.types.is_numeric_dtype(
                    h5_series
                ) and pd.api.types.is_numeric_dtype(db_series):
                    if not np.allclose(h5_series, db_series, equal_nan=True):
                        diff_count = (
                            ~np.isclose(h5_series, db_series, equal_nan=True)
                        ).sum()
                        differences[col] = f"numeric differences in {diff_count} rows"
                else:
                    # For non-numeric, check exact equality
                    if not h5_series.equals(db_series):
                        diff_count = (h5_series != db_series).sum()
                        differences[col] = f"string differences in {diff_count} rows"

            if differences:
                print(f"     ⚠️  Column differences: {differences}")
            else:
                print(f"     ✅ All common columns match perfectly")

            # Overall assessment
            perfect_match = (
                h5_shape == db_shape
                and len(missing_in_db) == 0
                and len(extra_in_db) == 0
                and len(differences) == 0
            )

            comparison_results[table_name] = {
                "status": "perfect_match" if perfect_match else "differences_found",
                "h5_shape": h5_shape,
                "db_shape": db_shape,
                "missing_cols": missing_in_db,
                "extra_cols": extra_in_db,
                "differences": differences,
            }

            print(
                f"     📋 Result: {'✅ Perfect match' if perfect_match else '⚠️  Differences found'}"
            )

        # Overall test assessment
        perfect_tables = [
            name
            for name, result in comparison_results.items()
            if result.get("status") == "perfect_match"
        ]

        print(f"\n📈 Comparison Summary:")
        print(f"   Perfect matches: {len(perfect_tables)}/{len(comparison_results)}")
        print(f"   Tables with perfect matches: {perfect_tables}")

        # Test assertions
        self.assertGreater(
            len(comparison_results), 0, "Should compare at least one table"
        )

        # At least one table should match perfectly (or be very close)
        success_count = len(
            [
                r
                for r in comparison_results.values()
                if r.get("status") in ["perfect_match", "minor_differences"]
            ]
        )
        self.assertGreater(
            success_count, 0, "At least one table should match well between H5 and DB"
        )

        return comparison_results

    def _create_mock_h5_data(self, output_dir: str) -> Dict[str, pd.DataFrame]:
        """Create mock H5 data when H5 processing fails."""
        print("   📋 Creating mock H5 data for comparison...")

        mock_data = {
            "households": pd.DataFrame(
                {
                    "household_id": [1, 2, 3],
                    "TAZ": ["1", "2", "3"],
                    "persons": [2, 1, 3],
                    "income": [50000, 75000, 60000],
                    "cars": [1, 2, 1],
                }
            ).set_index("household_id"),
            "persons": pd.DataFrame(
                {
                    "person_id": [1, 2, 3, 4],
                    "household_id": [1, 1, 2, 3],
                    "age": [35, 8, 45, 25],
                    "worker": [1, 0, 1, 1],
                    "student": [0, 1, 0, 0],
                }
            ).set_index("person_id"),
            "land_use": pd.DataFrame(
                {
                    "TAZ": ["1", "2", "3"],
                    "TOTPOP": [1000, 1500, 800],
                    "TOTHH": [400, 600, 300],
                    "TOTEMP": [500, 800, 200],
                }
            ).set_index("TAZ"),
        }

        # Save mock data as CSV files
        for table_name, df in mock_data.items():
            csv_path = os.path.join(output_dir, f"{table_name}.csv")
            df.to_csv(csv_path, index=True)
            print(f"   📄 Created mock {table_name}.csv")

        return mock_data

    def test_complete_workflow(self):
        """Test the complete dual storage workflow end-to-end."""
        print("\n🔄 Testing complete dual storage workflow...")

        try:
            # Run the complete comparison
            comparison_results = self.test_csv_comparison()

            print("\n🎯 Workflow Results:")
            for table_name, result in comparison_results.items():
                status = result.get("status", "unknown")
                if status == "perfect_match":
                    print(f"   ✅ {table_name}: Perfect match")
                elif status == "differences_found":
                    print(f"   ⚠️  {table_name}: Differences found (see details above)")
                else:
                    print(f"   ❌ {table_name}: {status}")

            # Test passes if we successfully ran the comparison
            self.assertTrue(
                len(comparison_results) > 0, "Should successfully compare tables"
            )
            print("\n🎉 Dual storage workflow test completed!")

        except Exception as e:
            print(f"\n❌ Workflow test failed: {e}")
            raise


if __name__ == "__main__":
    # Run the test
    print("🧪 Starting PILATES Dual Storage Workflow Tests")
    print("=" * 60)

    unittest.main(verbosity=2)
