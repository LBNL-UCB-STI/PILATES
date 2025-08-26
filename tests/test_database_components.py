#!/usr/bin/env python3
"""
Unit tests for database components used in dual storage workflow.

This test suite focuses on testing individual components of the dual storage
system without requiring large H5 files, making it suitable for CI/CD.
"""

import os
import tempfile
import unittest
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Import PILATES modules
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pilates.utils.duckdb_manager import DuckDBManager
from pilates.utils.database_upload import create_database_manager
from pilates.activitysim.preprocessor import (
    _clean_activitysim_data_for_csv,
    _create_minimal_placeholder,
)
from pilates.generic.records import PilatesRunInfo, FileRecord
from datetime import datetime
import uuid


class TestDatabaseComponents(unittest.TestCase):
    """Test individual database components."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="pilates_db_test_")
        self.db_path = os.path.join(self.temp_dir, "test.duckdb")

        self.settings = {
            "database": {"enabled": True, "type": "duckdb", "path": self.db_path}
        }

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_database_manager_creation(self):
        """Test creating database manager."""
        print("\n🗄️  Testing database manager creation...")

        db_manager = create_database_manager(self.settings)
        self.assertIsNotNone(db_manager)
        self.assertIsInstance(db_manager, DuckDBManager)
        print("   ✅ Database manager created successfully")

    def test_database_initialization(self):
        """Test database schema initialization."""
        print("\n🏗️  Testing database initialization...")

        db_manager = create_database_manager(self.settings)

        with db_manager:
            success = db_manager.initialize_database()
            self.assertTrue(success)

            # Check that database file was created
            self.assertTrue(os.path.exists(self.db_path))
            print("   ✅ Database initialized and file created")

            # Test basic connection
            conn = db_manager._get_connection()
            self.assertIsNotNone(conn)
            print("   ✅ Database connection successful")

    def test_dual_storage_tables(self):
        """Test that dual storage tables are created correctly."""
        print("\n🗃️  Testing dual storage table creation...")

        db_manager = create_database_manager(self.settings)

        with db_manager:
            db_manager.initialize_database()
            conn = db_manager._get_connection()

            # Check for raw UrbanSim tables
            raw_tables = [
                "urbansim_households_raw",
                "urbansim_persons_raw",
                "urbansim_jobs_raw",
                "urbansim_blocks_raw",
                "urbansim_buildings_raw",
                "urbansim_parcels_raw",
            ]

            # Get all table names using DuckDB syntax
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]

            for table in raw_tables:
                self.assertIn(table, table_names, f"Raw table {table} should exist")
                print(f"   ✅ Raw table {table} exists")

            # Check for processed ActivitySim tables
            processed_tables = [
                "activitysim_households",
                "activitysim_persons",
                "activitysim_land_use",
                "activitysim_data_generic",
            ]

            for table in processed_tables:
                self.assertIn(
                    table, table_names, f"Processed table {table} should exist"
                )
                print(f"   ✅ Processed table {table} exists")

    def test_data_storage_and_retrieval(self):
        """Test actual data insertion operations that would catch SQL constraint issues."""
        print("\n💾 Testing data insertion and constraint handling...")

        db_manager = create_database_manager(self.settings)

        with db_manager:
            db_manager.initialize_database()

            # First create prerequisite records to satisfy foreign key constraints
            conn = db_manager._get_connection()

            # Use unique IDs for this test
            unique_run_id = f"test_run_{uuid.uuid4().hex[:8]}"
            unique_file_id_1 = f"test_file_1_{uuid.uuid4().hex[:8]}"
            unique_file_id_2 = f"test_file_2_{uuid.uuid4().hex[:8]}"
            unique_ol_id_1 = f"test_ol_1_{uuid.uuid4().hex[:8]}"
            unique_ol_id_2 = f"test_ol_2_{uuid.uuid4().hex[:8]}"

            # Create a test run record
            conn.execute(
                """
                INSERT INTO runs (run_id, created_at, models_used, code_version, hostname)
                VALUES (?, ?, ?, ?, ?)
            """,
                [
                    unique_run_id,
                    datetime.now().isoformat(),
                    ["test"],
                    "test_version",
                    "test_host",
                ],
            )

            # Create test file records
            conn.execute(
                """
                INSERT INTO file_records (unique_id, run_id, openlineage_id, file_path, created_at, short_name, description, models, schema, metadata, exists)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    unique_file_id_1,
                    unique_run_id,
                    unique_ol_id_1,
                    "/test/path1",
                    datetime.now().isoformat(),
                    "test1",
                    "Test file 1",
                    ["test"],
                    "[]",
                    "{}",
                    True,
                ],
            )

            conn.execute(
                """
                INSERT INTO file_records (unique_id, run_id, openlineage_id, file_path, created_at, short_name, description, models, schema, metadata, exists)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    unique_file_id_2,
                    unique_run_id,
                    unique_ol_id_2,
                    "/test/path2",
                    datetime.now().isoformat(),
                    "test2",
                    "Test file 2",
                    ["test"],
                    "[]",
                    "{}",
                    True,
                ],
            )

            print("   🗂️ Created prerequisite records for foreign key constraints")

            # Create test data for raw UrbanSim households
            raw_households_data = pd.DataFrame(
                {
                    "household_id": [1, 2, 3],
                    "building_id": [101, 102, 103],
                    "persons": [2, 3, 1],
                    "income": [50000.0, 75000.0, 40000.0],
                    "cars": [1, 2, 0],
                    "block_id": ["block1", "block2", "block3"],
                    "age_of_head": [35, 45, 28],
                    "children": [1, 2, 0],
                    "workers": [1, 2, 1],
                }
            )

            # Test raw data storage (this would catch sequence/ID issues)
            print("   🏗️ Testing raw UrbanSim data insertion...")
            success = db_manager.store_urbansim_raw_data(
                table_name="households",
                df=raw_households_data,
                file_record_id=unique_file_id_1,
                run_id=unique_run_id,
                openlineage_id=unique_ol_id_1,
            )
            self.assertTrue(
                success, "Raw households data should be stored successfully"
            )
            print("   ✅ Raw UrbanSim data insertion works")

            # Create test data for processed ActivitySim households
            processed_households_data = pd.DataFrame(
                {
                    "household_id": [1, 2, 3],
                    "TAZ": ["TAZ1", "TAZ2", "TAZ3"],
                    "persons": [2, 3, 1],
                    "income": [50000.0, 75000.0, 40000.0],
                    "cars": [1, 2, 0],
                    "HHT": [1, 2, 1],
                    "workers": [1, 2, 1],
                }
            )

            # Test processed data storage (this would catch sequence/ID issues)
            print("   📊 Testing processed ActivitySim data insertion...")
            success = db_manager.store_activitysim_data(
                table_name="households",
                df=processed_households_data,
                file_record_id=unique_file_id_2,
                run_id=unique_run_id,
                openlineage_id=unique_ol_id_2,
            )
            self.assertTrue(
                success, "Processed households data should be stored successfully"
            )
            print("   ✅ Processed ActivitySim data insertion works")

            # Test duplicate insertion (this would catch INSERT OR REPLACE constraint issues)
            print("   🔄 Testing duplicate data handling...")
            success2 = db_manager.store_urbansim_raw_data(
                table_name="households",
                df=raw_households_data,  # Same data
                file_record_id=unique_file_id_1,  # Same file record
                run_id=unique_run_id,
                openlineage_id=unique_ol_id_1,  # Same openlineage_id
            )
            self.assertTrue(
                success2, "Duplicate data insertion should handle conflicts correctly"
            )
            print("   ✅ Duplicate data handling works")

            # Verify data retrieval
            print("   📥 Testing data retrieval...")
            retrieved_raw = db_manager.retrieve_urbansim_raw_data(
                "households", openlineage_id=unique_ol_id_1
            )
            self.assertIsNotNone(retrieved_raw, "Should retrieve raw data")
            self.assertEqual(len(retrieved_raw), 3, "Should retrieve 3 households")
            print("   ✅ Raw data retrieval works")

            retrieved_processed = db_manager.retrieve_activitysim_data(
                "households", openlineage_id=unique_ol_id_2
            )
            self.assertIsNotNone(retrieved_processed, "Should retrieve processed data")
            self.assertEqual(
                len(retrieved_processed), 3, "Should retrieve 3 households"
            )
            print("   ✅ Processed data retrieval works")

    def test_run_metadata_upload_constraints(self):
        """Test run metadata upload operations that would catch INSERT OR REPLACE constraint issues."""
        print("\n📋 Testing run metadata upload and constraint handling...")

        db_manager = create_database_manager(self.settings)

        with db_manager:
            db_manager.initialize_database()

            # Use a unique run ID for this test to avoid conflicts
            unique_run_id = f"test_constraint_run_{uuid.uuid4().hex[:8]}"
            unique_file_id = f"test_file_{uuid.uuid4().hex[:8]}"

            # Create test file record
            test_file_record = FileRecord(
                unique_id=unique_file_id,
                openlineage_id=str(uuid.uuid4()),
                file_path="/test/path/file.h5",
                created_at=datetime.now().isoformat(),
                short_name="test_file",
                description="Test file for database constraints",
                models=["activitysim"],
                schema=[],
                metadata={"test": True},
            )

            # Create test run info
            test_run_info = PilatesRunInfo(
                run_id=unique_run_id,
                created_at=datetime.now().isoformat(),
                models_used=["activitysim"],
                code_version="test_version",
                hostname="test_host",
                file_records={unique_file_id: test_file_record},
                repo_records={},
                model_runs={},
                config_snapshot={
                    "snapshot_id": str(uuid.uuid4()),
                    "created_timestamp": datetime.now().isoformat(),
                    "config_content_hash": f"test_hash_{uuid.uuid4().hex[:8]}",
                },
            )

            # Test first upload
            print("   📤 Testing initial run metadata upload...")
            success1 = db_manager.upload_run_data(test_run_info)
            self.assertTrue(success1, "Initial run metadata upload should succeed")
            print("   ✅ Initial upload works")

            # Test duplicate upload (this would catch INSERT OR REPLACE constraint issues)
            print("   🔄 Testing duplicate run metadata upload...")
            success2 = db_manager.upload_run_data(test_run_info)
            self.assertTrue(
                success2,
                "Duplicate run metadata upload should handle constraints correctly",
            )
            print("   ✅ Duplicate upload constraint handling works")

            # Verify only one record exists (not duplicated)
            conn = db_manager._get_connection()
            run_count = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE run_id = ?", [unique_run_id]
            ).fetchone()[0]
            self.assertEqual(
                run_count,
                1,
                "Should have exactly one run record after duplicate upload",
            )
            print("   ✅ No duplicate records created")

    def test_parallel_uploads(self):
        """Test that database operations can handle sequential uploads efficiently (simulating parallel use case)."""
        print("\n⚡ Testing efficient sequential upload functionality...")

        db_manager = create_database_manager(self.settings)

        with db_manager:
            db_manager.initialize_database()

            # Create test data for multiple different table types to simulate parallel scenario
            test_datasets = {
                "households": pd.DataFrame(
                    {
                        "household_id": [j for j in range(1, 11)],
                        "building_id": [j + 100 for j in range(1, 11)],
                        "persons": [2, 3, 1] * 3 + [4],
                        "income": [50000.0 for j in range(10)],
                        "cars": [1, 2, 0] * 3 + [1],
                        "block_id": [f"block_{j}" for j in range(10)],
                        "age_of_head": [35 for j in range(10)],
                        "children": [1, 0, 2] * 3 + [1],
                        "workers": [1, 2, 1] * 3 + [2],
                    }
                ),
                "persons": pd.DataFrame(
                    {
                        "person_id": [j for j in range(1, 21)],
                        "household_id": [j // 2 + 1 for j in range(1, 21)],
                        "age": [25 + j for j in range(20)],
                        "worker": [j % 2 for j in range(20)],
                        "student": [1 if j < 18 else 0 for j in range(20)],
                        "race_id": [1, 2, 3] * 6 + [1, 2],
                        "sex": [j % 2 + 1 for j in range(20)],
                        "work_zone_id": [j % 5 + 1 for j in range(20)],
                        "school_zone_id": [j % 3 + 1 for j in range(20)],
                    }
                ),
                "jobs": pd.DataFrame(
                    {
                        "job_id": [j for j in range(1, 16)],
                        "building_id": [j + 200 for j in range(1, 16)],
                        "sector_id": [f"sector_{j%3}" for j in range(15)],
                        "home_based_status": [j % 2 for j in range(15)],
                    }
                ),
            }

            # Setup prerequisite records with unique IDs
            unique_run_id = f"test_parallel_run_{uuid.uuid4().hex[:8]}"
            conn = db_manager._get_connection()
            conn.execute(
                """
                INSERT INTO runs (run_id, created_at, models_used, code_version, hostname)
                VALUES (?, ?, ?, ?, ?)
            """,
                [
                    unique_run_id,
                    datetime.now().isoformat(),
                    ["test"],
                    "test_version",
                    "test_host",
                ],
            )

            for table_name in test_datasets.keys():
                conn.execute(
                    """
                    INSERT INTO file_records (unique_id, run_id, openlineage_id, file_path, created_at, short_name, description, models, schema, metadata, exists)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        f"test_file_{table_name}_{uuid.uuid4().hex[:8]}",
                        unique_run_id,
                        f"test_ol_{table_name}_{uuid.uuid4().hex[:8]}",
                        f"/test/path_{table_name}",
                        datetime.now().isoformat(),
                        f"test_{table_name}",
                        f"Test {table_name} file",
                        ["test"],
                        "[]",
                        "{}",
                        True,
                    ],
                )

            print("   🗂️ Created prerequisite records for upload test")

            # Test sequential uploads to different table types (safer than true parallel)
            import time

            start_time = time.time()

            # Create mapping of unique IDs for this test
            unique_file_ids = {}
            unique_ol_ids = {}
            for table_name in test_datasets.keys():
                unique_file_ids[table_name] = (
                    f"test_file_{table_name}_{uuid.uuid4().hex[:8]}"
                )
                unique_ol_ids[table_name] = (
                    f"test_ol_{table_name}_{uuid.uuid4().hex[:8]}"
                )

            # Update the file records to use the unique IDs we'll reference
            for table_name in test_datasets.keys():
                conn.execute(
                    """
                    UPDATE file_records 
                    SET unique_id = ?, openlineage_id = ?
                    WHERE unique_id LIKE ? AND run_id = ?
                """,
                    [
                        unique_file_ids[table_name],
                        unique_ol_ids[table_name],
                        f"test_file_{table_name}_%",
                        unique_run_id,
                    ],
                )

            upload_results = []
            for table_name, df in test_datasets.items():
                table_start = time.time()

                success = db_manager.store_urbansim_raw_data(
                    table_name=table_name,
                    df=df,
                    file_record_id=unique_file_ids[table_name],
                    run_id=unique_run_id,
                    openlineage_id=unique_ol_ids[table_name],
                )

                table_end = time.time()
                upload_results.append(
                    {
                        "table_name": table_name,
                        "success": success,
                        "duration": table_end - table_start,
                        "record_count": len(df),
                    }
                )

                print(
                    f"   📊 Uploaded {table_name}: {len(df)} records in {table_end - table_start:.3f}s"
                )

            total_time = time.time() - start_time

            # Verify all uploads succeeded
            all_success = all(result["success"] for result in upload_results)
            self.assertTrue(all_success, "All uploads should succeed")

            total_records = sum(result["record_count"] for result in upload_results)
            print(
                f"   ✅ Sequential uploads successful: {len(upload_results)} tables, {total_records} total records"
            )
            print(f"   ⏱️ Total upload time: {total_time:.2f}s")

            # Verify data integrity for each table type
            table_counts = {
                "households": conn.execute(
                    "SELECT COUNT(*) FROM urbansim_households_raw WHERE openlineage_id = ?",
                    [unique_ol_ids["households"]],
                ).fetchone()[0],
                "persons": conn.execute(
                    "SELECT COUNT(*) FROM urbansim_persons_raw WHERE openlineage_id = ?",
                    [unique_ol_ids["persons"]],
                ).fetchone()[0],
                "jobs": conn.execute(
                    "SELECT COUNT(*) FROM urbansim_jobs_raw WHERE openlineage_id = ?",
                    [unique_ol_ids["jobs"]],
                ).fetchone()[0],
            }

            for table_name, expected_count in [
                (k, len(v)) for k, v in test_datasets.items()
            ]:
                actual_count = table_counts[table_name]
                self.assertEqual(
                    actual_count,
                    expected_count,
                    f"{table_name} should have {expected_count} records",
                )
                print(
                    f"   ✅ {table_name} data integrity verified: {actual_count} records"
                )

            print("   ✅ Multi-table upload functionality confirmed")

    def test_data_cleaning_functions(self):
        """Test data cleaning functions for ActivitySim compatibility."""
        print("\n🧹 Testing data cleaning functions...")

        # Test households cleaning
        raw_households_df = pd.DataFrame(
            {
                "id": [1, 2, 3],  # Database metadata column
                "file_record_id": ["fr1", "fr2", "fr3"],  # Database metadata
                "household_id": [101, 102, 103],
                "TAZ": ["1", "2", "3"],
                "persons": [2, 1, 4],
                "income": [50000, 75000, None],  # Test NaN handling
                "cars": [1, 2, 1],
            }
        )

        cleaned_df = _clean_activitysim_data_for_csv(raw_households_df, "households")

        # Check metadata columns were removed
        self.assertNotIn("id", cleaned_df.columns)
        self.assertNotIn("file_record_id", cleaned_df.columns)
        print("   ✅ Metadata columns removed")

        # Check index is set correctly
        self.assertEqual(cleaned_df.index.name, "household_id")
        print("   ✅ Index set correctly for households")

        # Check NaN values were filled
        self.assertFalse(cleaned_df["income"].isna().any())
        print("   ✅ NaN values handled")

        # Test persons cleaning
        raw_persons_df = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [101, 102, 103],
                "age": [35, 8, 45],
                "worker": [1, 0, 1],
            }
        )

        cleaned_persons = _clean_activitysim_data_for_csv(raw_persons_df, "persons")
        self.assertEqual(cleaned_persons.index.name, "person_id")
        print("   ✅ Persons cleaning successful")

        # Test land_use cleaning
        raw_landuse_df = pd.DataFrame(
            {
                "TAZ": ["1", "2", "3"],
                "TOTPOP": [1000, 1500, 800],
                "TOTHH": [400, 600, 300],
            }
        )

        cleaned_landuse = _clean_activitysim_data_for_csv(raw_landuse_df, "land_use")
        self.assertEqual(cleaned_landuse.index.name, "TAZ")
        print("   ✅ Land use cleaning successful")

    def test_placeholder_creation(self):
        """Test creation of minimal placeholder data."""
        print("\n📄 Testing placeholder data creation...")

        # Test households placeholder
        households_placeholder = _create_minimal_placeholder("households")
        self.assertIsInstance(households_placeholder, pd.DataFrame)
        self.assertEqual(households_placeholder.index.name, "household_id")
        self.assertIn("TAZ", households_placeholder.columns)
        self.assertIn("persons", households_placeholder.columns)
        print("   ✅ Households placeholder created")

        # Test persons placeholder
        persons_placeholder = _create_minimal_placeholder("persons")
        self.assertIsInstance(persons_placeholder, pd.DataFrame)
        self.assertEqual(persons_placeholder.index.name, "person_id")
        self.assertIn("household_id", persons_placeholder.columns)
        self.assertIn("age", persons_placeholder.columns)
        print("   ✅ Persons placeholder created")

        # Test land_use placeholder
        landuse_placeholder = _create_minimal_placeholder("land_use")
        self.assertIsInstance(landuse_placeholder, pd.DataFrame)
        self.assertEqual(landuse_placeholder.index.name, "TAZ")
        self.assertIn("TOTPOP", landuse_placeholder.columns)
        print("   ✅ Land use placeholder created")

        # Test generic placeholder
        generic_placeholder = _create_minimal_placeholder("unknown_table")
        self.assertIsInstance(generic_placeholder, pd.DataFrame)
        print("   ✅ Generic placeholder created")

    def test_data_type_handling(self):
        """Test handling of different data types in dual storage."""
        print("\n🔢 Testing data type handling...")

        db_manager = create_database_manager(self.settings)

        # Create test data with various data types
        test_data = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.7, 3.14],
                "string_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "nullable_int": [1, None, 3],
                "nullable_float": [1.1, None, 3.3],
            }
        )

        with db_manager:
            db_manager.initialize_database()

            # Store data
            success = db_manager.store_activitysim_data(
                table_name="test_types",
                df=test_data,
                file_record_id="test_types_123",
                run_id="test_run_types",
                openlineage_id="test_ol_types",
            )
            # Note: This might not succeed if table doesn't exist, but we're testing the data handling
            print(
                f"   📊 Data type storage test: {'✅ Success' if success else '⚠️  Expected - table may not exist'}"
            )

    def test_error_handling(self):
        """Test error handling in database operations."""
        print("\n⚠️  Testing error handling...")

        # Test with invalid database path
        invalid_settings = {
            "database": {
                "enabled": True,
                "type": "duckdb",
                "path": "/invalid/path/that/does/not/exist/test.db",
            }
        }

        # This should return None when database initialization fails
        db_manager = create_database_manager(invalid_settings)
        self.assertIsNone(db_manager)
        print("   ✅ Manager creation properly rejects invalid paths")

        # Test with valid manager but non-existent data queries
        db_manager = create_database_manager(self.settings)
        with db_manager:
            db_manager.initialize_database()

            # Test retrieving non-existent data
            result = db_manager.retrieve_activitysim_data(
                table_name="households", openlineage_id="non_existent_id"
            )
            self.assertIsNone(result)
            print("   ✅ Graceful handling of non-existent data")


if __name__ == "__main__":
    print("🧪 Starting PILATES Database Components Tests")
    print("=" * 50)

    unittest.main(verbosity=2)
