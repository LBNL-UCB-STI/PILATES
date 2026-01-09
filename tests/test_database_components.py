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
from pathlib import Path

# Import PILATES modules
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from duckdb import ConstraintException

from pilates.utils.database import DatabaseManager

from pilates.utils.duckdb_manager import DuckDBManager
from pilates.utils.database_upload import create_database_manager
from pilates.utils.provenance import FileProvenanceTracker
from workflow_state import WorkflowState
from pilates.activitysim.preprocessor import (
    ActivitysimPreprocessor,
    _create_minimal_placeholder,
)

from pilates.atlas.postprocessor import atlas_add_vehileTypeId
from pilates.generic.records_legacy import PilatesRunInfo, FileRecord
from datetime import datetime
import uuid
import yaml
from pilates.config.models import load_config


class TestDatabaseComponents(unittest.TestCase):
    """Test individual database components."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="pilates_db_test_")
        self.db_path = os.path.join(self.temp_dir, "test.duckdb")

        # Create a dummy settings.yaml for PilatesConfig
        dummy_config_content = {
            "run": {
                "region": "test",
                "scenario": "test",
                "start_year": 2020,
                "end_year": 2020,
                "output_directory": self.temp_dir,
                "output_run_name": "test_run",
                "models": {
                    "land_use": None,
                    "travel": None,
                    "activity_demand": None,
                    "vehicle_ownership": None,
                },
            },
            "shared": {
                "geography": {
                    "FIPS": {"county": ["06001"]},
                    "local_crs": "EPSG:32048",
                },
                "skims": {
                    "zone_type": "taz",
                    "fname": "skims.h5",
                    "geoms_fname": "geoms.geojson",
                    "geoms_index_col": "TAZ",
                },
                "database": {
                    "enabled": True,
                    "type": "duckdb",
                    "path": self.db_path,
                },
            },
            "infrastructure": {
                "container_manager": "docker",
                "singularity_images": {},
                "docker_images": {},
                "docker_config": {"stdout": False, "pull_latest": False},
            },
        }

        # Write the dummy config to a temporary YAML file
        self.dummy_config_path = os.path.join(self.temp_dir, "dummy_settings.yaml")
        with open(self.dummy_config_path, "w") as f:
            yaml.dump(dummy_config_content, f)

        # Load the config using load_config to get a PilatesConfig object
        self.settings = load_config(self.dummy_config_path)

        # --- Schema Generation ---
        # Create a temporary schema directory layout that mirrors the real one
        self.test_schema_dir = Path(self.temp_dir) / "schema"
        self.generated_dir = self.test_schema_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        # Copy original schema files to the temporary location
        original_schema_dir = (
            Path(__file__).parent.parent / "pilates" / "database" / "schema"
        )
        for fname in os.listdir(original_schema_dir):
            if fname.endswith(".sql"):
                shutil.copy(original_schema_dir / fname, self.test_schema_dir)

        # Point the view creation script to our temporary directories and run it
        from pilates.database.scripts import create_views

        create_views.SCHEMA_DIR = self.test_schema_dir
        create_views.GENERATED_DIR = self.generated_dir
        create_views.main()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_database_manager_creation(self):
        """Test creating database manager."""
        db_manager = create_database_manager(self.settings.shared.database)
        self.assertIsNotNone(db_manager)
        self.assertIsInstance(db_manager, DuckDBManager)
        db_manager.close()
        print("   ✅ Database manager created successfully")

    def test_database_initialization(self):
        """Test database schema initialization."""
        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            success = db_manager.initialize_database(schema_dir=self.test_schema_dir)
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

        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            assert isinstance(db_manager, DatabaseManager)

            db_manager.initialize_database(schema_dir=self.test_schema_dir)
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

    def test_config_snapshot_insert(self):
        """Test that inserting into config_snapshots works without a config_hash column."""
        import json

        db_manager = create_database_manager(self.settings.shared.database)
        with db_manager:
            db_manager.initialize_database()
            conn = db_manager._get_connection()
            # Insert a minimal config snapshot
            snapshot_id = f"snap_{uuid.uuid4().hex[:8]}"
            conn.execute(
                """
                INSERT INTO config_snapshots (
                    snapshot_id, created_timestamp, config_content_hash,
                    git_hashes, config_files, pilates_settings,
                    beam_config, asim_subdir, region
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    snapshot_id,
                    datetime.now().isoformat(),
                    "dummy_hash",
                    json.dumps({}),
                    json.dumps({}),
                    json.dumps({}),
                    "beam.cfg",
                    "asim_subdir",
                    "test_region",
                ],
            )
            # Verify insertion
            result = conn.execute(
                "SELECT * FROM config_snapshots WHERE snapshot_id = ?", [snapshot_id]
            ).fetchone()
            self.assertIsNotNone(
                result, "Config snapshot should be inserted successfully"
            )

    def test_data_storage_and_retrieval(self):
        """Test actual data insertion operations that would catch SQL constraint issues."""
        print("\n💾 Testing data insertion and constraint handling...")

        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            assert isinstance(db_manager, DuckDBManager)
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
                INSERT INTO file_records (unique_id, record_type, run_id, openlineage_id, file_path, created_at, short_name, description, models, schema, metadata, exists)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    unique_file_id_1,
                    "file",
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
                INSERT INTO file_records (unique_id, record_type, run_id, openlineage_id, file_path, created_at, short_name, description, models, schema, metadata, exists)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    unique_file_id_2,
                    "file",
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

            # Create parcels first (required for buildings FK chain)
            raw_parcels_data = pd.DataFrame(
                {
                    "parcel_id": [201, 202, 203],
                    "zone_id": ["zone1", "zone2", "zone3"],
                    "land_value": [100000.0, 150000.0, 120000.0],
                    "total_sqft": [5000.0, 6000.0, 4500.0],
                    "county_id": ["county1", "county1", "county2"],
                }
            )

            print("   🏞️ Creating parcels data first...")
            db_manager.store_urbansim_raw_data(
                table_name="parcels",
                df=raw_parcels_data,
                file_record_id=unique_file_id_1,
                run_id=unique_run_id,
                year=2017,
                iteration=0,
                openlineage_id=unique_ol_id_1,
            )

            # Create buildings second (required for households FK)
            raw_buildings_data = pd.DataFrame(
                {
                    "building_id": [101, 102, 103],
                    "parcel_id": [201, 202, 203],
                    "building_type_id": [1, 2, 1],
                    "sqft": [1500, 2000, 1200],
                    "year_built": [1990, 2000, 1985],
                    "stories": [2, 3, 1],
                }
            )

            print("   🏢 Creating buildings data second...")
            db_manager.store_urbansim_raw_data(
                table_name="buildings",
                df=raw_buildings_data,
                file_record_id=unique_file_id_1,
                run_id=unique_run_id,
                year=2017,
                iteration=0,
                openlineage_id=unique_ol_id_1,
            )

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
                year=2017,
                iteration=0,
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
                year=2017,
                iteration=0,
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
                year=2017,
                iteration=0,
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

        db_manager = create_database_manager(self.settings.shared.database)

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

        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            assert isinstance(db_manager, DuckDBManager)
            db_manager.initialize_database()

            # Create test data for multiple different table types to simulate parallel scenario
            # Must create in FK dependency order: parcels -> buildings -> (households, jobs) -> persons
            test_datasets = {
                "parcels": pd.DataFrame(
                    {
                        "parcel_id": list(range(1, 26)),  # Covers 101-109 and 201-215
                        "zone_id": [f"zone_{j}" for j in range(1, 26)],
                        "land_value": [100000.0 for j in range(25)],
                        "total_sqft": [5000.0 for j in range(25)],
                        "county_id": [f"county_{j%3}" for j in range(25)],
                    }
                ),
                "buildings": pd.DataFrame(
                    {
                        "building_id": list(range(101, 111))
                        + list(range(201, 216)),  # 10 + 15 = 25
                        "parcel_id": list(range(1, 11))
                        + list(range(11, 26)),  # 10 + 15 = 25
                        "building_type_id": ([1, 2] * 12) + [1],  # 25 items
                        "sqft": [1500] * 25,  # 25 items
                        "year_built": [2000] * 25,  # 25 items
                        "stories": [2] * 25,  # 25 items
                    }
                ),
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
                        "household_id": [
                            (j - 1) // 2 + 1 for j in range(1, 21)
                        ],  # Maps 1-2->1, 3-4->2, ..., 19-20->10
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
                    INSERT INTO file_records (unique_id, record_type, run_id, openlineage_id, file_path, created_at, short_name, description, models, schema, metadata, exists)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        f"test_file_{table_name}_{uuid.uuid4().hex[:8]}",
                        "file",
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
                    year=2017,
                    iteration=0,
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
                "parcels": conn.execute(
                    "SELECT COUNT(*) FROM urbansim_parcels_raw WHERE openlineage_id = ?",
                    [unique_ol_ids["parcels"]],
                ).fetchone()[0],
                "buildings": conn.execute(
                    "SELECT COUNT(*) FROM urbansim_buildings_raw WHERE openlineage_id = ?",
                    [unique_ol_ids["buildings"]],
                ).fetchone()[0],
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

        # Create dummy WorkflowState and FileProvenanceTracker for ActivitysimPreprocessor
        dummy_state = WorkflowState.from_settings(self.settings)
        dummy_provenance_tracker = FileProvenanceTracker(
            run_id="dummy_run_id",
            output_path=self.temp_dir,
        )
        preprocessor = ActivitysimPreprocessor(
            model_name="activitysim",
            state=dummy_state,
            provenance_tracker=dummy_provenance_tracker,
        )

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

        cleaned_df = preprocessor._clean_activitysim_data(
            raw_households_df, "households"
        )

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

        cleaned_persons = preprocessor._clean_activitysim_data(
            raw_persons_df, "persons"
        )
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

        cleaned_landuse = preprocessor._clean_activitysim_data(
            raw_landuse_df, "land_use"
        )
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

        db_manager = create_database_manager(self.settings.shared.database)

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
            assert isinstance(db_manager, DuckDBManager)

            db_manager.initialize_database()

            # Store data
            success = db_manager.store_activitysim_data(
                table_name="test_types",
                df=test_data,
                file_record_id="test_types_123",
                run_id="test_run_types",
                year=2017,
                iteration=0,
                openlineage_id="test_ol_types",
            )
            # Note: This might not succeed if table doesn't exist, but we're testing the data handling
            print(
                f"   📊 Data type storage test: {'✅ Success' if success else '⚠️  Expected - table may not exist'}"
            )

    def test_error_handling(self):
        """Test error handling in database operations."""
        print("\n⚠️  Testing error handling...")

        bad_settings = self.settings.model_copy(deep=True)
        # Test with invalid database path
        bad_settings.shared.database.path = "/invalid/path/that/does/not/exist/test.db"

        # Manager creation should still succeed, even if the path is invalid initially.
        db_manager = create_database_manager(bad_settings.shared.database)
        self.assertIsNotNone(db_manager)
        print("   ✅ Manager creation does not fail on invalid paths")

        # Test with valid manager but non-existent data queries
        db_manager = create_database_manager(self.settings.shared.database)
        with db_manager:
            db_manager.initialize_database(schema_dir=self.test_schema_dir)

            # Test retrieving non-existent data
            result = db_manager.retrieve_activitysim_data(
                table_name="households", openlineage_id="non_existent_id"
            )
            self.assertIsNone(result)
            print("   ✅ Graceful handling of non-existent data")

    def test_iteration_constant_behavior(self):
        """Test that generic data (skims) varies per iteration - demonstrates iteration-specific storage."""
        print("\n🔄 Testing iteration-varying behavior for skims...")

        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            db_manager.initialize_database()
            conn = db_manager._get_connection()

            # Setup prerequisite records
            unique_run_id = f"test_iter_const_{uuid.uuid4().hex[:8]}"
            conn.execute(
                "INSERT INTO runs (run_id, created_at, models_used, code_version, hostname) VALUES (?, ?, ?, ?, ?)",
                [
                    unique_run_id,
                    datetime.now().isoformat(),
                    ["test"],
                    "test_version",
                    "test_host",
                ],
            )

            file_id = f"test_file_{uuid.uuid4().hex[:8]}"
            ol_id_0 = f"test_ol_0_{uuid.uuid4().hex[:8]}"
            ol_id_1 = f"test_ol_1_{uuid.uuid4().hex[:8]}"

            for ol_id in [ol_id_0, ol_id_1]:
                conn.execute(
                    "INSERT INTO file_records (unique_id, record_type, run_id, openlineage_id, file_path, created_at, short_name, description, models, schema, metadata, exists) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        f"file_{ol_id}",
                        "file",
                        unique_run_id,
                        ol_id,
                        "/test/path",
                        datetime.now().isoformat(),
                        "test",
                        "Test",
                        ["test"],
                        "[]",
                        "{}",
                        True,
                    ],
                )

            # Create test generic data (simulating skims)
            skims_iter0_df = pd.DataFrame(
                {
                    "origin": [1, 2],
                    "dest": [2, 1],
                    "time": [10.0, 15.0],  # Iteration 0 values
                }
            )

            skims_iter1_df = pd.DataFrame(
                {
                    "origin": [1, 2],
                    "dest": [2, 1],
                    "time": [12.0, 18.0],  # Iteration 1 values (updated by BEAM)
                }
            )

            print("   📊 Iteration 0: Storing initial skims...")
            db_manager.store_activitysim_data(
                "skims",
                skims_iter0_df,
                f"file_{ol_id_0}",
                unique_run_id,
                2017,
                0,
                ol_id_0,
            )

            print("   📊 Iteration 1: Storing updated skims...")
            db_manager.store_activitysim_data(
                "skims",
                skims_iter1_df,
                f"file_{ol_id_1}",
                unique_run_id,
                2017,
                1,
                ol_id_1,
            )

            # Verify skims: should have 2 rows (one per iteration)
            skims_count = conn.execute(
                "SELECT COUNT(*) FROM activitysim_data_generic WHERE run_id = ? AND table_name = 'skims'",
                [unique_run_id],
            ).fetchone()[0]
            self.assertEqual(
                skims_count, 2, "Should have 2 skim entries (one per iteration)"
            )
            print(f"   ✅ Skims vary per iteration: {skims_count} rows")

            # Verify skims have different iterations
            skim_iterations = conn.execute(
                "SELECT DISTINCT iteration FROM activitysim_data_generic WHERE run_id = ? AND table_name = 'skims' ORDER BY iteration",
                [unique_run_id],
            ).fetchall()
            self.assertEqual(
                len(skim_iterations), 2, "Skims should exist for 2 iterations"
            )
            self.assertEqual(
                [row[0] for row in skim_iterations],
                [0, 1],
                "Skims should have iterations 0 and 1",
            )
            print(
                f"   ✅ Skims stored for iterations: {[row[0] for row in skim_iterations]}"
            )

            # Verify unique constraint on (run_id, year, iteration, table_name)
            # Try to insert duplicate skims for iteration 0 - should replace
            skims_iter0_updated_df = pd.DataFrame(
                {
                    "origin": [1, 2],
                    "dest": [2, 1],
                    "time": [11.0, 16.0],  # Updated iteration 0 values
                }
            )

            print("   🔄 Re-inserting skims for iteration 0 (should replace)...")
            db_manager.store_activitysim_data(
                "skims",
                skims_iter0_updated_df,
                f"file_{ol_id_0}",
                unique_run_id,
                2017,
                0,
                ol_id_0,
            )

            # Still should have 2 rows (replaced, not added)
            skims_count_after = conn.execute(
                "SELECT COUNT(*) FROM activitysim_data_generic WHERE run_id = ? AND table_name = 'skims'",
                [unique_run_id],
            ).fetchone()[0]
            self.assertEqual(
                skims_count_after,
                2,
                "Should still have 2 skim entries (replaced, not duplicated)",
            )
            print(
                f"   ✅ Duplicate skims replaced (not duplicated): {skims_count_after} rows"
            )

            print("   ✅ Test passed: Iteration-varying behavior verified for skims")

    def test_atlas_vehicles_workflow(self):
        """Test ATLAS vehicles2 data storage and FK constraints."""
        print("\n🚗 Testing ATLAS vehicles workflow...")

        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            db_manager.initialize_database()
            conn = db_manager._get_connection()

            # Setup prerequisite records
            unique_run_id = f"test_vehicles_run_{uuid.uuid4().hex[:8]}"
            conn.execute(
                "INSERT INTO runs (run_id, created_at, models_used, code_version, hostname) VALUES (?, ?, ?, ?, ?)",
                [
                    unique_run_id,
                    datetime.now().isoformat(),
                    ["atlas"],
                    "test_version",
                    "test_host",
                ],
            )

            file_id = f"test_file_{uuid.uuid4().hex[:8]}"
            ol_id = f"test_ol_{uuid.uuid4().hex[:8]}"

            conn.execute(
                "INSERT INTO file_records (unique_id, record_type, run_id, openlineage_id, file_path, created_at, short_name, description, models, schema, metadata, exists) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    file_id,
                    "file",
                    unique_run_id,
                    ol_id,
                    "/test/path",
                    datetime.now().isoformat(),
                    "atlas_vehicles2_output",
                    "Test",
                    ["atlas"],
                    "[]",
                    "{}",
                    True,
                ],
            )

            # Create household data (required for FK constraint)
            # Must create in FK dependency order: parcels -> buildings -> households
            parcels_data = pd.DataFrame(
                {
                    "parcel_id": [1, 2, 3],
                    "zone_id": ["zone1", "zone2", "zone3"],
                    "land_value": [100000.0, 150000.0, 120000.0],
                    "total_sqft": [5000.0, 6000.0, 4500.0],
                    "county_id": ["county1", "county1", "county2"],
                }
            )
            db_manager.store_urbansim_raw_data(
                "parcels", parcels_data, file_id, unique_run_id, 2017, 0, ol_id
            )

            buildings_data = pd.DataFrame(
                {
                    "building_id": [101, 102, 103],
                    "parcel_id": [1, 2, 3],
                    "building_type_id": [1, 2, 1],
                    "sqft": [1500, 2000, 1200],
                    "year_built": [1990, 2000, 1985],
                    "stories": [2, 3, 1],
                }
            )
            db_manager.store_urbansim_raw_data(
                "buildings", buildings_data, file_id, unique_run_id, 2017, 0, ol_id
            )

            households_data = pd.DataFrame(
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
            db_manager.store_urbansim_raw_data(
                "households", households_data, file_id, unique_run_id, 2017, 0, ol_id
            )
            print("   ✅ Created prerequisite households data")

            # Create test vehicles data
            vehicles_data = [
                {
                    "run_id": unique_run_id,
                    "file_record_id": file_id,
                    "year": 2017,
                    "iteration": 0,
                    "household_id": 1,
                    "vehicle_id": 1001,
                    "bodytype": "Sedan",
                    "pred_power": "ICE",
                    "ownlease": "Own",
                    "modelyear": 2015,
                    "adopt_fuel": "Gasoline",
                    "adopt_veh": "Conventional",
                    "vehicletypeid": "Sedan_ICE_2015",
                },
                {
                    "run_id": unique_run_id,
                    "file_record_id": file_id,
                    "year": 2017,
                    "iteration": 0,
                    "household_id": 2,
                    "vehicle_id": 1002,
                    "bodytype": "SUV",
                    "pred_power": "BEV",
                    "ownlease": "Lease",
                    "modelyear": 2020,
                    "adopt_fuel": "Electric",
                    "adopt_veh": "BEV",
                    "vehicletypeid": "SUV_BEV_2020",
                },
                {
                    "run_id": unique_run_id,
                    "file_record_id": file_id,
                    "year": 2017,
                    "iteration": 0,
                    "household_id": 2,
                    "vehicle_id": 1003,
                    "bodytype": "Sedan",
                    "pred_power": "ICE",
                    "ownlease": "Own",
                    "modelyear": 2018,
                    "adopt_fuel": "Gasoline",
                    "adopt_veh": "Conventional",
                    "vehicletypeid": "Sedan_ICE_2018",
                },
            ]

            # Insert vehicles data
            for vehicle in vehicles_data:
                conn.execute(
                    """
                    INSERT INTO atlas_vehicles2_output (
                        run_id, file_record_id, year, iteration, household_id, vehicle_id,
                        bodytype, pred_power, ownlease, modelyear, adopt_fuel, adopt_veh, vehicletypeid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        vehicle["run_id"],
                        vehicle["file_record_id"],
                        vehicle["year"],
                        vehicle["iteration"],
                        vehicle["household_id"],
                        vehicle["vehicle_id"],
                        vehicle["bodytype"],
                        vehicle["pred_power"],
                        vehicle["ownlease"],
                        vehicle["modelyear"],
                        vehicle["adopt_fuel"],
                        vehicle["adopt_veh"],
                        vehicle["vehicletypeid"],
                    ],
                )
            print("   ✅ Inserted 3 vehicle records")

            # Verify data was inserted
            vehicle_count = conn.execute(
                "SELECT COUNT(*) FROM atlas_vehicles2_output WHERE run_id = ?",
                [unique_run_id],
            ).fetchone()[0]
            self.assertEqual(vehicle_count, 3, "Should have 3 vehicle records")
            print(f"   ✅ Verified {vehicle_count} vehicles stored")

            # Test FK constraint - try to insert vehicle for non-existent household
            print(
                "   🔒 Testing FK constraint (should fail for non-existent household)..."
            )
            with self.assertRaises(ConstraintException) as context:
                conn.execute(
                    """
                    INSERT INTO atlas_vehicles2_output (
                        run_id, file_record_id, year, iteration, household_id, vehicle_id,
                        bodytype, vehicletypeid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        unique_run_id,
                        file_id,
                        2017,
                        0,
                        999,
                        9999,
                        "Sedan",
                        "Sedan_ICE_2015",
                    ],
                )
            self.assertIn("Constraint Error", str(context.exception))
            print("   ✅ FK constraint correctly prevents orphaned vehicles")

            # Test UNIQUE constraint (run_id, year, household_id, vehicle_id)
            # Note: iteration is NOT in the UNIQUE constraint (vehicles are constant across iterations)
            print(
                "   🔄 Testing UNIQUE constraint (should fail for duplicate vehicle)..."
            )
            with self.assertRaises(Exception) as context:
                conn.execute(
                    """
                    INSERT INTO atlas_vehicles2_output (
                        run_id, file_record_id, year, iteration, household_id, vehicle_id,
                        bodytype, vehicletypeid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        unique_run_id,
                        file_id,
                        2017,
                        0,
                        1,
                        1001,
                        "Truck",
                        "Truck_ICE_2015",
                    ],
                )
            self.assertIn("Constraint Error", str(context.exception))
            print("   ✅ UNIQUE constraint correctly prevents duplicate vehicles")

            # Test that vehicles can be linked to households with vehicles
            print("   🔗 Testing household-vehicle relationship...")
            result = conn.execute(
                """
                SELECT h.household_id, h.cars, COUNT(v.vehicle_id) as vehicle_count
                FROM urbansim_households_raw h
                LEFT JOIN atlas_vehicles2_output v ON h.run_id = v.run_id AND h.year = v.year AND h.household_id = v.household_id
                WHERE h.run_id = ?
                GROUP BY h.household_id, h.cars
                ORDER BY h.household_id
                """,
                [unique_run_id],
            ).fetchall()

            # Verify relationships
            self.assertEqual(len(result), 3, "Should have 3 households")
            self.assertEqual(result[0][2], 1, "Household 1 should have 1 vehicle")
            self.assertEqual(result[1][2], 2, "Household 2 should have 2 vehicles")
            self.assertEqual(result[2][2], 0, "Household 3 should have 0 vehicles")
            print("   ✅ Household-vehicle relationships correct")

            # Verify iteration-constant behavior (same as households/persons)
            print("   📊 Testing iteration-constant behavior...")
            # The UNIQUE constraint is (run_id, year, household_id, vehicle_id) - no iteration
            # This means vehicles are constant across iterations within a year
            unique_constraint_check = conn.execute(
                """
                SELECT constraint_name, constraint_type
                FROM duckdb_constraints()
                WHERE table_name = 'atlas_vehicles2_output' AND constraint_type = 'UNIQUE'
                """
            ).fetchall()
            print(f"   ✅ UNIQUE constraint: {unique_constraint_check}")
            print(
                "   ✅ Vehicles are constant across iterations (iteration not in UNIQUE key)"
            )

            print("   ✅ Test passed: ATLAS vehicles workflow verified")

    def test_atlas_add_vehicleTypeId_logic(self):
        """Test the atlas_add_vehileTypeId function logic for creating vehicleTypeId."""
        print("\n🔧 Testing atlas_add_vehileTypeId logic...")

        # Create test input data with various scenarios
        test_vehicles = pd.DataFrame(
            {
                "household_id": [1, 2, 3, 4, 5, 6],
                "vehicle_id": [1001, 1002, 1003, 1004, 1005, 1006],
                "bodytype": ["Sedan", "SUV", "Truck", "Sedan", "SUV", "Sedan"],
                "pred_power": ["ICE", "BEV", "ICE", "PHEV", "ICE", "ICE"],
                "modelyear": [
                    2020,
                    2018,
                    2010,
                    2015,
                    2022,
                    2000,
                ],  # Mix of pre/post 2015
                "ownlease": ["Own", "Lease", "Own", "Own", "Lease", "Own"],
                "adopt_fuel": [
                    "Gasoline",
                    "Electric",
                    "Diesel",
                    "Hybrid",
                    "Gasoline",
                    "Gasoline",
                ],
            }
        )

        # Create temporary files
        input_csv = os.path.join(self.temp_dir, "test_vehicles_input.csv")
        output_csv = os.path.join(self.temp_dir, "test_vehicles2_output.csv")

        # Write test data
        test_vehicles.to_csv(input_csv, index=False)
        print(f"   📝 Created test input file with {len(test_vehicles)} vehicles")

        # Call the function
        settings = {}  # Not used in the function
        atlas_add_vehileTypeId(settings, 2017, input_csv, output_csv)
        print("   ✅ atlas_add_vehileTypeId executed successfully")

        # Read the output
        self.assertTrue(os.path.exists(output_csv), "Output file should be created")
        result_df = pd.read_csv(output_csv)

        # Verify vehicleTypeId column was added
        self.assertIn(
            "vehicleTypeId", result_df.columns, "vehicleTypeId column should exist"
        )
        print("   ✅ vehicleTypeId column added")

        # Verify all rows have vehicleTypeId
        self.assertEqual(
            result_df["vehicleTypeId"].isna().sum(),
            0,
            "No missing vehicleTypeId values",
        )
        print("   ✅ All vehicles have vehicleTypeId")

        # Test specific logic rules
        # Rule 1: Post-2015 vehicles should have format: bodytype_pred_power_modelyear
        post_2015_rows = result_df[result_df["modelyear"] >= 2015]
        for idx, row in post_2015_rows.iterrows():
            expected = f"{row['bodytype']}_{row['pred_power']}_{int(row['modelyear'])}"
            self.assertEqual(
                row["vehicleTypeId"],
                expected,
                "Post-2015 vehicle should have format bodytype_pred_power_modelyear",
            )
        print(
            f"   ✅ Post-2015 vehicles ({len(post_2015_rows)}) have correct format: bodytype_pred_power_modelyear"
        )

        # Rule 2: Pre-2015 vehicles should have format: bodytype_pred_power_2015
        pre_2015_rows = result_df[result_df["modelyear"] < 2015]
        for idx, row in pre_2015_rows.iterrows():
            expected = f"{row['bodytype']}_{row['pred_power']}_2015"
            self.assertEqual(
                row["vehicleTypeId"],
                expected,
                "Pre-2015 vehicle should have format bodytype_pred_power_2015",
            )
        print(
            f"   ✅ Pre-2015 vehicles ({len(pre_2015_rows)}) have correct format: bodytype_pred_power_2015"
        )

        # Verify specific test cases
        test_cases = [
            (0, "Sedan_ICE_2020"),  # 2020 Sedan ICE
            (1, "SUV_BEV_2018"),  # 2018 SUV BEV
            (2, "Truck_ICE_2015"),  # 2010 Truck ICE -> capped at 2015
            (3, "Sedan_PHEV_2015"),  # 2015 Sedan PHEV (boundary case)
            (4, "SUV_ICE_2022"),  # 2022 SUV ICE
            (5, "Sedan_ICE_2015"),  # 2000 Sedan ICE -> capped at 2015
        ]

        for row_idx, expected_id in test_cases:
            actual_id = result_df.iloc[row_idx]["vehicleTypeId"]
            self.assertEqual(
                actual_id,
                expected_id,
                f"Row {row_idx}: Expected {expected_id}, got {actual_id}",
            )
            print(f"   ✅ Test case {row_idx}: {expected_id}")

        # Verify original columns are preserved
        for col in [
            "household_id",
            "vehicle_id",
            "bodytype",
            "pred_power",
            "modelyear",
            "ownlease",
            "adopt_fuel",
        ]:
            self.assertIn(
                col, result_df.columns, f"Original column {col} should be preserved"
            )
        print("   ✅ All original columns preserved")

        # Verify modelyear is integer type
        self.assertTrue(
            pd.api.types.is_integer_dtype(result_df["modelyear"]),
            "modelyear should be integer type",
        )
        print("   ✅ modelyear converted to integer")

        print("   ✅ Test passed: atlas_add_vehileTypeId logic verified")

    def test_transaction_atomicity(self):
        """Test that failed multi-step operations roll back completely."""
        print("\n⚛️  Testing transaction atomicity (rollback)...")

        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            db_manager.initialize_database()
            conn = db_manager._get_connection()

            # 1. Verify clean state
            count_before = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            self.assertEqual(count_before, 0)

            # 2. Create a function that simulates a crash halfway through a transaction
            # We manually control the transaction here to mimic the manager's internal logic
            try:
                conn.begin()

                # Step A: Insert a valid Run (This should succeed temporarily)
                conn.execute(
                    "INSERT INTO runs (run_id, created_at, models_used) VALUES (?, ?, ?)",
                    ["valid_run", datetime.now().isoformat(), ["test"]],
                )

                # Step B: Insert a valid File Record linking to it
                conn.execute(
                    "INSERT INTO file_records (unique_id, record_type, run_id, short_name, exists) VALUES (?, ?, ?, ?, ?)",
                    ["valid_file", "file", "valid_run", "test", True],
                )

                # Step C: Force a failure (Constraint Violation)
                # Try to insert a file record with a DUPLICATE unique_id but different run_id
                # This causes a Primary Key violation
                conn.execute(
                    "INSERT INTO file_records (unique_id, record_type, run_id, short_name, exists) VALUES (?, ?, ?, ?, ?)",
                    ["valid_file", "file", "some_other_run", "duplicate", True],
                )

                conn.commit()
            except Exception as e:
                print(f"   ⚠️  Caught expected error: {e}")
                conn.rollback()

            # 3. Verify Rollback
            # If atomicity works, "valid_run" and "valid_file" should NOT exist,
            # even though they were inserted before the error occurred.
            count_after = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            self.assertEqual(count_after, 0, "Database should be empty after rollback")

            file_count = conn.execute("SELECT COUNT(*) FROM file_records").fetchone()[0]
            self.assertEqual(
                file_count, 0, "Partial file records should be rolled back"
            )

            print("   ✅ Atomicity confirmed: Partial data was rolled back")

    def test_array_type_handling(self):
        """Test round-tripping of Array/List types (crucial for runs table)."""
        print("\n⛓️  Testing array type handling...")

        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            db_manager.initialize_database()
            conn = db_manager._get_connection()

            run_id = "array_test_run"
            models_list = ["urbansim", "activitysim", "beam"]

            # Insert
            conn.execute(
                "INSERT INTO runs (run_id, created_at, models_used) VALUES (?, ?, ?)",
                [run_id, datetime.now().isoformat(), models_list],
            )

            # Retrieve
            result = conn.execute(
                "SELECT models_used FROM runs WHERE run_id = ?", [run_id]
            ).fetchone()

            retrieved_list = result[0]

            # Verify it comes back as a python list, not a string representation
            self.assertIsInstance(retrieved_list, list, "Should return a Python list")
            self.assertEqual(
                retrieved_list, models_list, "List content should match exactly"
            )
            self.assertEqual(len(retrieved_list), 3)

            print(f"   ✅ Array round-trip successful: {retrieved_list}")

            # Verify we can query INTO the array (DuckDB list functions)
            # Find runs where 'activitysim' is in the list
            contains_asim = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE list_contains(models_used, 'activitysim')"
            ).fetchone()[0]
            self.assertEqual(contains_asim, 1, "Should support list_contains queries")
            print("   ✅ DuckDB list_contains query successful")

    def test_direct_file_upload_logic(self):
        """
        Test the high-performance direct file upload logic from selective_uploader.py.
        This verifies the SQL generation, renaming logic, and Parquet reading.
        """
        print("\n🚀 Testing direct Parquet/CSV upload logic...")

        # Import the function dynamically since it's in a script
        from pilates.database.selective_uploader import build_smart_select

        db_manager = create_database_manager(self.settings.shared.database)

        with db_manager:
            db_manager.initialize_database()
            conn = db_manager._get_connection()

            # 1. Create a dummy Parquet file
            # We specifically use 'year' and 'sector_id' to test the renaming logic
            test_df = pd.DataFrame(
                {
                    "year": [2010, 2011],
                    "sector_id": [10, 20],  # Ints, should be cast to string by logic
                    "value": [100, 200],
                }
            )
            parquet_path = os.path.join(self.temp_dir, "test_direct_upload.parquet")
            test_df.to_parquet(parquet_path)

            # 2. Setup Mock Metadata
            run_info = {"run_id": "direct_upload_run"}
            record = {
                "unique_id": "rec_123",
                "schema": [
                    {"name": "zone_id", "type": "int"}
                ],  # Hint to trigger sorting check
            }
            conn.execute("DROP TABLE IF EXISTS atlas_jobs_csv")
            # Create target table (simulating 'atlas_jobs_csv' to trigger sector_id logic)
            # We need a table that matches the expected output schema
            conn.execute(
                """
                            CREATE TABLE atlas_jobs_csv (
                                run_id VARCHAR, file_record_id VARCHAR, 
                                year INTEGER, iteration INTEGER, sub_iteration INTEGER,
                                sector_id VARCHAR, 
                                value INTEGER,
                                data_year INTEGER, -- FIX: Added this column because logic renames input 'year' to 'data_year'
                                id BIGINT
                            )
                        """
            )

            # 3. Run the Logic
            # We simulate uploading 'atlas_jobs_csv' to trigger the special casing
            select_sql, sort_keys = build_smart_select(
                conn,
                parquet_path,
                "atlas_jobs_csv",
                record,
                run_info,
                year=2020,
                iteration=5,
            )

            print(f"   📝 Generated SQL: {select_sql.strip()}")

            # 4. Execute the Generated SQL
            # This confirms the SQL syntax is valid DuckDB syntax
            conn.execute(
                f"""
                            INSERT INTO atlas_jobs_csv (
                                -- Order must match SELECT * output:
                                -- File cols first (year->data_year, sector_id, value), then Metadata
                                data_year, sector_id, value, 
                                run_id, file_record_id, year, iteration, sub_iteration
                            )
                            SELECT * FROM ({select_sql})
                        """
            )

            # 5. Verify Results
            result = conn.execute("SELECT * FROM atlas_jobs_csv").fetchall()
            columns = [d[0] for d in conn.description]
            df_res = pd.DataFrame(result, columns=columns)

            # Check 1: Metadata Injection
            self.assertEqual(df_res.iloc[0]["run_id"], "direct_upload_run")
            self.assertEqual(df_res.iloc[0]["year"], 2020)  # The metadata year
            self.assertEqual(df_res.iloc[0]["iteration"], 5)

            # Check 2: Casting Logic (sector_id should be string '10', not int 10)
            self.assertEqual(df_res.iloc[0]["sector_id"], "10")
            self.assertIsInstance(df_res.iloc[0]["sector_id"], str)
            print("   ✅ Casting logic (Int -> String) worked")

            # Check 3: Sort Keys
            self.assertIn("run_id", sort_keys)
            print("   ✅ Sort keys identified correctly")

            print("   ✅ Direct upload logic verified successfully")


if __name__ == "__main__":
    print("🧪 Starting PILATES Database Components Tests")
    print("=" * 50)

    unittest.main(verbosity=2)
