#!/usr/bin/env python3
"""
Tests for the hybrid (database vs. external file) storage workflow.

This test suite validates the functionality of the view-based system that allows
querying data transparently, whether it's stored in DuckDB or as external
Parquet files.
"""

import os
import tempfile
import unittest
import shutil
import pandas as pd
from pathlib import Path
import uuid
import subprocess

# Import PILATES modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pilates.utils.duckdb_manager import DuckDBManager
from pilates.database.scripts import create_views
from pilates.database.selective_uploader import run_uploader


# Since selective_uploader is a script, we'll use subprocess to run it
UPLOADER_SCRIPT_PATH = Path(__file__).parent.parent / "pilates" / "database" / "selective_uploader.py"
PYTHON_EXECUTABLE = "/Users/zaneedell/miniforge3/envs/PILATES/bin/python"

class TestHybridStorage(unittest.TestCase):
    """Test the dual storage (in-db vs. external) workflow."""

    def setUp(self):
        """Set up a temporary test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pilates_hybrid_test_"))
        self.db_path = self.temp_dir / "test.duckdb"
        self.db_manager = DuckDBManager(database_path=str(self.db_path))

        # --- Schema Generation ---
        # Create a temporary schema directory layout that mirrors the real one
        self.test_schema_dir = self.temp_dir / "schema"
        self.generated_dir = self.test_schema_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        # Copy original schema files to the temporary location
        original_schema_dir = Path(__file__).parent.parent / "pilates" / "database" / "schema"
        for fname in os.listdir(original_schema_dir):
            if fname.endswith('.sql'):
                shutil.copy(original_schema_dir / fname, self.test_schema_dir)

        # Point the view creation script to our temporary directories and run it
        create_views.SCHEMA_DIR = self.test_schema_dir
        create_views.GENERATED_DIR = self.generated_dir
        create_views.main()

        # Now, initialize the database using the temporary schema directory
        with self.db_manager:
            self.db_manager.initialize_database(schema_dir=str(self.test_schema_dir))


    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)

    def _create_dummy_run(self, run_id: str, file_name: str, data: pd.DataFrame) -> Path:
        """Helper to create a dummy run directory with a parquet file and run_info.json."""
        run_dir = self.temp_dir / run_id
        run_dir.mkdir()
        
        parquet_path = run_dir / file_name
        data.to_parquet(parquet_path)
        
        file_record = {
            "unique_id": f"hash_{run_id}",
            "openlineage_id": str(uuid.uuid4()),
            "file_path": str(parquet_path.relative_to(run_dir)),
            "created_at": "2025-01-01T00:00:00Z",
            "short_name": file_name.replace(".parquet", ""),
            "description": "Dummy test file",
            "models": ["dummy_model"],
            "schema": [],
            "metadata": {},
            "exists": True,
            "year": 2020,
            "iteration": 0,
            "sub_iteration": 0,
        }

        run_info = {
            "run_id": run_id,
            "created_at": "2025-01-01T00:00:00Z",
            "models_used": ["dummy_model"],
            "code_version": "test",
            "hostname": "test-host",
            "file_records": {
                f"hash_{run_id}": file_record
            }
        }
        
        run_info_path = run_dir / "run_info.json"
        with open(run_info_path, 'w') as f:
            import json
            json.dump(run_info, f, indent=4)
            
        return run_info_path

    def test_parquet_upload_to_database(self):
        """
        Tests that Parquet files are uploaded into the database by default.
        """
        print("\n📤 Testing Parquet upload to database...")
        
        # 1. Setup
        run_id = "run_upload"
        df = pd.DataFrame({'household_id': [1, 2], 'hhsize': [3, 4]})
        run_info_path = self._create_dummy_run(run_id, "households_asim_out.parquet", df)
        
        # 2. Execute
        success = run_uploader(
            run_info_path=str(run_info_path),
            database_path=str(self.db_path),
            tables=["households_asim_out"]
        )

        self.assertTrue(success, f"Uploader script failed")

        # 3. Verify
        with self.db_manager:
            conn = self.db_manager._get_connection()

            # Check physical table has data
            count = conn.execute("SELECT COUNT(*) FROM uploaded_households_asim_out").fetchone()[0]
            self.assertEqual(count, 2, "Data should be in the uploaded_ table")
            print(f"   ✅ `uploaded_households_asim_out` contains {count} rows.")

            # Check file_record metadata
            storage_loc = conn.execute("SELECT storage_location FROM file_records WHERE run_id=?", [run_id]).fetchone()[0]
            self.assertEqual(storage_loc, 'database', "storage_location should be 'database'")
            print(f"   ✅ file_record storage_location is '{storage_loc}'.")

            # Check combined query
            result_relation = self.db_manager.query_hybrid_table('households_asim_out')
            view_count = result_relation.filter(f"run_id='{run_id}'").aggregate('COUNT(*)').fetchone()[0]
            self.assertEqual(view_count, 2, "Hybrid query should return the uploaded data.")
            print(f"   ✅ Hybrid query returns {view_count} rows.")

    def test_parquet_link_external(self):
        """
        Tests that Parquet files are linked externally with the --no-upload-parquet flag.
        """
        print("\n🔗 Testing external Parquet file linking...")

        # 1. Setup
        run_id = "run_link"
        df = pd.DataFrame({'household_id': [10, 20], 'hhsize': [1, 5]})
        run_info_path = self._create_dummy_run(run_id, "households_asim_out.parquet", df)
        
        # 2. Execute

        success = run_uploader(
            run_info_path=str(run_info_path),
            database_path=str(self.db_path),
            tables=["households_asim_out"],
            no_upload_parquet=True
        )
        self.assertTrue(success, f"Uploader script failed")

        # 3. Verify
        with self.db_manager:
            conn = self.db_manager._get_connection()
            
            # Check physical table is empty
            count = conn.execute("SELECT COUNT(*) FROM uploaded_households_asim_out").fetchone()[0]
            self.assertEqual(count, 0, "Data should NOT be in the uploaded_ table")
            print(f"   ✅ `uploaded_households_asim_out` is empty as expected.")

            # Check file_record metadata
            record = conn.execute("SELECT storage_location, file_path FROM file_records WHERE run_id=?", [run_id]).fetchone()
            self.assertEqual(record[0], 'external', "storage_location should be 'external'")
            self.assertTrue(Path(record[1]).exists(), "File path in record should exist")
            self.assertTrue(Path(record[1]).is_absolute(), "File path in record should be absolute")
            print(f"   ✅ file_record storage_location is '{record[0]}'.")

            # Check combined query
            result_relation = self.db_manager.query_hybrid_table('households_asim_out')
            view_count = result_relation.filter(f"run_id='{run_id}'").aggregate('COUNT(*)').fetchone()[0]
            self.assertEqual(view_count, 2, "Hybrid query should read from the external file.")
            print(f"   ✅ Hybrid query returns {view_count} rows from external file.")

    def test_unified_view_combines_sources(self):
        """
        Tests that the unified view correctly queries from both the database and external files.
        """
        print("\n🤝 Testing unified view with mixed sources...")

        # 1. Setup & Execute for Run 1 (Upload)
        run_id_1 = "run_hybrid_upload"
        df1 = pd.DataFrame({'household_id': [1, 2, 3], 'hhsize': [3, 4, 1]})
        run_info_path_1 = self._create_dummy_run(run_id_1, "households_asim_out.parquet", df1)

        success1 = run_uploader(
            run_info_path=str(run_info_path_1),
            database_path=str(self.db_path),
            tables=["households_asim_out"]
        )
        # self.assertTrue(success1, f"Uploader script failed for run 1")

        print("   - Run 1 (upload) complete.")

        # 2. Setup & Execute for Run 2 (Link)
        run_id_2 = "run_hybrid_link"
        df2 = pd.DataFrame({'household_id': [10, 20], 'hhsize': [1, 5]})
        run_info_path_2 = self._create_dummy_run(run_id_2, "households_asim_out.parquet", df2)

        success2 = run_uploader(
            run_info_path=str(run_info_path_2),
            database_path=str(self.db_path),
            tables=["households_asim_out"],
            no_upload_parquet=True
        )
        # self.assertTrue(success2, f"Uploader script failed for run 2")
        print("   - Run 2 (link) complete.")

        self.db_manager.close()

        # 3. Verify
        with self.db_manager:
            # --- ADD THIS DIAGNOSTIC BLOCK ---
            print("\n--- DIAGNOSTIC: DUMPING file_records TABLE ---")
            conn = self.db_manager._get_connection()
            records_df = conn.execute("SELECT run_id, unique_id, storage_location, logical_table_name, data_format FROM file_records").fetchdf()
            print(records_df.T)
            print("--------------------------------------------\n")
            # --- END DIAGNOSTIC BLOCK ---
            result_relation = self.db_manager.query_hybrid_table('households_asim_out')

            # You can now work with the result
            # CORRECT WAY to get the total count:
            total_count = result_relation.aggregate('COUNT(*)').fetchone()[0]

            self.assertEqual(total_count, 5, "Unified query should return combined count from both sources.")
            print(f"   ✅ Unified query returned a total of {total_count} rows.")

            # To get data for a specific run, use .filter() and then a terminator
            run1_data = result_relation.filter(f"run_id = '{run_id_1}'").fetchall()
            self.assertEqual(len(run1_data), 3)

            run2_data = result_relation.filter(f"run_id = '{run_id_2}'").fetchall()
            self.assertEqual(len(run2_data), 2)
            print(f"   ✅ Verified correct row counts for each run_id.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
