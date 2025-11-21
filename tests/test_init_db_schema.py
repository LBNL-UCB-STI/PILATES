import unittest
import os
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add the project root to the Python path to allow imports from pilates.*
project_root = Path(__file__).resolve().parent.parent.parent
import sys

sys.path.insert(0, str(project_root))

from pilates.utils.duckdb_manager import DuckDBManager


class TestInitDbSchema(unittest.TestCase):
    """Tests for the init_db_schema.py script."""

    def setUp(self):
        """Set up a temporary directory and database path for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="pilates_test_db_init_")
        self.db_path = os.path.join(self.temp_dir, "test.duckdb")
        self.script_path = (
            Path(__file__).parent.parent / "pilates" / "database" / "init_db_schema.py"
        )
        self.script_path = str(self.script_path.resolve())
        self.python_executable = (
            sys.executable
        )  # Use the same python executable that is running the tests

    def tearDown(self):
        """Clean up the temporary directory after testing."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_schema_initialization(self):
        """Test that the init_db_schema.py script successfully initializes the database schema."""
        print(f"\n🧪 Testing schema initialization with script: {self.script_path}")
        print(f"   Using Python executable: {self.python_executable}")

        # Run the script as a subprocess
        result = subprocess.run(
            [self.python_executable, self.script_path, self.db_path],
            capture_output=True,
            text=True,
            check=False,  # Do not raise an exception for non-zero exit codes
        )

        # Print output for debugging
        print("--- Script STDOUT ---")
        print(result.stdout)
        print("--- Script STDERR ---")
        print(result.stderr)
        print("---------------------")

        # Assert that the script exited successfully
        self.assertEqual(
            result.returncode,
            0,
            f"Script failed with exit code {result.returncode}. Stderr: {result.stderr}",
        )

        # Assert that the database file was created
        self.assertTrue(
            os.path.exists(self.db_path), "DuckDB database file was not created."
        )
        print(f"   ✅ Database file created at: {self.db_path}")

        # Connect to the database and verify tables
        db_manager = DuckDBManager(self.db_path)
        with db_manager:
            conn = db_manager._get_connection()
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]

            expected_tables = [
                "runs",
                "file_records",
                "model_runs",
                "config_snapshots",
                "openlineage_events",
                "urbansim_households_raw",
                "urbansim_persons_raw",
                "urbansim_jobs_raw",
                "urbansim_blocks_raw",
                "urbansim_buildings_raw",
                "urbansim_parcels_raw",
                "activitysim_households",
                "activitysim_persons",
                "activitysim_land_use",
                "activitysim_data_generic",
                "h5_table_records",
                "snapshots",
                "usim_data_reference_counties",
                "usim_data_reference_ect",
            ]

            for table in expected_tables:
                self.assertIn(table, table_names, f"Table '{table}' was not created.")
                print(f"   ✅ Table '{table}' exists.")

        print("   ✅ All expected tables verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
