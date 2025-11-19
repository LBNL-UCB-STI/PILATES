#!/usr/bin/env python3
"""
Unit tests for schema inference and statistical analysis.

This suite verifies the logic used to generate run_info.json, ensuring that
data types, value ranges (min/max), and categorical enums are correctly
detected from Parquet and CSV files.
"""

import os
import shutil
import tempfile
import unittest
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Adjust import based on your project structure
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pilates.generic.records import RecordStore
from pilates.utils.schema_inference import get_schema_from_file, analyze_series_stats


class TestSchemaInference(unittest.TestCase):

    def setUp(self):
        """Create a temporary workspace for generating test files."""
        self.temp_dir = tempfile.mkdtemp(prefix="pilates_schema_test_")
        # We instantiate the class containing the logic.
        # If these methods become static or standalone functions, adjust accordingly.
        self.rs = RecordStore()

    def tearDown(self):
        """Cleanup temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_parquet_file(self, filename, df):
        """Helper to write Parquet files with statistics."""
        path = os.path.join(self.temp_dir, filename)
        table = pa.Table.from_pandas(df)
        # We must ensure statistics are written to the footer for the optimizer to work
        pq.write_table(table, path, write_statistics=True)
        return path

    def create_csv_file(self, filename, df):
        """Helper to write CSV files."""
        path = os.path.join(self.temp_dir, filename)
        df.to_csv(path, index=False)
        return path

    def test_integer_range_detection(self):
        """
        Test that we correctly identify Min/Max for integers.
        This is crucial for the SQL generator to decide between SMALLINT, INTEGER, and BIGINT.
        """
        print("\n🔍 Testing Integer Range Detection...")

        df = pd.DataFrame(
            {
                "tiny_int": np.array([-100, 0, 100], dtype=np.int8),  # Fits in SMALLINT
                "standard_int": np.array(
                    [-50000, 0, 50000], dtype=np.int32
                ),  # Fits in INTEGER
                "big_int": np.array(
                    [3_000_000_000, 4_000_000_000, 3_500_000_000], dtype=np.int64
                ),  # Needs BIGINT
            }
        )

        file_path = self.create_parquet_file("integers.parquet", df)

        # Run Inference
        schema = get_schema_from_file(file_path)

        # Verify Tiny Int
        tiny = next(col for col in schema if col["name"] == "tiny_int")
        self.assertEqual(tiny["min"], -100)
        self.assertEqual(tiny["max"], 100)
        self.assertIn("int", tiny["type"].lower())
        print("   ✅ Tiny Int stats captured")

        # Verify Standard Int
        std = next(col for col in schema if col["name"] == "standard_int")
        self.assertEqual(std["min"], -50000)
        self.assertEqual(std["max"], 50000)
        print("   ✅ Standard Int stats captured")

        # Verify Big Int
        big = next(col for col in schema if col["name"] == "big_int")
        self.assertEqual(big["min"], 3_000_000_000)
        self.assertGreater(big["max"], 2_147_483_647)  # Greater than max 32-bit int
        print("   ✅ Big Int stats captured")

    def test_enum_detection(self):
        """
        Test that low-cardinality string columns are flagged as ENUMs.
        """
        print("\n🔍 Testing Enum Detection...")

        # 'mode' has low cardinality (3 unique values) -> Should be ENUM
        # 'id' has high cardinality (100 unique values) -> Should be VARCHAR
        df = pd.DataFrame(
            {
                "mode": ["WALK", "DRIVE", "TRANSIT", "WALK"] * 25,
                "id": [f"trip_{i}" for i in range(100)],
            }
        )

        file_path = self.create_csv_file("enums.csv", df)

        # We need to access the internal logic or ensure _get_schema_from_file calls _analyze_series_stats
        # Note: _get_schema_from_file calls _analyze_series_stats internally for CSVs
        schema = get_schema_from_file(file_path)

        # Verify Enum
        mode_col = next(col for col in schema if col["name"] == "mode")
        self.assertTrue(mode_col.get("is_enum"), "Should detect 'mode' as enum")
        self.assertCountEqual(mode_col.get("enum_values"), ["WALK", "DRIVE", "TRANSIT"])
        print("   ✅ Low cardinality strings detected as Enum")

        # Verify Non-Enum
        id_col = next(col for col in schema if col["name"] == "id")
        self.assertFalse(
            id_col.get("is_enum"), "High cardinality 'id' should NOT be enum"
        )
        print("   ✅ High cardinality strings ignored")

    def test_integer_like_floats(self):
        """
        Test detection of floats that are actually integers (often due to NaNs).
        This allows us to cast to INTEGER in SQL if safe.
        """
        print("\n🔍 Testing Integer-like Floats...")

        df = pd.DataFrame(
            {
                "real_float": [1.1, 2.5, 3.9],
                "fake_float": [1.0, 2.0, 3.0],  # Can be safely cast to INT
                "float_with_nan": [1.0, 2.0, None],  # Can be cast to INT (NULLABLE)
            }
        )

        # We use CSV here because Parquet preserves the type more strictly,
        # while CSV/Pandas often infers Float for ints with NaNs.
        file_path = self.create_csv_file("floats.csv", df)
        schema = get_schema_from_file(file_path)

        # Real Float
        real = next(col for col in schema if col["name"] == "real_float")
        self.assertFalse(real.get("is_integer_like"))

        # Fake Float
        fake = next(col for col in schema if col["name"] == "fake_float")
        self.assertTrue(fake.get("is_integer_like"), "1.0, 2.0 should be integer-like")

        # Float with NaN
        nan_col = next(col for col in schema if col["name"] == "float_with_nan")
        self.assertTrue(
            nan_col.get("is_integer_like"), "1.0, 2.0, NaN should be integer-like"
        )
        print("   ✅ Integer-like floats correctly identified")

    def test_parquet_metadata_optimization(self):
        """
        Verify that the Parquet reader extracts min/max without loading the dataframe.
        """
        print("\n🔍 Testing Parquet Metadata Optimization...")
        df = pd.DataFrame({"val": range(1000)})
        file_path = self.create_parquet_file("metadata.parquet", df)

        schema = get_schema_from_file(file_path)

        if not schema:
            self.fail("Schema list is empty. Check if PyArrow can read the file.")

        col = schema[0]
        self.assertEqual(col["min"], 0)
        self.assertEqual(col["max"], 999)
        print("   ✅ Parquet stats extracted")

    def test_numpy_serialization(self):
        """
        Ensure that stats extracted are JSON serializable (Python native types),
        not NumPy types (int64, float64), which break json.dump().
        """
        print("\n🔍 Testing JSON Serialization Compatibility...")

        df = pd.DataFrame({"val": np.array([1, 2, 3], dtype=np.int64)})

        # Test via the helper method directly if accessible, or via CSV path
        # _analyze_series_stats is where the casting happens
        stats = analyze_series_stats(df["val"])

        min_val = stats["min"]
        max_val = stats["max"]

        # Check types
        self.assertNotIsInstance(
            min_val, np.generic, "Min value should be native Python int/float"
        )
        self.assertNotIsInstance(
            max_val, np.generic, "Max value should be native Python int/float"
        )

        # Verify it dumps to JSON without error
        try:
            json_str = json.dumps(stats)
            print("   ✅ Stats are JSON serializable")
        except TypeError as e:
            self.fail(f"Stats dictionary contains non-serializable types: {e}")

    def test_boolean_handling(self):
        """Test boolean logic."""
        print("\n🔍 Testing Boolean Handling...")
        df = pd.DataFrame({"flag": [True, False, True]})
        file_path = self.create_parquet_file("bools.parquet", df)

        schema = get_schema_from_file(file_path)

        if not schema:
            self.fail("Schema list is empty. Check get_schema_from_file logic.")

        col = schema[0]
        self.assertIn("bool", col["type"].lower())
        print("   ✅ Booleans identified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
