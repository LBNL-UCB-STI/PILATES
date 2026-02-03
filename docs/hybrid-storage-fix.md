# Design Doc: Implementing a Hybrid Storage System in PILATES with DuckDB UDTFs

**Author:** Zach. (via Gemini)
**Date:** 2025-11-20
**Status:** Proposed

## 1. Overview

This document outlines the plan to implement a hybrid storage architecture for Parquet files within the PILATES project. The goal is to allow data to be either stored directly in a DuckDB database or linked externally on the filesystem, while providing a seamless and transparent querying experience for the end-user.

The initial view-based approach failed due to a fundamental limitation in DuckDB's `read_parquet` function. This plan details a refined strategy using a Python User-Defined Table Function (UDTF) to overcome this limitation, resulting in a robust, efficient, and maintainable solution.

## 2. The Goal

The primary objective is to create a flexible data storage system where:
-   Large Parquet files can be registered with the PILATES database without needing to be fully uploaded, saving storage space and ingestion time.
-   Smaller or critical Parquet files can be ingested directly into DuckDB tables for performance or portability.
-   Analysts can query a single, unified view (e.g., `households_asim_out`) and get complete results, regardless of where the underlying data for each run is physically stored.

## 3. The Core Problem: `read_parquet` Limitation

The initial implementation attempted to create a SQL view that would `UNION` data from an `uploaded_` table with data read directly from external files referenced in the `file_records` table. This approach failed with a specific and critical DuckDB error:

> **Binder Error:** Table function "read_parquet" does not support lateral join column parameters - cannot use column "fr.file_path" in this context. The function only supports literals as parameters.

**Analysis:** This error occurs because DuckDB's query planner needs to know the schema (column names and types) of all data sources *before* it begins executing a query.
-   When using `read_parquet('/path/to/file.parquet')`, DuckDB can inspect the file's metadata ahead of time to determine its schema.
-   When using `read_parquet(fr.file_path)`, the value of `fr.file_path` is only determined during query execution. The planner cannot know which file (and therefore which schema) it will need to read, so it fails during the "binding" phase.

This limitation makes a pure-SQL view for dynamic file paths impossible.

## 4. The Solution: A Python User-Defined Table Function (UDTF)

The correct and idiomatic way to solve this in DuckDB is to encapsulate the dynamic logic within a Python UDTF. This approach combines the flexibility of Python with the high performance of DuckDB's native execution engine.

**How it Works:**

1.  A SQL view (e.g., `households_asim_out`) will call a Python function registered in DuckDB (e.g., `python_get_hybrid_table('households_asim_out')`).
2.  The Python function executes a simple SQL query against the `file_records` table to get a list of all *absolute paths* for the required external Parquet files.
3.  The function then dynamically constructs a *new* SQL query string, passing the collected list of literal paths to DuckDB's native `read_parquet` function (e.g., `SELECT * FROM read_parquet(['/path/1.parquet', '/path/2.parquet'])`).
4.  DuckDB executes this generated query using its highly optimized Parquet reader.
5.  The results are returned to the original SQL view as a table, which can then be combined with the data from the `uploaded_` table.

This design is highly **memory-efficient**, as the Python code only ever handles a list of file paths (strings). The actual loading and processing of the large Parquet files is delegated entirely to DuckDB's native engine.

## 5. Detailed Implementation Plan

### Step 1: Implement the UDTF in `pilates/utils/duckdb_manager.py`

The core logic will reside in a new Python function. This function will be responsible for finding and loading external Parquet data.

```python
# In pilates/utils/duckdb_manager.py

import duckdb
import pandas as pd

# The UDTF can be a standalone function.
def get_external_files_df(duckdb_connection, logical_table_name: str) -> pd.DataFrame:
    """
    A DuckDB Python UDTF to read and return data from external Parquet files.
    
    This function is called from a SQL VIEW. It queries the file_records table
    to find all external files for a given logical table, then uses DuckDB's
    own read_parquet function to load them efficiently.
    """
    # DuckDB passes arguments in a tuple, so we extract the first element.
    logical_table_name = logical_table_name[0]

    # Query the file_records table to find all relevant external Parquet files.
    # It is CRITICAL that `file_path` stores an absolute path.
    file_records = duckdb_connection.execute(
        """
        SELECT file_path 
        FROM file_records 
        WHERE logical_table_name = ? AND storage_location = 'external'
        """,
        [logical_table_name]
    ).fetchall()

    if not file_records:
        # EDGE CASE: No external files found.
        # To avoid errors, return an empty DataFrame with the correct schema.
        # We can get the schema by querying the corresponding 'uploaded_' table.
        return duckdb_connection.execute(f"SELECT * FROM uploaded_{logical_table_name} LIMIT 0").fetchdf()

    # Flatten the list of tuples into a simple list of paths.
    file_paths = [rec[0] for rec in file_records]

    # Construct and execute the final query using DuckDB's optimized reader.
    # The list of paths is passed as a literal to read_parquet.
    query = "SELECT * FROM read_parquet(?)"
    
    # fetchdf() is convenient as it returns a Pandas DataFrame.
    result_df = duckdb_connection.execute(query, [file_paths]).fetchdf()
    
    return result_df

```

### Step 2: Register the UDTF in `pilates/utils/duckdb_manager.py`

The Python function must be registered with each DuckDB connection so it can be called from SQL. The `DuckDBManager` is the perfect place to manage this.

```python
# In pilates/utils/duckdb_manager.py, inside the DuckDBManager class

class DuckDBManager:
    # ... existing __init__, __enter__, __exit__ methods ...

    def _get_connection(self):
        """Gets or creates a DuckDB connection and registers custom functions."""
        # ... your existing connection logic ...
        
        if not hasattr(self, 'conn') or self.conn.is_closed():
            self.conn = duckdb.connect(self.database_path, read_only=self.read_only)
            
            # --- Register the UDTF ---
            # This makes the python_get_hybrid_table function available in SQL.
            self.conn.create_function(
                "python_get_hybrid_table",  # SQL name for the function
                get_external_files_df,      # The Python function to call
                parameters=['VARCHAR'],     # List of SQL input types
                return_type='TABLE',        # The function returns a full table/relation
                python_udtf_init_parameters=['duckdb_connection'] # Special parameter to pass the connection context
            )
        return self.conn
    
    # ... other methods ...
```

### Step 3: Ensure Absolute Paths in `pilates/database/selective_uploader.py`

**Crucial for Robustness:** The system will fail if the file paths stored in the database are relative. The uploader script must be modified to resolve and store absolute paths. This ensures that DuckDB can find the files regardless of where the query is executed from.

```python
# In pilates/database/selective_uploader.py (conceptual change)

# Get the absolute path of the run_info.json file
run_info_path = Path(args.run_info_path).resolve()
# The directory containing run_info.json is our base for resolving relative paths
run_dir = run_info_path.parent

# Inside the loop that processes file records from run_info.json
for file_hash, file_record in run_info['file_records'].items():
    # ...
    relative_file_path = Path(file_record['file_path'])
    # Create an absolute path from the run directory and the relative path
    absolute_file_path = str(run_dir / relative_file_path)
    
    # Use 'absolute_file_path' when inserting or updating the file_records table.
    # ...
```

### Step 4: Generate Robust Views in `pilates/database/scripts/create_views.py`

To prevent `UNION` failures from mismatched column orders, the view generation script must be updated to use explicit column lists instead of `SELECT *`.

```python
# In pilates/database/scripts/create_views.py, update the VIEW_TEMPLATE

# You already have the logic to parse column names. Use it here.
# columns = ['run_id', 'file_record_id', 'household_id', 'hhsize', ...]

column_list_str = ",\n".join([f"    {quote_keyword(col)}" for col in columns])

VIEW_TEMPLATE = r"""-- View: {table_name}
CREATE OR REPLACE VIEW {table_name} AS
-- Query 1: Data stored directly in the database
SELECT
{column_list}
FROM uploaded_{table_name}

UNION ALL

-- Query 2: Data from external Parquet files, accessed via UDTF
SELECT
{column_list}
FROM python_get_hybrid_table('{table_name}');

COMMENT ON VIEW {table_name} IS 'Unified view for {table_name} data, combining uploaded and external sources via UDTF.';
"""

# The rest of the script uses this new template
view = VIEW_TEMPLATE.format(
    table_name=table_name,
    column_list=column_list_str
)
```

## 6. Updated Test Strategy

The existing test suite in `tests/test_hybrid_storage.py` is perfectly designed to validate this new implementation. No changes to the tests themselves should be necessary. The goal is for the tests to **pass** with the new UDTF-based implementation, proving that the abstraction is successful and the querying interface remains transparent.

## 7. Conclusion

By shifting from a pure-SQL view to a Python UDTF, we can effectively work around DuckDB's limitations while retaining all the benefits of a hybrid storage system. This plan provides a clear, robust, and efficient path forward. Once implemented, PILATES will have a powerful and flexible data management backend capable of handling diverse analytical workflows.