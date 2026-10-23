# SQL and DuckDB

Each PILATES run archive maintains a DuckDB database at `<archive>/.consist/provenance.duckdb` containing run metadata, artifact lineage, and materialized simulation tables.

## Inspecting Provenance with the CLI

### Consist CLI

Use the `consist` CLI to inspect runs, artifacts, schemas, and lineage from a `.consist/provenance.duckdb` file:

```bash
# List recorded workflow steps and cache status
consist runs --db outputs/my_run/.consist/provenance.duckdb

# List logged input and output artifacts
consist artifacts --db outputs/my_run/.consist/provenance.duckdb

# Preview rows from a logged tabular artifact
consist preview final_trips --db outputs/my_run/.consist/provenance.duckdb --limit 10

# Print upstream artifact and step lineage
consist lineage final_trips --db outputs/my_run/.consist/provenance.duckdb

# Validate artifact paths and database integrity
consist validate --db outputs/my_run/.consist/provenance.duckdb

# List registered hybrid views
consist views --db outputs/my_run/.consist/provenance.duckdb
```

### `pilates-consist-analysis` CLI

Use the `pilates-consist-analysis` command to scan multiple output directories and export summary tables:

```bash
# Discover runs across output roots
pilates-consist-analysis discover-runs outputs/

# Export a multi-year epoch panel to CSV or Parquet
pilates-consist-analysis epoch-panel outputs/ --out panel.parquet

# Check DuckDB database health and run facet tagging
pilates-consist-analysis db-health outputs/
pilates-consist-analysis run-tagging outputs/
```

## Querying DuckDB from Python

### Using `Archive` and `Epoch` SQL Helpers

`Archive` and `Epoch` expose SQL execution helpers bound to the archive's DuckDB store and registered hybrid views:

```python
from pilates_consist_analysis import Archive

archive = Archive.from_dir("outputs/my_run")

# Load a registered table for a specific forecast year
trips_2020 = archive.table("final_trips", year=2020)

# Run custom SQL scoped to a single forecast epoch
epoch_2020 = archive.epoch(year=2020)
mode_counts = epoch_2020.sql(
    """
    SELECT trip_mode, COUNT(*) AS trips
    FROM final_trips
    GROUP BY trip_mode
    ORDER BY trips DESC
    """
)
```

### Using `consist.run_query` and Direct DuckDB Connections

For cross-run joins or custom analytical queries over cold Parquet files and hot `dlt` tables, use `consist.run_query(...)` or open a read-only DuckDB connection:

```python
import consist
import duckdb

db_path = "outputs/my_run/.consist/provenance.duckdb"

# Execute a query through Consist's hybrid view layer
df = consist.run_query(
    """
    SELECT r.id AS run_id, r.model_name, r.status, a.key AS artifact_key, a.driver
    FROM run r
    JOIN artifact a ON a.run_id = r.id
    ORDER BY r.started_at
    """,
    db_path=db_path,
)

# Or connect with the DuckDB Python client in read-only mode
with duckdb.connect(db_path, read_only=True) as con:
    tables = con.sql("SHOW ALL TABLES").df()
```

See [Database and Schemas](../reference/database_and_schemas.md) for the list of ingested table schemas and column definitions.
