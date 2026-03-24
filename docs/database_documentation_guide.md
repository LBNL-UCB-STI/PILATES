# Database Documentation Guide (Schema + Run DB Inspection)

This guide covers two distinct things:

1. How to generate a schema diagram from PILATES' curated Consist schemas
2. How to inspect a run's Consist DuckDB (`.consist/*.duckdb`) for debugging and analysis

PILATES currently has a reliable ERD generator in-tree. Other "data dictionary"
export tooling may exist on some branches, but in this checkout the shell helper
`./export_database_docs.sh` references a CLI module that is not present under
`pilates/utils/`. Treat that as a known gap unless you add/restore that tool.

## Source Of Truth

### Curated Schema Stubs

PILATES keeps curated SQLModel table definitions under:

- `pilates/database/schema/`
- `pilates/database/schema/registry.py` (schema registration / selection)

These stubs are what the ERD generator reads. They are designed to represent
the schema Consist uses for artifact logging and related metadata.

### Live Run Database

Each run writes a DuckDB file under the run directory:

- `<run_dir>/.consist/<run.consist_db_filename>` (often `provenance.duckdb`)

The live DB is the authoritative record for that specific run.

## Generate ERDs (Recommended)

The primary supported documentation artifact right now is an ERD generated from
the curated schema stubs.

Script:

- `pilates/database/scripts/generate_schema_erd.py`

### Mermaid ERD (Best For Static Sites)

Mermaid is a good default because:

- it is plain text
- GitHub can render it
- static site builders can render it without shipping extra JS assets

```bash
python pilates/database/scripts/generate_schema_erd.py \
  --format mermaid \
  --output docs/database/pilates_schema_erd.mmd
```

Options:

- `--no-columns` to show only table relationships
- `--stdout` to print instead of writing

### Graphviz DOT (Optional)

```bash
python pilates/database/scripts/generate_schema_erd.py \
  --format dot \
  --output docs/database/pilates_schema_erd.dot
```

If you have Graphviz installed, you can render directly from the script:

```bash
python pilates/database/scripts/generate_schema_erd.py \
  --format dot \
  --output docs/database/pilates_schema_erd.dot \
  --render svg
```

### Interactive HTML ERD (Optional)

```bash
python pilates/database/scripts/generate_schema_erd.py \
  --format html \
  --output docs/database/pilates_schema_erd.html
```

Important:

- The generated HTML expects Cytoscape to be present at
  `docs/diagrams/node_modules/cytoscape/dist/cytoscape.min.js`.
- In this repository, the vendored `node_modules/` currently lives under
  `docs-internal/diagrams/node_modules/` (not `docs/diagrams/`), so the HTML ERD
  will not render unless you provide those assets at the expected path.

Practical approaches:

1. For a static site, prefer the Mermaid ERD.
2. For local interactive viewing, create `docs/diagrams/` and supply the assets
   (copy or symlink) so the HTML can load Cytoscape.

## Inspect A Run's DuckDB

### Locate The DB

In a run directory, look for:

- `.consist/provenance.duckdb` (or whatever `run.consist_db_filename` is set to)

Related sidecars:

- `.consist/provenance.duckdb.wal`
- `.consist/snapshots/`

### Quick Health Check (Python)

```bash
python - <<'PY'
from pilates.utils.consist_analysis import print_duckdb_health
print_duckdb_health(db_path="/path/to/run/.consist/provenance.duckdb", probe_open=True)
PY
```

### Explore With DuckDB CLI (If Installed)

```bash
duckdb /path/to/run/.consist/provenance.duckdb
```

Useful commands and queries that are safe across schemas:

```sql
SHOW TABLES;

-- System catalog views for structure discovery
SELECT * FROM duckdb_tables() ORDER BY table_name;
SELECT * FROM duckdb_columns() ORDER BY table_name, column_index;
```

If the Consist schema is present (typical), you can also sanity-check for core
tables that PILATES expects Consist to manage:

```sql
-- This table is referenced by PILATES compatibility logic.
SELECT COUNT(*) FROM artifact;
```

If that query fails, do not guess the schema. Use `SHOW TABLES;` and inspect the
tables that are actually present for your Consist version.

## "Data Dictionary" Exports

This checkout contains `./export_database_docs.sh`, but it currently references
`pilates/utils/export_data_dictionary.py`, which is not present in-tree. As a
result, the script will fail unless you restore that CLI or update the script.

Until a supported export tool is reintroduced, the recommended path is:

1. Generate an ERD from curated schemas (Mermaid).
2. Use DuckDB introspection (`duckdb_tables()`, `duckdb_columns()`) to extract
   table/column lists for a specific run DB.

Example: dump a "schema index" to stdout:

```bash
duckdb /path/to/run/.consist/provenance.duckdb -c \
  "SELECT table_name, column_name, data_type FROM duckdb_columns() ORDER BY table_name, column_index;"
```

## Keeping Docs Current

- Regenerate ERDs whenever `pilates/database/schema/` or the schema registry changes.
- Treat generated ERDs as derived artifacts. For a docs site, prefer committing:
  - `docs/database/pilates_schema_erd.mmd`
  - optionally a rendered SVG/PNG if your site pipeline prefers images
- When debugging a specific run, always trust the live `.consist/*.duckdb` file
  over the curated schema stubs. The stubs are documentation and validation
  aids, not a guarantee of the exact tables present in any given Consist version.

## Related Docs

- `docs/database-setup.md`
- `docs/workflow_primer.md`
- `docs/test_output_preservation.md`
- `docs/lineage_map.md`
