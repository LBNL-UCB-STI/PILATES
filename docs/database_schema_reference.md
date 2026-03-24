# Database Schema Reference

This page is the concise reference for the database schema surfaces that matter
most in this repository.

There are two related but different schema views:

1. curated schema stubs in `pilates/database/schema/`
2. the live run database written under `.consist/*.duckdb`

The curated schema is useful for documentation and ERDs. The live run database
is the source of truth for an actual run.

## Curated Schema Source

The curated schema registry lives in:

- `pilates/database/schema/registry.py`

That registry maps workflow artifact keys to SQLModel schema classes and
provides `get_consist_schemas()` for schema registration.

The main schema modules are:

- `pilates/database/schema/activitysim_schema.py`
- `pilates/database/schema/atlas_schema.py`
- `pilates/database/schema/beam_schema.py`
- `pilates/database/schema/urbansim_schema.py`

## Main Curated Schema Families

### ActivitySim

Representative curated tables include:

- `households_asim_in`
- `land_use_asim_in`
- `persons_asim_in`
- `households_asim_out`
- `persons_asim_out`
- `tours_asim_out`
- `trips_asim_out`
- `beam_plans_asim_out`
- `accessibility_asim_out`
- `joint_tour_participants_asim_out`

These cover the main ActivitySim input and output tables that PILATES treats as
workflow-facing artifacts.

### BEAM

Representative curated tables include:

- `plans_beam_in`
- `households_beam_in`
- `persons_beam_in`
- `vehicles_beam_in`
- `beam_plans_out`
- `linkstats`
- `beam_network_final`
- `events_parquet_*`
- `path_traversal_links_*`
- `route_history_*`

Some BEAM schemas are exact-key mappings and some are prefix-based families.

### ATLAS

Representative curated tables include:

- `atlas_blocks_csv`
- `atlas_grave_csv`
- `atlas_households_csv`
- `atlas_jobs_csv`
- `atlas_persons_csv`
- `atlas_residential_csv`
- `atlas_vehicles2_input`
- `atlas_vehicles2_output`
- year-scoped output families such as `householdv_*` and `vehicles_*`

### UrbanSim

The curated UrbanSim surface in this registry is narrower and mostly focused on
workflow-relevant updated tables, such as:

- `activitysim_postprocess_usim_households_table_updated`
- `activitysim_postprocess_usim_persons_table_updated`

## Live Run Database

Each run typically writes a DuckDB file under:

- `<run_dir>/.consist/<run.consist_db_filename>`

Often this is:

- `<run_dir>/.consist/provenance.duckdb`

The live database may differ from the curated schema docs depending on:

- Consist version
- enabled models
- which steps actually ran
- whether some tables are present only as logged artifacts rather than as
  curated schema-backed views

When in doubt, trust the live DB over the curated docs.

## Safe Introspection Queries

If you have DuckDB installed:

```bash
duckdb /path/to/run/.consist/provenance.duckdb -c "SHOW TABLES;"
```

Useful structure discovery:

```bash
duckdb /path/to/run/.consist/provenance.duckdb -c \
  "SELECT * FROM duckdb_tables() ORDER BY table_name;"
```

```bash
duckdb /path/to/run/.consist/provenance.duckdb -c \
  "SELECT table_name, column_name, data_type FROM duckdb_columns() ORDER BY table_name, column_index;"
```

If you do not have the CLI, you can still run health checks from Python:

```bash
python - <<'PY'
from pilates.utils.consist_analysis import print_duckdb_health
print_duckdb_health(db_path="/path/to/run/.consist/provenance.duckdb", probe_open=True)
PY
```

## Generated Schema Docs In This Repo

This checkout also contains generated reference artifacts under:

- `docs-internal/database/database_schema.md`
- `docs-internal/database/database_schema.html`
- `docs-internal/database/database_schema.json`
- `docs-internal/database/database_schema.csv`
- `docs-internal/database/pilates_schema_erd.*`

Treat those as derived documentation snapshots, not as guarantees about every
live run database.

## Current Limitation

The helper script:

```bash
./export_database_docs.sh
```

currently references `pilates/utils/export_data_dictionary.py`, which is not
present in this checkout. So the ERD/introspection workflow is more reliable
than the shell export helper right now.

## Related Docs

- `docs/database-setup.md`
- `docs/database_documentation_guide.md`
- `docs/provenance_and_lineage.md`
