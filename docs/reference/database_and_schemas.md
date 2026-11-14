# Database and Schema Reference

PILATES uses Consist and DuckDB (`.consist/provenance.duckdb` inside each archive run directory) to index run provenance, associate tabular outputs with SQLModel schemas, and expose hybrid SQL views over Parquet and CSV artifacts.

## Configuration Settings

Configure the shared provenance catalog under `shared.database` and run-local database behavior under `run` in `config.yaml`:

```yaml
shared:
  database:
    enabled: true
    type: duckdb
    path: /path/to/shared/provenance.duckdb

run:
  consist_db_local_run: true
  consist_db_lock_retries: 20
  consist_db_lock_wait_seconds: 0.5
```

| Config Key | Default | Description |
| --- | --- | --- |
| `shared.database.enabled` | `false` | Enables shared DuckDB catalog integration. |
| `shared.database.type` | `"duckdb"` | Database backend type. |
| `shared.database.path` | Required under `shared.database` | Shared DuckDB catalog path (used when `consist_db_local_run: false` or `consist_db_seed_from_shared_on_start: true`). |
| `shared.database.snapshot_path` | `null` | Optional matrix snapshot file path. |
| `run.consist_db_local_run` | `true` | Stores the active DuckDB provenance database in `.consist/<consist_db_filename>` inside the run directory. |
| `run.consist_db_lock_retries` | `20` | Number of retry attempts when acquiring the DuckDB file lock. |
| `run.consist_db_lock_wait_seconds` | `0.5` | Delay in seconds between DuckDB lock acquisition attempts. |

## Registered SQLModel Schemas

`pilates/database/schema/registry.py` maps coupler artifact keys to SQLModel classes via `get_consist_schemas()` and `get_schema_for_key(key)`.

### Exact Artifact Key Mappings

| Coupler / Artifact Key | SQLModel Schema Class | Module |
| --- | --- | --- |
| `land_use_asim_in` / `land_use_asim_out` | `LandUseAsimIn` / `LandUseAsimOut` | `pilates/database/schema/activitysim_schema.py` |
| `households_asim_in` / `households_asim_out` | `HouseholdsAsimIn` / `HouseholdsAsimOut` | `pilates/database/schema/activitysim_schema.py` |
| `persons_asim_in` / `persons_asim_out` | `PersonsAsimIn` / `PersonsAsimOut` | `pilates/database/schema/activitysim_schema.py` |
| `tours_asim_out` | `ToursAsimOut` | `pilates/database/schema/activitysim_schema.py` |
| `trips_asim_out` | `TripsAsimOut` | `pilates/database/schema/activitysim_schema.py` |
| `beam_plans_asim_out` | `BeamPlansAsimOut` | `pilates/database/schema/activitysim_schema.py` |
| `accessibility_asim_out` | `AccessibilityAsimOut` | `pilates/database/schema/activitysim_schema.py` |
| `linkstats` / `linkstats_warmstart` | `BeamLinkstats` | `pilates/database/schema/beam_schema.py` |
| `beam_network_final` | `BeamNetworkFinal` | `pilates/database/schema/beam_schema.py` |
| `beam_plans_out` | `BeamPlansOut` | `pilates/database/schema/beam_schema.py` |
| `atlas_households_csv`, `atlas_vehicles2_output` | `AtlasHousehold`, `VehiclesAtlasOut` | `pilates/database/schema/atlas_schema.py` |

### Prefix Families and Split BEAM Event Keys

`get_schema_for_key(key)` also resolves dynamic year-tagged and event-type-prefixed keys:

- **UrbanSim tables** (`urbansim_postprocess_usim_*_table_updated` / `*_archived`): Maps `households`, `persons`, `jobs`, `blocks`, `residential_units`, `taz_zone_geoms`, `work_locations`, and `graveyard` artifacts to schemas in `pilates/database/schema/urbansim_schema.py`.
- **ATLAS tables** (`householdv_<year>`, `vehicles_<year>`): Maps year-scoped ATLAS outputs to `HouseholdVAtlasOut` and `VehiclesAtlasOut` in `pilates/database/schema/atlas_schema.py`.
- **BEAM prefix families** (`linkstats_*`, `events_parquet_*`, `final_vehicles_*`, `path_traversal_links_*`, `route_history_*`): Maps versioned BEAM outputs to schemas in `pilates/database/schema/beam_schema.py`.
- **Split BEAM event tables** (`events_parquet_<EventType>`): Maps per-event-type tables (such as `events_parquet_PathTraversal`, `events_parquet_ModeChoice`, `events_parquet_PersonEntersVehicle`, `events_parquet_actstart`, `events_parquet_actend`, `events_parquet_RefuelSessionEvent`, and `events_parquet_ChargingPlugInEvent`) to specialized SQLModel classes in `pilates/database/schema/beam_schema.py`.

## Generating Schema ERDs and Reference Diagrams

Run `pilates/database/scripts/generate_schema_erd.py` to generate Entity-Relationship Diagrams and column tables from the SQLModel definitions in `pilates/database/schema/`:

```bash
# Generate a Mermaid ERD block
python pilates/database/scripts/generate_schema_erd.py --format mermaid

# Generate Graphviz DOT or standalone HTML documentation
python pilates/database/scripts/generate_schema_erd.py --format dot -o schema.dot
python pilates/database/scripts/generate_schema_erd.py --format html -o schema.html
```

For querying these tables with Python or the CLI, see [SQL and DuckDB](../analysis/sql_and_duckdb.md).
