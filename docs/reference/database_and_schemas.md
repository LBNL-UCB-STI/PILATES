# Database and Schema Reference

PILATES uses Consist and DuckDB (`<output_directory>/.consist/provenance.duckdb`) to index run provenance, validate tabular outputs against SQLModel schemas, and expose hybrid SQL views over Parquet and CSV artifacts.

## Configuration Settings

Configure DuckDB ingestion and schema validation under `shared.database` and `run` in `config.yaml`:

```yaml
shared:
  database:
    enabled: true
    strict_schema: false
    ingest_outputs: true

run:
  consist_db_lock_retries: 20
  consist_db_lock_wait_seconds: 0.5
```

| Config Key | Default | Description |
| --- | --- | --- |
| `shared.database.enabled` | `true` | Enables DuckDB table ingestion and hybrid view creation during workflow execution. |
| `shared.database.strict_schema` | `false` | Enforces strict SQLModel column and type validation during `dlt` ingestion; when `false`, Consist infers extra columns at runtime. |
| `shared.database.ingest_outputs` | `true` | Materializes registered step outputs into `.consist/provenance.duckdb`. |
| `run.consist_db_lock_retries` | `20` | Number of retry attempts when acquiring the DuckDB file lock. |
| `run.consist_db_lock_wait_seconds` | `0.5` | Delay in seconds between DuckDB lock acquisition attempts. |

## Registered SQLModel Schemas

`pilates/database/schema/registry.py` maps coupler artifact keys to SQLModel classes via `get_consist_schemas()` and `get_schema_for_key(key)`.

### Exact Artifact Key Mappings

| Coupler / Artifact Key | SQLModel Schema Class | Module | Primary Keys / Identifiers |
| --- | --- | --- | --- |
| `final_land_use` | `ActivitySimLandUse` | `pilates/database/schema/activitysim.py` | `zone_id` |
| `final_households` | `ActivitySimHouseholds` | `pilates/database/schema/activitysim.py` | `household_id` |
| `final_persons` | `ActivitySimPersons` | `pilates/database/schema/activitysim.py` | `person_id`, `household_id` |
| `final_tours` | `ActivitySimTours` | `pilates/database/schema/activitysim.py` | `tour_id`, `person_id`, `household_id` |
| `final_trips` | `ActivitySimTrips` | `pilates/database/schema/activitysim.py` | `trip_id`, `tour_id`, `person_id`, `household_id` |
| `beam_linkstats` | `BeamLinkStats` | `pilates/database/schema/beam.py` | `link`, `hour` |
| `beam_events` | `BeamEvents` | `pilates/database/schema/beam.py` | `time`, `type`, `person`, `vehicle` |

### Prefix Families and Split BEAM Event Keys

`get_schema_for_key(key)` also resolves dynamic year-tagged and event-type-prefixed keys:

- **UrbanSim tables** (`urbansim_*` and `<year>_urbansim_*`): Maps `households`, `persons`, `jobs`, `buildings`, `blocks`, and `zones` artifacts to the corresponding schemas in `pilates/database/schema/urbansim.py`.
- **ATLAS tables** (`atlas_households*`, `atlas_vehicles*`): Maps vehicle ownership outputs to `AtlasHouseholds` and `AtlasVehicles` in `pilates/database/schema/atlas.py`.
- **Split BEAM event tables** (`beam_events_<EventType>`): Maps per-event-type tables (such as `beam_events_PathTraversal`, `beam_events_ModeChoice`, `beam_events_PersonEntersVehicle`, `beam_events_actstart`, `beam_events_actend`, `beam_events_RefuelSessionEvent`, and `beam_events_ChargingPlugInEvent`) to specialized SQLModel classes in `pilates/database/schema/beam.py`.

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
