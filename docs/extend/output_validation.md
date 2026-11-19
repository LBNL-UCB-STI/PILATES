# Output Validation

PILATES validates model outputs at the end of every step before promoting artifacts to the scenario coupler or downstream stages.

## Validation Sequence

At the end of each step, `execute_step(...)` and the model's `Postprocessor` run four checks in order:

1. **Required coupler key presence**: `StepOutputsBase.assert_required(...)` verifies that every key listed in `required_outputs` exists on the coupler and resolves to a non-empty artifact reference.
2. **Filesystem path verification**: `StepOutputsBase.validate_paths(...)` checks that every path listed in `required_path_fields` exists on disk before Consist logs or archives the file.
3. **Tabular schema and primary-key checks**: For steps with registered SQLModel schemas in `pilates/database/schema/registry.py`, Consist and the postprocessor validate column names, data types, and non-null key constraints. See [Database and Schemas](../reference/database_and_schemas.md) for the schema registry.
4. **Domain-specific structural invariants**: Postprocessors verify cross-table referential integrity, zone coordinate bounds, and positive row counts on core outputs.

## Configuring Output Validation by Model

### UrbanSim

UrbanSim postprocessing checks that forecasted HDF5 and Parquet tables contain all required geographic and demographic keys (`block_id`, `zone_id`, `household_id`, `person_id`) and match canonical zone identifiers. See [Zone and Skim Alignment](../reference/zone_and_skim_alignment.md) for zone validation rules.

### ATLAS

ATLAS validation verifies that `households.csv` and `vehicles.csv` exist for the target simulation year, contain non-zero household counts when input populations are non-empty, and preserve `household_id` alignment with UrbanSim.

### ActivitySim

ActivitySim output validation checks both coupler artifact bindings and optional DuckDB ingestion schemas (`final_households`, `final_persons`, `final_tours`, `final_trips`, `final_land_use`). Enable strict schema enforcement during ingestion with `shared.database.strict_schema`:

```yaml
shared:
  database:
    enabled: true
    strict_schema: true
```

### BEAM

BEAM postprocessing verifies `FINAL_SKIMS_OMX` (or `ZARR_SKIMS`), `BEAM_LINKSTATS`, and `BEAM_EVENTS`. Skim validation confirms 0-based matrix coordinates and finite travel time values before the next replanning or forecast year begins.

## Adding Custom Validation to a New Model

Implement custom checks in two places:

1. **Path and key contracts on `StepOutputs`**: Define `required_outputs` and `required_path_fields` on your `StepOutputsBase` subclass in `pilates/<model>/outputs.py`. `execute_step(...)` runs `assert_required()` and `validate_paths()` after `scenario.run(...)` completes.
2. **Content checks in `Postprocessor`**: Raise a `ValueError` or `FileNotFoundError` inside `pilates/<model>/postprocessor.py` when an output table is empty, missing required columns, or violates domain constraints. Raising before coupler promotion keeps corrupt artifacts out of `.consist/provenance.duckdb` and prevents false cache hits on subsequent runs.
