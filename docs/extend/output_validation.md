# Output Validation

PILATES validates model outputs at the end of every step before promoting artifacts to `scenario.coupler` or downstream stages.

## Validation Sequence

At the end of each step, `project_outputs` and the model's `Postprocessor` run three checks in order:

1. **Filesystem path verification**: `StepOutputsBase.validate(context)` checks that every path in `required_path_fields`, every non-`None` path in `optional_path_fields`, and every `Path` value in `dict_path_fields` exists on disk.
2. **Semantic `OutputValidator` checks**: `StepOutputsBase.validate(context)` runs each `OutputValidator` declared in `validators` (for example, `_BeamPostprocessExpectedOutputsValidator` in `pilates/beam/outputs.py`, which requires `zarr_skims` when ActivitySim is active and `final_skims_omx` when UrbanSim, ATLAS, or `write_skims_to_omx` is enabled).
3. **Domain invariants and schemas**: Postprocessors verify cross-table referential integrity, zone coordinate bounds, and positive row counts on core outputs. Registered SQLModel schemas in `pilates/database/schema/registry.py` define column types and keys for tabular artifacts (see [Database and Schemas](../reference/database_and_schemas.md)).

## Validation by Model

- **UrbanSim**: Checks that forecasted HDF5 and Parquet tables contain required geographic and demographic keys (`block_id`, `zone_id`, `household_id`, `person_id`) and match canonical zone IDs. See [Zone and Skim Alignment](../reference/zone_and_skim_alignment.md).
- **ATLAS**: Verifies that `households` and `vehicles` CSVs exist for the target simulation year, contain non-zero household counts when input populations are non-empty, and preserve `household_id` alignment with UrbanSim.
- **ActivitySim**: Validates `households_asim_out`, `persons_asim_out`, `tours_asim_out`, `trips_asim_out`, `beam_plans_asim_out`, and `accessibility_asim_out` against expected paths and schemas.
- **BEAM**: Verifies `final_skims_omx` (or `zarr_skims`), `linkstats`, and `events_parquet_*` outputs. Skim validation confirms 0-based matrix coordinates and finite travel time values before the next replanning iteration or forecast year begins.

## Adding Custom Validation to a New Model

Implement custom checks in two places:

1. **Path fields and validators on `StepOutputsBase`**: Define `required_path_fields`, `optional_path_fields`, `dict_path_fields`, and `validators` on your `StepOutputsBase` subclass in `pilates/<model>/outputs.py`. Call `outputs.validate(ValidationContext(...))` inside `project_outputs` or your step body.
2. **Content checks in `Postprocessor`**: Raise `ValueError` or `FileNotFoundError` inside `pilates/<model>/postprocessor.py` when an output table is empty, missing required columns, or violates domain constraints. Failing before coupler publication keeps corrupt artifacts out of `.consist/provenance.duckdb` and prevents false cache hits on subsequent runs.
