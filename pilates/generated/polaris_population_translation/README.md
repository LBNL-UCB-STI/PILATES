# POLARIS Translation SQL

This directory is a first-pass SQL translation package for turning the
post-ATLAS population export into POLARIS-style demand files.

Execution order:
1. `00_inputs.sql`
2. `10_households.sql`
3. `20_persons.sql`
4. `30_vehicle_dimensions.sql`
5. `40_vehicles.sql`
6. `50_vehicle_class.sql`
7. `60_vehicle_type.sql`
8. `99_exports.sql`

Design notes:
- Each file owns one logical translation step.
- Intermediate views are named for the concept they document.
- Source substitutions should happen in `00_inputs.sql`.
- Mapping substitutions should happen in the file that owns that target table.

Current assumptions:
- `location` and `school_location_id` are temporarily set from `block_id`.
- Household `type` follows the same one-vs-many rule currently used by the
  ActivitySim preprocessor: `persons == 1 -> 1`, otherwise `4`.
- ATLAS adopt lookup files are available for the export year and scenario.
- `housing_unit_type` remains best-effort until a cleaner building/unit join is added.
