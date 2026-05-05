# Zone ID Management

This document describes the intended zone-ID contract in PILATES and the code
paths that are supposed to enforce it.

This topic is critical because a silent mismatch in zone ordering can corrupt:

- ActivitySim land-use inputs
- BEAM-generated skims
- UrbanSim skim lookups
- ATLAS accessibility calculations

The goal is to make the zone contract explicit enough that developers know what
must stay true and where to check it.

## The Core Invariants

PILATES is trying to maintain these invariants:

1. There is one authoritative zone definition file for a run.
2. That file defines the valid canonical zone IDs.
3. All model-facing zone tables are derived from the same canonical ordering.
4. Zarr skim coordinates used by ActivitySim should be 0-based contiguous
   integers.
5. The original canonical zone IDs should still be recoverable from skim
   metadata, not inferred from the 0-based coordinate values.

If any of these are violated, downstream model behavior can be subtly wrong
even when the run does not crash.

## Single Source Of Truth

The authoritative source is the configured canonical zone geometry file:

- `shared.geography.zones.source_file`

Required companion fields:

- `shared.geography.zones.canonical_id_col`
- `shared.geography.zones.activitysim_index_col`

Important nuance:

- `canonical_id_col` is the column in the source geometry file that defines the
  real zone identity.
- `activitysim_index_col` is the name PILATES uses for that indexed zone field
  downstream.
- In the current code, `activitysim_index_col` is mostly a field name, not a
  promise that the stored values themselves are already ActivitySim's internal
  0-based index.

## Authoritative Utility Functions

The main shared utility module is:

- `pilates/utils/zone_utils.py`

The important functions are:

### `load_canonical_zones(settings, workspace)`

This is the main source-of-truth loader.

Current behavior:

- resolves the configured canonical zone source
- reads the geometry file
- checks that `canonical_id_col` exists
- checks that the IDs are unique
- sets the canonical ID column as the index
- renames the index to `activitysim_index_col`
- sorts by that index
- casts the index to string

Result:

- downstream code gets a canonically ordered GeoDataFrame whose index values are
  the canonical zone IDs, represented as strings

### `get_block_to_zone_mapping(settings, year, workspace)`

This builds a block-to-zone mapping by spatial intersection against the
canonical zones.

That mapping is used by preprocessors that need to aggregate or assign
block-level data to the canonical zone system.

### `ensure_0_based_and_flag_zarr_skims(skim_path, settings, workspace)`

This is the helper that tries to normalize `skims.zarr` files into the form
expected by ActivitySim.

Intended role:

- make `otaz` and `dtaz` use 0-based contiguous coordinates
- add the `preprocessed=zero-based-contiguous` attribute

Current caveat:

- this helper is a normalizer, not a full correctness proof
- see the review findings below for the current gaps in that logic

## How The Contract Flows Through The Models

### ActivitySim Inputs

ActivitySim preprocessing uses the canonical zones in two main ways:

1. it establishes authoritative zone order through `load_canonical_zones(...)`
2. it builds a block-to-zone mapping through `get_block_to_zone_mapping(...)`

That mapping is then used to assign zone IDs to:

- blocks
- households
- persons
- jobs

The resulting `land_use.csv`, household table, and person table are supposed to
be aligned to the same canonical zone system.

Related code:

- `pilates/activitysim/preprocessor.py`

### ActivitySim Skims Cache

During ActivitySim compilation and later iterative runs, PILATES works with
`skims.zarr`.

The intended contract is:

- coordinate values in `otaz` and `dtaz` are 0-based contiguous integers
- original canonical zone IDs are preserved in metadata such as
  `original_zone_ids`

Related code:

- `pilates/activitysim/runner.py`
- `pilates/utils/zone_utils.py`

### BEAM Zone File

BEAM preprocessing generates a canonically sorted zone geometry file for BEAM to
consume.

Related code:

- `pilates/beam/preprocessor.py`

The intended idea is that BEAM sees the same zone system as the rest of the
workflow, and that any skim writer it uses preserves the original canonical
ordering in metadata.

### BEAM Skim Verification And Merge

BEAM postprocessing is where the workflow attempts to verify and merge new skim
artifacts back into the shared ActivitySim cache.

Related code:

- `pilates/beam/postprocessor.py`

The intended checks are:

- the skim artifact carries the original canonical zone ordering
- that ordering matches `load_canonical_zones(...)`
- the Zarr coordinates are 0-based contiguous before downstream reuse

### UrbanSim And ATLAS

UrbanSim and ATLAS also consume the canonical zone definitions rather than
inventing their own zone order.

Related code:

- `pilates/urbansim/preprocessor.py`
- `pilates/atlas/preprocessor.py`

Those paths use `load_canonical_zones(...)` and/or `get_block_to_zone_mapping(...)`
to align skim or accessibility calculations to the same zone system.

## What Developers Should Do

If you touch zone-sensitive code:

1. load the canonical zone order from `load_canonical_zones(...)`
2. do not invent a new zone ordering locally
3. do not assume "sorted somehow" is good enough
4. preserve original zone IDs when converting to 0-based skim coordinates
5. add validation before publishing a new skim artifact

For a preserved run workspace, use the operator diagnostic to check the emitted
canonical zone table and compare it with the run-local UrbanSim datastore when
one is present:

```bash
python scripts/verify_zone_ids.py /path/to/preserved/workspace
```

## Current Safeguards

The codebase currently has three important safeguards:

1. `load_canonical_zones(...)`
   validates that the canonical ID column exists and is unique
2. `verify_skim_zone_order(...)` in the BEAM postprocessor
   checks that BEAM skim metadata matches the canonical ordering
3. `ensure_0_based_and_flag_zarr_skims(...)`
   tries to normalize Zarr skim coordinates into ActivitySim-friendly form

These are useful, but they are not the same thing as a complete proof that the
zone contract is enforced everywhere.

## Review Findings

These are the main code smells or inconsistencies I found while checking this
doc against the implementation.

### 1. `ensure_0_based_and_flag_zarr_skims(...)` can mark malformed coordinates as preprocessed

File:

- `pilates/utils/zone_utils.py`

Issue:

- the function treats "first `otaz` value is not 1" as evidence that the skims
  are already 0-based
- it then adds `preprocessed=zero-based-contiguous` if missing
- it does not verify that the full coordinate sequence is actually
  `0..N-1`
- it does not validate `dtaz` independently before adding the flag

Why this matters:

- a malformed or partially shifted Zarr could be blessed as
  `zero-based-contiguous` even when it is not
- downstream code may then skip realignment because it trusts the flag

### 2. BEAM zone-file preparation can re-sort away from canonical order

File:

- `pilates/beam/preprocessor.py`

Issue:

- `_prepare_beam_zone_shapefile(...)` starts from `load_canonical_zones(...)`
  but then may sort again by `settings.beam.skim_zone_geoid_col`

Why this matters:

- if `skim_zone_geoid_col` is not the same ordering as the canonical ID index,
  the exported BEAM zone file no longer reflects the canonical order described
  in this doc
- that is exactly the kind of subtle reorder that can propagate into skim output

### 3. `verify_skim_zone_order(...)` validates `otaz` but not `dtaz`

File:

- `pilates/beam/postprocessor.py`

Issue:

- the current verification explicitly checks `otaz` coordinates against
  `np.arange(len(canonical_order))`
- it does not perform the same check for `dtaz`

Why this matters:

- the doc previously claimed both origin and destination coordinates were
  verified
- the code currently only proves that for `otaz`

### 4. One postprocessing path appears to call `load_canonical_zones(...)` with the wrong signature

File:

- `pilates/postprocessing/postprocessor.py`

Issue:

- `_add_geometry_to_events(...)` calls `load_canonical_zones(settings)` without
  the required `workspace` argument

Why this matters:

- if that path is exercised, it is likely to fail at runtime
- even if it is dead or rarely used, it suggests not all downstream consumers
  were updated when `load_canonical_zones(...)` became the required API

## Practical Recommendation

If this area is a persistent source of pain, the most valuable next hardening
steps are:

1. make `ensure_0_based_and_flag_zarr_skims(...)` validate full `otaz` and
   `dtaz` sequences before setting the preprocessed flag
2. remove or justify the second sort in `_prepare_beam_zone_shapefile(...)`
3. make `verify_skim_zone_order(...)` validate `dtaz` as well as `otaz`
4. add direct tests for malformed-but-not-1-based Zarr coordinates
5. clean up the stale `load_canonical_zones(settings)` call

## Related Docs

- `docs/land_use_skim_alignment.md`
- `docs/run/configuration_reference.md`
- `docs/run/troubleshooting.md`
