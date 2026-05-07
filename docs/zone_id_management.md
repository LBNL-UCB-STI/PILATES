# Zone ID Management

PILATES keeps model-facing zone data aligned to one canonical zone system for a
run. That contract matters because ActivitySim land-use inputs, BEAM skims,
UrbanSim skim lookups, and ATLAS accessibility calculations can all be wrong if
two stages silently use different zone orderings.

## Public Contract

Every run should use one configured canonical zone geometry file:

- `shared.geography.zones.source_file`
- `shared.geography.zones.canonical_id_col`
- `shared.geography.zones.activitysim_index_col`

`canonical_id_col` names the real zone identifier in the source geometry.
`activitysim_index_col` names the downstream indexed zone field. Model adapters
should load the canonical order from the shared utility instead of sorting or
constructing zone IDs locally.

The stable invariants are:

1. the configured source file defines the valid canonical zone IDs
2. canonical zone IDs are unique
3. model-facing zone tables are derived from the same canonical ordering
4. ActivitySim-facing Zarr skim coordinates are 0-based contiguous integers
5. original canonical zone IDs remain recoverable from skim metadata

## Shared Utilities

The shared zone helpers live in `pilates/utils/zone_utils.py`.

`load_canonical_zones(settings, workspace)` is the source-of-truth loader. It
resolves the configured geometry file, checks that the canonical ID column
exists, checks uniqueness, indexes by the canonical IDs, applies the configured
index name, sorts by that index, and returns IDs as strings.

`get_block_to_zone_mapping(settings, year, workspace)` builds block-to-zone
assignments against the canonical zones for preprocessors that aggregate or
assign block-level data.

`ensure_0_based_and_flag_zarr_skims(skim_path, settings, workspace)` is the
current ActivitySim skim preparation helper. It records the `preprocessed`
metadata flag when it treats a Zarr skim cache as ready for downstream reuse.
It is an operational guardrail, not a complete proof that every origin and
destination coordinate is valid.

## Model Flow

ActivitySim preprocessing uses the canonical zones and block-to-zone mapping to
align `land_use.csv`, households, persons, and jobs to the same zone system.

BEAM preprocessing writes a zone file from the canonical zones, and BEAM
postprocessing verifies skim zone metadata before merging generated skims back
into the shared ActivitySim cache.

UrbanSim and ATLAS consume the canonical zone definitions for skim and
accessibility calculations rather than establishing independent zone orders.

## Current Safeguards

The codebase has these safeguards in the active workflow:

- `load_canonical_zones(...)` fails if the configured canonical ID column is
  missing or non-unique
- ActivitySim and related preprocessors use the shared loader rather than
  treating local table order as authoritative
- BEAM skim postprocessing checks the original canonical ordering recorded in
  skim metadata against the configured canonical zones
- skim preparation records metadata on ActivitySim-facing Zarr caches so later
  stages can tell whether PILATES has already processed the cache
- `scripts/verify_zone_ids.py` provides an operator diagnostic for preserved
  workspaces

These safeguards are intended to catch common drift and ordering problems. They
do not remove the need to preserve canonical ordering when adding new model
handoffs or skim-writing paths.

## Developer Rules

When touching zone-sensitive code:

1. call `load_canonical_zones(settings, workspace)` for authoritative order
2. do not infer zone order from arbitrary file, dataframe, or database order
3. do not add a second sort unless it is explicitly proven to preserve canonical
   order
4. preserve original zone IDs when converting skims to 0-based coordinates
5. validate skim metadata before publishing or reusing skim artifacts
6. keep new diagnostics additive and local to the handoff being checked

## Troubleshooting

If a preserved workspace has a root `canonical_zones.csv` with `zone_key` and
`asim_id` columns, run the preserved-workspace diagnostic first:

```bash
python scripts/verify_zone_ids.py /path/to/preserved/workspace
```

If the canonical file lives somewhere else in the preserved workspace, pass it
explicitly:

```bash
python scripts/verify_zone_ids.py /path/to/preserved/workspace \
  --canonical path/within/workspace/canonical_zones.csv
```

Use it when:

- ActivitySim crashes or changes behavior around `land_use.csv`, households,
  persons, jobs, or skim lookup
- BEAM-generated skims merge successfully but downstream results look shifted
- UrbanSim or ATLAS accessibility results appear inconsistent with the expected
  geography
- a run was restarted or rehydrated from archived inputs and the zone source is
  uncertain

If the diagnostic finds a mismatch, check the configured
`shared.geography.zones.*` fields, the preserved canonical zone file, and the
skim metadata before changing model-specific preprocessing code.

## Related Docs

- `docs/land_use_skim_alignment.md`
- `docs/run/configuration_reference.md`
- `docs/run/troubleshooting.md`
