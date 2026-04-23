---
title: Run Discovery and Runsets
summary: Discovering runs, building run indexes, grouping runsets, and reasoning about epochs.
---

# Run Discovery and Runsets

## Adjacent Pages

- Read [Analysis Overview](overview.md) first.
- Then use [Scenario Comparison](scenario_comparison.md) or [Datasets](datasets.md).
- Pair this with [Glossary](../reference/glossary.md) for shared terms.

## Run Index

`RunIndex.build()` scans the tracker for runs, normalizes the key fields, and returns a DataFrame with these columns:

- `run_id`
- `parent_run_id`
- `name`
- `status`
- `scenario_id`
- `year`
- `iteration`
- `model`
- `seed`
- `created_at`
- `ended_at`
- `is_complete`
- `is_completed_status`
- `is_converged_candidate`
- `has_parent`
- `archive_run_dir`

The index also records which source supplied each normalized field.
That source usage is part of the object, not just a convenience field, so the docs should treat it as a real output of the discovery path.

`RunIndex` exposes filtered views over the same frame:

- `scenarios()`
- `years()`
- `models()`
- `filter(...)`

## Run Sets

`RunSet` extends Consist run-set behavior with PILATES-specific helpers:

- `filter(...)`
- `latest(group_by=...)`
- `split_by(field)`
- `align(other, on=...)`
- `converged(group_by=...)`

The `converged()` helper keeps only completed runs with the maximum iteration per grouping key.
The default grouping is `year` plus `scenario_id` when no group is passed.
Runs missing iteration are skipped and warned about.

## Epochs

`build_epoch_panel()` groups runs by year, iteration, scenario ID, and model.
It returns an `EpochPanel` containing `SimulationEpoch` objects.

`SimulationEpoch` tracks:

- `year`
- `outer_iteration`
- `scenario_id`
- `runs` by model

`EpochPanel` exposes:

- `years()`
- `converged_epochs()`
- `epoch(year)`
- `to_frame()`

`converged_epochs()` keeps the latest complete epoch per year and scenario.
An epoch is complete only when every model run in that epoch is completed.

## Selection Behavior

The code filters out runs missing year, iteration, or model when building epochs.
It warns rather than inventing values.
That is the contract the docs should preserve.
