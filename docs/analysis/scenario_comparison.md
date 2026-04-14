---
title: Scenario Comparison
summary: Comparing scenario outputs, aligned runs, and summarized differences across runsets.
---

# Scenario Comparison

## Adjacent Pages

- Read [Run Discovery and Runsets](run_discovery_and_runsets.md) first.
- Pair this with [Datasets](datasets.md).
- Use [Analysis Patterns](analysis_patterns.md) for question-driven workflows.

## Comparison Objects

`compare_scenarios()` builds a `ScenarioComparison` object with:

- `left_name` and `right_name`
- `left_run_ids` and `right_run_ids`
- `aligned_on`
- `aligned_keys`
- `config_diff`
- `dataset_summaries`
- `dataset_frames`

The higher-level `Comparison` wrapper adds convenience accessors for:

- `summary()`
- `dataset_summaries()`
- `config_diff()`
- `frame(dataset)`
- `linkstats_summary()`
- `asim_iteration_summary()`
- `skims_summary()`
- `mode_shares()`
- `trip_purposes()`

## Alignment

The comparison path first aligns the selected run sets.
It can:

- use `align_on` to choose the key used for pairing
- call `latest(group_by=...)` before alignment
- call `converged(group_by=...)` before alignment when `use_converged=True`

If converged comparison is requested but one side does not have complete epoch candidates for the overlapping alignment keys, the code raises a `ValueError`.
That failure is intentional and pinned by tests.

## Dataset Surfaces

The comparison builder compares these dataset families:

- `linkstats`
- `asim_trips`
- `skims`

For each dataset it builds:

- a per-aligned-pair comparison frame
- a dataset summary row with row counts, overlap counts, and delta aggregates

It also builds a config-diff frame from the tracker query service when available.

`write_scenario_comparison()` writes:

- `scenario_comparison_summary.csv`
- `scenario_comparison_config_diff.csv`
- one CSV per dataset family, such as `scenario_linkstats_comparison.csv`
- `dataset_manifest.json`
