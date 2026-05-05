---
title: Datasets
summary: Dataset-oriented analysis surfaces for linkstats, skims, ActivitySim trips, and exported bundles.
---

# Datasets

## Adjacent Pages

- Read [Opening Archives](opening_archives.md) first.
- Use [Consist Analysis CLI](consist_analysis_cli.md) for command entrypoints.
- Pair this with [Artifact Semantics](../workflow/artifact_semantics.md).

## Linkstats

`build_linkstats_dataset()` returns a `LinkstatsDataset` with:

- `artifacts`
- `summary`
- `deltas`

It discovers BEAM linkstats artifacts through `pilates.utils.consist_analysis.find_linkstats_artifacts()`, then summarizes grouped views and delta rows.
The builder accepts year, iteration, artifact family, namespace, grouped-mode, grouped-missing-files, grouped-schema-id, traveltime weighting, and row limit filters.

`write_linkstats_dataset()` writes:

- `linkstats_artifacts.csv`
- `linkstats_summary.csv`
- `linkstats_deltas.csv`
- `dataset_manifest.json`

## ActivitySim Trips

`build_activitysim_trips_dataset()` returns an `ActivitySimTripsDataset` with:

- `artifacts`
- `mode_counts`
- `purpose_mode_counts`
- `depart_hour_counts`
- `iteration_summary`
- `mode_deltas`
- `equilibrium_pairs`

It discovers ActivitySim trip artifacts, builds grouped summaries, and can keep only the latest artifact per iteration.

`write_activitysim_trips_dataset()` writes:

- `asim_trips_mode_counts.csv`
- `asim_trips_purpose_mode_counts.csv`
- `asim_trips_depart_hour_counts.csv`
- `asim_trips_iteration_summary.csv`
- `asim_trips_mode_deltas.csv`
- `asim_trips_equilibrium_pairs.csv`
- `dataset_manifest.json`

## Skims

`build_skim_convergence_dataset()` returns a `SkimConvergenceDataset` with:

- `artifacts`
- `matrices`
- `summary`
- `deltas`

It discovers OpenMatrix concept keys from the tracker when keys are not supplied, then builds matrix-level, summary-level, and iteration-delta frames.

`write_skim_convergence_dataset()` writes:

- `skim_artifacts.csv`
- `skim_matrices.csv`
- `skim_summary.csv`
- `skim_deltas.csv`
- `dataset_manifest.json`

## When To Use These Builders

Use the dataset builders when you want a packaged CSV and manifest surface.
Use `Epoch.sql()` or `export_sql_query()` when you want a direct query result.
Use `ScenarioComparison` when you want paired-run comparison frames instead of one-sided summaries.
