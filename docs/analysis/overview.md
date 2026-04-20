---
title: Analysis Overview
summary: What post-run analysis means in PILATES and which analysis surfaces exist.
---

# Analysis Overview

## Reading Path

- Continue to [Opening Archives](opening_archives.md).
- If you want concrete scripts first, go to [Consist in Action](consist_in_action.md).
- Then choose [Consist Analysis CLI](consist_analysis_cli.md), [Run Discovery and Runsets](run_discovery_and_runsets.md), or [SQL and DuckDB](sql_and_duckdb.md).
- For direct scenario comparisons, go to [Scenario Comparison](scenario_comparison.md).
- For packaged dataset outputs, go to [Datasets](datasets.md).

## Public Surface

PILATES exposes analysis through two layers:

- The Python API, centered on `AnalysisSession`, `Archive`, `RunIndex`, `RunSet`, `EpochPanel`, `Epoch`, and `Comparison`.
- The CLI, which wraps the same analysis helpers for discovery, dataset building, inspection, export, and comparison.

The analysis package reads archived runs through a Consist tracker, then builds these surfaces:

- a run index over discovered runs and their source metadata
- run sets for filtering, grouping, latest-selection, and alignment
- epoch panels that group runs by year, iteration, scenario, and model
- dataset frames for linkstats, ActivitySim trips, and skim convergence
- scenario comparison frames and summary manifests
- SQL and DuckDB export helpers for ad hoc inspection

## Start Here

If you want to inspect an archived run, start with [Opening Archives](opening_archives.md).
If you want small read-only examples, start with [Consist in Action](consist_in_action.md).
If you want the command list, start with [Consist Analysis CLI](consist_analysis_cli.md).
If you want to understand run grouping first, start with [Run Discovery and Runsets](run_discovery_and_runsets.md).
If you want a one-off SQL-style inspection path, go straight to [SQL and DuckDB](sql_and_duckdb.md).
