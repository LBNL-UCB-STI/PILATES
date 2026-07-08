---
title: Analysis Overview
summary: What post-run analysis means in PILATES and which analysis surfaces exist.
---

# Analysis Overview

## Reading Path

- Continue to [Opening Archives](opening_archives.md).
- If you want concrete scripts first, go to [Consist in Action](consist_in_action.md).
- Then choose [Consist Analysis CLI](consist_analysis_cli.md), [Run Discovery and Runsets](run_discovery_and_runsets.md), or [SQL and DuckDB](sql_and_duckdb.md).
- For beginner comparisons, use [Faceted Comparison](scenario_comparison.md).
- For the post-cleanup dataset-builder status, go to [Datasets](datasets.md).

## Public Surface

PILATES exposes analysis through two layers:

- The Python API, centered on `Archive`, faceted Ibis tables, and small helpers such as `delta()`, `difference()`, `delta_change()`, and `rank()`.
- The CLI, which wraps discovery, epoch summaries, run-tagging checks, and DB health checks.

The analysis package reads archived runs through a Consist tracker, then builds these surfaces:

- a run index over discovered runs and their source metadata
- faceted Ibis tables over tabular artifacts
- grouped outcome tables over arbitrary facets such as `scenario_id`, `year`, `iteration`, and policy parameters
- simple delta, difference, delta-of-delta, and rank helpers
- advanced run sets and epoch panels for run-selection workflows
- SQL and DuckDB access for ad hoc inspection

## Start Here

If you want to inspect an archived run, start with [Opening Archives](opening_archives.md).
If you want small read-only examples, start with [Consist in Action](consist_in_action.md).
If you want the command list, start with [Consist Analysis CLI](consist_analysis_cli.md).
If you want to understand run grouping internals, start with [Run Discovery and Runsets](run_discovery_and_runsets.md).
If you want a one-off SQL-style inspection path, go straight to [SQL and DuckDB](sql_and_duckdb.md).
