---
title: Analysis Overview
summary: What post-run analysis means in PILATES and which analysis surfaces exist.
---

# Analysis Overview

## Purpose

Frame analysis as a first-class public part of PILATES rather than an afterthought after execution.

## Who This Is For

- Users opening archived runs after execution.
- Analysts comparing scenarios, building datasets, or querying run metadata and artifacts.

## This Page Answers

- What is the current public analysis surface after a run completes?
- How do archive directories, Consist DuckDB files, trackers, runsets, and datasets fit together?
- Which analysis entrypoint should a reader choose first?

## Reading Path

- Continue to [Opening Archives](opening_archives.md).
- Then choose [Consist Analysis CLI](consist_analysis_cli.md), [Run Discovery and Runsets](run_discovery_and_runsets.md), or [SQL and DuckDB](sql_and_duckdb.md).
- For direct scenario comparisons, go to [Scenario Comparison](scenario_comparison.md).

## Source Material To Mine

- `analysis/src/pilates_consist_analysis/`
- `pilates/utils/consist_analysis.py`
- existing SQL example material in `docs/example_queries.sql`
