---
title: Opening Archives
summary: Archive directory, DuckDB, and tracker mental model for post-run analysis.
---

# Opening Archives

## Purpose

Explain how to reopen a completed PILATES run as an analysis target.

## Who This Is For

- Users inspecting archived outputs after local or HPC execution.
- Analysts who need to understand the relationship between the run archive, `.consist` DB, mounts, and tracker access mode.

## This Page Answers

- What files and directories make up an analyzable run archive?
- How do archive run dir, workspace mount, DB path, and recovery roots fit together?
- Which helpers create an analysis tracker and what assumptions do they make?

## Adjacent Pages

- Read [Analysis Overview](overview.md) first.
- Then use [SQL and DuckDB](sql_and_duckdb.md) or [Consist Analysis CLI](consist_analysis_cli.md).
- Pair this with [Workspace Layout](../reference/workspace_layout.md) for path semantics.

## Source Material To Mine

- `pilates/utils/consist_analysis.py`
- `analysis/src/pilates_consist_analysis/runtime.py`
- replay-first archive design notes and current HPC storage behavior
