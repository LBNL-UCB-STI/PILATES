---
title: Database Setup
summary: Enabling and validating the current run-local Consist DuckDB and related database surfaces.
---

# Database Setup

## Purpose

Document how the current run-local database surface is enabled and what a healthy setup looks like.

## Who This Is For

- Users turning on DB logging or checking whether a run emitted the expected DB artifacts.
- Analysts and operators who need the run-local DB before deeper schema or SQL work.

## This Page Answers

- Which config knobs control the run-local database surface?
- Where should the DB live inside a run archive?
- What is the quickest validation path after a run starts or completes?

## Adjacent Pages

- Pair this with [Database Documentation Guide](database_documentation_guide.md).
- Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) for direct inspection workflows.
- Use [Opening Archives](../analysis/opening_archives.md) for archive-side access.

## Source Material To Mine

- Prior database-setup guide content.
- Current DB-health helpers in `pilates/utils/consist_analysis.py`.
