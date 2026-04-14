---
title: Troubleshooting
summary: Common startup, runtime, restart, data, and analysis failures in the current PILATES runtime.
---

# Troubleshooting

## Purpose

Collect the highest-value failure modes in one current-state runbook.

## Who This Is For

- Users blocked on setup or execution.
- Operators debugging replay, restart, data layout, or analysis-surface problems.

## This Page Answers

- What are the most common startup and config failures?
- What should I check when replay, restart, or cache behavior surprises me?
- Where do data, DB, and analysis-specific problems usually surface?

## Planned Sections

- Startup and install failures
- Config and CLI failures
- Runtime, cache, and restart failures
- Data layout and alignment failures
- DuckDB and analysis-surface failures

## Adjacent Pages

- Use [Getting Started](../start-here/getting_started.md) for the normal path.
- Use [Restart and Resume](restart_and_resume.md) for replay-specific behavior.
- Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) for post-run DB inspection.

## Source Material To Mine

- Existing troubleshooting cases from the old page.
- Restart-related diagnostics from the replay-first refactor.
- Current DB and analysis helper behavior in `pilates/utils/consist_analysis.py`.
