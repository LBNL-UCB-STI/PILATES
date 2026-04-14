---
title: Consist Analysis CLI
summary: Public analysis CLI and API entrypoints for archived-run inspection and export workflows.
---

# Consist Analysis CLI

## Purpose

Document the current command and API surface exposed by the analysis package.

## Who This Is For

- Analysts using the packaged analysis CLI instead of writing ad hoc scripts.
- Contributors extending or documenting the public analysis command surface.

## This Page Answers

- Which high-level commands exist today for discovery, datasets, export, and comparison?
- Which commands are the stable public entrypoints versus implementation helpers?
- How should readers choose between the CLI and direct Python APIs?

## Adjacent Pages

- Start with [Opening Archives](opening_archives.md).
- Pair this with [Run Discovery and Runsets](run_discovery_and_runsets.md) and [Datasets](datasets.md).
- Use [Scenario Comparison](scenario_comparison.md) for paired-run workflows.

## Source Material To Mine

- `analysis/src/pilates_consist_analysis/cli.py`
- `analysis/src/pilates_consist_analysis/api.py`
- existing CLI-oriented tests under `tests/`
