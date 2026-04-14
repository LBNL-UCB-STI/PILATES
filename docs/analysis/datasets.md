---
title: Datasets
summary: Dataset-oriented analysis surfaces for linkstats, skims, ActivitySim trips, and exported bundles.
---

# Datasets

## Purpose

Map the packaged dataset builders onto the artifact families they depend on.

## Who This Is For

- Analysts who want higher-level dataset products instead of raw artifact inspection.
- Contributors documenting or extending packaged analysis outputs.

## This Page Answers

- Which packaged datasets exist today and what artifacts do they consume?
- When should a reader choose a dataset builder versus direct SQL or direct artifact access?
- What do linkstats, skims, ActivitySim trips, and exported scenario bundles currently mean?

## Adjacent Pages

- Read [Opening Archives](opening_archives.md) first.
- Use [Consist Analysis CLI](consist_analysis_cli.md) for command entrypoints.
- Pair this with [Artifact Semantics](../workflow/artifact_semantics.md).

## Source Material To Mine

- `datasets.py`, `skim_analysis.py`, `activitysim_trips.py`, and `packaging.py`
- current analysis command surface and tests
