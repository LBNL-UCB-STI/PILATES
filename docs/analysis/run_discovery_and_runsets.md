---
title: Run Discovery and Runsets
summary: Discovering runs, building run indexes, grouping runsets, and reasoning about epochs.
---

# Run Discovery and Runsets

## Purpose

Explain the analysis-side grouping model for runs, runsets, and epochs.

## Who This Is For

- Analysts comparing many archived runs.
- Contributors documenting scenario alignment, converged epochs, and run selection behavior.

## This Page Answers

- How do readers discover runs and build useful subsets?
- What is a runset and how does it differ from a single run or a scenario epoch?
- How do run indexing, epoch panels, and selection helpers fit together?

## Adjacent Pages

- Read [Analysis Overview](overview.md) first.
- Then use [Scenario Comparison](scenario_comparison.md) or [Datasets](datasets.md).
- Pair this with [Glossary](../reference/glossary.md) for shared terms.

## Source Material To Mine

- `run_index.py`, `runset.py`, `epochs.py`, and related analysis APIs.
- existing analysis tests around run tagging and epoch views.
