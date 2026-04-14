---
title: Scenario Comparison
summary: Comparing scenario outputs, aligned runs, and summarized differences across runsets.
---

# Scenario Comparison

## Purpose

Describe the current comparison workflow for paired scenarios and aligned run selections.

## Who This Is For

- Analysts comparing baseline and alternative scenarios.
- Contributors extending the comparison API or documenting expected comparison outputs.

## This Page Answers

- How are left and right scenario selections aligned?
- Which datasets and summary surfaces are available for comparison today?
- What assumptions does the comparison logic make about runsets, alignment keys, and completeness?

## Adjacent Pages

- Read [Run Discovery and Runsets](run_discovery_and_runsets.md) first.
- Pair this with [Datasets](datasets.md).
- Use [Analysis Patterns](analysis_patterns.md) for question-driven workflows.

## Source Material To Mine

- `scenario_compare.py` and `comparison_api.py`
- related comparison tests under `tests/`
