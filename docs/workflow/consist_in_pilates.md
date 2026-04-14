---
title: Consist in PILATES
summary: What Consist owns in PILATES and how replay, caching, staging, and analysis fit together.
---

# Consist in PILATES

## Purpose

Treat Consist as a first-class part of the public runtime model rather than a side note under provenance.

## Who This Is For

- Contributors working on runtime, restart, lineage, or cache behavior.
- Advanced users who need to understand how PILATES uses scenarios, steps, artifacts, and analysis trackers.

## This Page Answers

- What does Consist own versus what does PILATES own?
- How do cache hits, replay, input staging, artifact logging, and archive recovery roots fit together?
- How should readers think about runs, steps, the coupler, artifacts, and analysis access?

## Adjacent Pages

- Read [Scenario Lifecycle](../run/scenario_lifecycle.md) first.
- Continue to [Step Contracts](step_contracts.md) and [Artifact Semantics](artifact_semantics.md).
- For post-run analysis, go to [Opening Archives](../analysis/opening_archives.md).

## Source Material To Mine

- `docs-internal/CONSIST_RESTART_ARCHIVE_INTEGRATION_PLAN.md`
- `pilates/utils/consist_config.py`
- `pilates/utils/consist_analysis.py`
- current runtime and step decoration surfaces
