---
title: Restart and Resume
summary: Replay-first restart model, cache-hit behavior, and operator expectations.
---

# Restart and Resume

## Purpose

Document how restart works now that PILATES is centered on replay and declared inputs and outputs.

## Who This Is For

- Users resuming interrupted runs.
- Operators reasoning about cache hits, workspace loss, archive roots, and replay safety.

## This Page Answers

- What does resume mean in the post-refactor runtime?
- How do cache hits, replay, and requested input staging fit together?
- What should an operator expect from restart after interruption or local workspace loss?

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) first.
- Use [HPC Overview](hpc_overview.md) for archive-root and workspace differences on clusters.
- Use [Consist in PILATES](../workflow/consist_in_pilates.md) for the ownership model behind replay.

## Source Material To Mine

- `docs-internal/CONSIST_RESTART_ARCHIVE_INTEGRATION_PLAN.md`
- Current launcher, bootstrap, and restart modules.
- Active restart-related troubleshooting cases.
