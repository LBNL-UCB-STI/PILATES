---
title: Workspace Layout
summary: Stable path semantics for local workspace, archive roots, and run-local artifacts.
---

# Workspace Layout

## Purpose

Document the stable path model that readers should rely on instead of ambient working-directory assumptions.

## Who This Is For

- Contributors editing model code or restart behavior.
- Operators reasoning about local workspace, archive roots, and mounted inputs on local or HPC runs.

## This Page Answers

- What is the difference between the mutable workspace and the durable archive root?
- Which path families are model-owned versus runtime-owned?
- Where do run-local artifacts such as `.consist` DBs, run state, and staged inputs usually live?

## Adjacent Pages

- Read [Scenario Lifecycle](../run/scenario_lifecycle.md) first.
- Pair this with [Opening Archives](../analysis/opening_archives.md).
- Use [Lawrencium](../run/lawrencium.md) for cluster-specific storage posture.

## Source Material To Mine

- `pilates/workspace.py`
- current HPC scripts and replay-first archive notes
