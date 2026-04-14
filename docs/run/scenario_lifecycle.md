---
title: Scenario Lifecycle
summary: High-level shape of a PILATES run from bootstrap through archive and restart.
---

# Scenario Lifecycle

## Purpose

Give the run-level mental model before readers dive into CLI flags, config fields, or workflow internals.

## Who This Is For

- Users who know they want to run PILATES but need to understand what a run actually does.
- Operators reasoning about output roots, archive roots, restart state, and replay.

## This Page Answers

- What are the major phases of a scenario run?
- Where do bootstrap, stage execution, archive state, and restart state fit?
- How should users think about fresh reruns versus replay through cache hits?

## Adjacent Pages

- Then read [CLI](cli.md) and [Configuration Reference](configuration_reference.md).
- For restart details, go to [Restart and Resume](restart_and_resume.md).
- For workflow semantics, go to [Workflow Primer](../workflow/workflow_primer.md).

## Source Material To Mine

- Runtime assembly in `pilates/runtime/launcher.py`.
- Current restart/archive behavior in the replay-first refactor notes.
- Active run-state handling in `workflow_state.py`.
