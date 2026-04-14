---
title: Workflow Primer
summary: Conceptual entrypoint for how the current PILATES runtime fits together.
---

# Workflow Primer

## Purpose

Explain the current end-to-end workflow shape before readers dive into detailed semantics or code ownership.

## Who This Is For

- Contributors orienting themselves in the post-refactor runtime.
- Advanced users who need the mental model behind yearly staging, inner loops, and cross-model handoffs.

## This Page Answers

- What are the major layers between `run.py` and model execution?
- How do stages, steps, typed outputs, and the coupler fit together?
- Which docs should a contributor read next depending on the task?

## Reading Path

- Continue to [Architecture](architecture.md) for the stable layer map.
- Then read [Stages and Steps](stages_and_steps.md) and [Step Contracts](step_contracts.md).
- If Consist behavior matters, go to [Consist in PILATES](consist_in_pilates.md).

## Source Material To Mine

- Runtime assembly in `pilates/runtime/launcher.py`.
- Stage orchestration in `pilates/workflows/stages/`.
- Step-building and contract layers under `pilates/workflows/`.
