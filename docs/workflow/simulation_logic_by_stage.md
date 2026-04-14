---
title: Simulation Logic by Stage
summary: Science-facing description of what each stage is logically doing in the overall simulation.
---

# Simulation Logic by Stage

## Purpose

Separate the logical simulation meaning of each stage from the code ownership description in architecture docs.

## Who This Is For

- Domain readers who want to understand why the workflow is ordered the way it is.
- Contributors documenting scientific boundaries between model families.

## This Page Answers

- What problem is each major stage solving in the simulation?
- How do yearly land use, vehicle ownership, travel demand, traffic assignment, and postprocessing relate logically?
- Which stage boundaries are scientific handoffs versus implementation conveniences?

## Adjacent Pages

- Read [Stages and Steps](stages_and_steps.md) for the runtime execution model.
- Use [Artifact Flow](artifact_flow.md) for the concrete handoffs.
- Use [Model Stack](../reference/model_stack.md) for the role of each model family.

## Source Material To Mine

- Existing README model descriptions.
- Current stage ordering and loop structure in the runtime.
- Artifact handoff docs and lineage maps.
