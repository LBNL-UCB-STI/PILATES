---
title: Workflow Primer
summary: Conceptual entrypoint for how the current PILATES runtime fits together.
---

# Workflow Primer

## Reading Path

- Continue to [Architecture](architecture.md) for the stable layer map.
- Then read [Stages and Steps](stages_and_steps.md) and [Step Contracts](step_contracts.md).
- If Consist behavior matters, go to [Consist in PILATES](consist_in_pilates.md).

## If You Only Need One Mental Model

Keep this split in mind:

- the launcher owns run lifecycle
- stages own ordering and loop structure
- step factories own typed execution boundaries
- typed outputs and the coupler carry workflow-visible results forward
