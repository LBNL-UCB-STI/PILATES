---
title: Workflow Primer
summary: Conceptual entrypoint for how the current PILATES runtime fits together.
---

# Workflow Primer

## Reading Path

- Start with [Scenario Lifecycle](../run/scenario_lifecycle.md) if you want to know what happens during `python run.py`.
- Continue to [Architecture](architecture.md) for the stable layer map.
- Read [Stages and Steps](stages_and_steps.md) for the practical execution units.
- Read [Model Boundaries](../reference/model_boundaries.md) for the per-model handoff map.
- Read [Step Contracts](step_contracts.md) when you need the catalog, typed-output, and coupler-key details.
- If Consist behavior matters, go to [Consist in PILATES](consist_in_pilates.md).

## If You Only Need One Mental Model

Keep this split in mind:

- **Launcher** owns run lifecycle: settings, state, storage roots, Consist tracker, bootstrap, scenario context, year loop, snapshots, and shutdown.
- **Enabled workflow surface** is the shared projection of settings plus state that decides the active run shape.
- **Stages** own ordering and loop structure.
- **Step factories** own typed execution boundaries.
- **Typed outputs and the coupler** carry workflow-visible results forward.

That is the practical way to read the runtime. Start broad in the launcher,
then move down one layer at a time: launcher → stage → step factory → model
adapter.

## How A Run Moves Through The Code

The runtime path is:

1. `run.py` calls `pilates/runtime/launcher.py`.
2. The launcher prepares a run context: settings, `WorkflowState`, enabled surface, storage roots, tracker, workspace, and restart paths.
3. The launcher runs restart preflight and bootstrap before entering the Consist scenario context.
4. The launcher builds the scenario contract from the enabled surface, declares coupler outputs, and seeds bootstrap artifacts.
5. The year loop calls the major stage modules in order.
6. Each stage uses step refs, binding plans, and typed outputs to execute model phases.
7. The launcher snapshots Consist state at stage and iteration boundaries and finalizes archive state on shutdown.

If you are debugging where a file came from, this sequence matters. A file can
be a local workspace scratch file, a typed step output, a coupler-published
artifact, or an archived Consist output. Only the latter three are part of the
workflow-visible contract.
