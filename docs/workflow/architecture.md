---
title: Workflow Architecture
summary: Ownership boundaries for native Consist execution in PILATES.
---

# Workflow Architecture

PILATES is a stage-oriented workflow with native Consist execution at each
model boundary. A stage decides *when* a step runs. A `StepDefinition` decides
*what* that boundary consumes and produces.

## The execution path

```text
semantic roles -> ResolvedStepInputs -> Consist run -> RunResult.outputs
               -> typed PILATES projection -> next stage decision
```

The resolver reads the current semantic-role map once and freezes its selected
artifacts and destinations. `execute_step(...)` materializes those inputs,
runs the decorated callable through Consist, and projects persisted outputs
into a typed object. Stages retain ordering, iteration, and workflow-state
policy; they do not choose alternate input sources or implement execution
plumbing.

## Ownership

| Concern | Owner |
| --- | --- |
| Artifact identity, materialization, cache admission, persisted lineage | Consist |
| Semantic-role selection, typed validation, stage sequence, restart policy | PILATES |
| Model-local preparation, execution, and postprocessing | Model adapter |
| Current semantic role values | Scenario coupler |
| Historical committed boundary facts | Snapshot artifacts |

`WorkflowState` is durable control state. `Workspace` is run-local layout.
The coupler is not a history store, and archive roots describe byte locations
for existing artifacts rather than new workflow roles.

## Restart policy

The sole committed mid-stage restart is `beam_run_completed` to
`beam_postprocess`. Its snapshot pins the immediate successor inputs. On
restart, PILATES validates and hydrates that closure to current destinations,
re-resolves the native successor, preserves the mutation gate, and executes
postprocess once. A path existing upstream does not make the in-progress
postprocess region restartable.

All other restart behavior is stage/year-frontier policy. In particular,
ActivitySim is not generically replayed from an interrupted mid-stage state.

## Reading path

- Start with [Stages and Steps](stages_and_steps.md).
- Read [Step Contracts](step_contracts.md) for the executable boundary.
- Read [Consist in PILATES](consist_in_pilates.md) for lifecycle and artifact
  behavior.
