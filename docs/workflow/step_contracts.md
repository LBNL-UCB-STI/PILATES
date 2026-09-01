---
title: Step Contracts
summary: Native Consist step definitions and typed PILATES output boundaries.
---

# Step Contracts

Each workflow boundary has one native contract. The sequence is deliberately
short:

```text
define semantic roles
-> resolve them once
-> execute through Consist
-> archive/refresh/republish when enabled
-> project Consist outputs into typed PILATES outputs
-> sequence the next step
```

## Contract ownership

`WorkflowStepSpec` is the policy catalog entry. It owns the canonical step
name, phase, stage placement, order, enablement, dependencies, provenance
policy, and dynamic semantic families. It reads static Consist metadata from
the committed `StepDefinition`; it does not duplicate input or output claims.

`StepDefinition` owns one decorated Consist callable, its input resolver, its
typed-output projector, and (when needed) output-path, execution, and cache
providers. The decorated callable declares static semantic roles and schema
outputs. Dynamic selections remain resolver-owned.

## Resolve once, execute once

A resolver returns `ResolvedStepInputs`. It selects the current coupler values
for the declared semantic roles, freezes selected artifacts in
`ResolvedStepInputs.binding`, and records their deterministic materialization
destinations. Required roles must be complete before execution.

Stages call `execute_step(...)` with the `StepDefinition`. `execute_step()`
uses that single resolved selection (or resolves once), runs the decorated
callable through Consist, then performs the current PILATES action-evidence
refresh and, when enabled, archive/refresh/republish bridge. It gives the
resulting persisted `RunResult.outputs` to the projector. A projector validates
current declared destinations and returns the typed PILATES output object
consumed by the following stage decision. `scenario.run()` does not itself make
this archive-validation and refreshed-publication guarantee.

## Typed outputs and artifacts

Typed outputs subclass `StepOutputsBase`. They declare public record keys,
required and optional path fields, nested path maps, and semantic validators.
They represent the current execution's persisted outputs; they do not discover
historical sources or reconstruct inputs.

The coupler is the current semantic-role map. Consist owns artifact identity,
materialization, cache admission, and persisted output lineage. PILATES owns
semantic role selection, typed validation, stage sequence, and restart policy.

Every native definition now has an `InputContract`, but that structural fact is
not blanket cache-promotion evidence. Native execution or a cache hit also
does not establish portable promotion. `beam_preprocess` is the completed
boundary with fresh-workspace promotion evidence; other definitions remain
boundary-specific work, including a separate HDF5 promotion gate.

## Restart boundary

There is one committed mid-stage restart boundary: `beam_run_completed` to
`beam_postprocess`. PILATES snapshots the exact resolved successor closure,
then a restart validates that snapshot, hydrates the recorded artifacts to the
current resolver destinations, re-resolves the successor, and executes native
postprocess once. The `beam_postprocess_in_progress` mutation gate remains
non-restartable. Other restarts resume their normal stage/year frontier; a
skipped ActivitySim mid-stage does not receive a second recovery path.

## Adjacent pages

- Read [Stages and Steps](stages_and_steps.md) for sequencing.
- Continue to [Model Integration Guide](../extend/model_integration_guide.md).
- Pair this with [Artifact Semantics](artifact_semantics.md).

## Evidence basis

- Native execution: `pilates/workflows/step_execution.py`
- Contract type: `pilates/workflows/step_definition.py`
- Catalog policy: `pilates/workflows/catalog.py`
- Native model definitions: `pilates/workflows/steps/`
