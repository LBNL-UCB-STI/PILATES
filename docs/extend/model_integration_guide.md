---
title: Model Integration Guide
summary: Architecture guide for native Consist model boundaries in PILATES.
---

# Model Integration Guide

PILATES model integration has one execution path:

```text
semantic roles -> resolver -> Consist execution -> persisted outputs
               -> typed projection -> next stage sequence
```

## Native model boundary

A model package provides preprocessors, runners, postprocessors, and typed
output classes. Its workflow step module provides:

- a decorated Consist callable with static semantic metadata;
- a resolver returning `ResolvedStepInputs` with concrete selected artifacts;
- a projector consuming the current `RunResult.outputs`;
- a `StepDefinition` that binds those pieces together.

Every `StepDefinition` declares an `InputContract`. Its optional
`config_contract` field is required for `complete`, which requires demonstrated
portable identity closure; it may be absent for `incomplete`. `incomplete` is a
status and evidence record for a boundary that remains open; it is not a
cache-promotion mechanism.

The catalog references that definition for static metadata while retaining only
PILATES policy: stage placement, ordering, enablement, dependencies,
provenance, and dynamic semantic families.

## Input and output discipline

Resolve input roles once. A resolver may record diagnostics, but execution
uses its concrete `ResolvedStepInputs.binding`, not a second key lookup. Give each
input a deterministic destination so Consist can materialize it. A projector
must validate the declared current destination; it must not fall back to an
archive source or a previous workspace.

After `scenario.run(...)`, the shared `execute_step(...)` bridge refreshes its
post-run action evidence, archives configured outputs, and republishes archived
artifacts to the coupler before it invokes the projector. This central PILATES
bridge policy must not be recreated as a second archive, refresh, or republish
path in a model integration.

Declare static roles in `@define_step`. Keep invocation-dependent behavior in
the resolver and dynamic output-path provider. For example, ActivitySim selects
one skim input and conditionally produces Zarr; the definition does not claim
that Zarr is unconditionally both an input and output.

## Stage and restart policy

Stages call `execute_step(...)` explicitly in their required order. They own
iteration and `WorkflowState` transitions, while Consist owns generic cache and
artifact behavior.

The only committed mid-stage restart is BEAM run completion to BEAM
postprocess. It validates a pinned successor closure from the tracker snapshot,
hydrates it to the normal resolver destinations, re-resolves postprocess, and
executes it once. Other restarts return to their normal stage/year frontier.

## Reading and validation

1. Read the relevant `StepDefinition` and its tests.
2. Read the policy entry in `pilates/workflows/catalog.py`.
3. Read the owning stage's explicit execution calls.
4. Review `step_consist_meta.py` only for identity, facet, and config-adapter
   policy.

`scripts/new_model_scaffold.py` creates a mechanical native starting point with
conservative incomplete adapter contracts. Do not treat its output as a turnkey
model integration; use the [Model Contract Checklist](model_contract_checklist.md)
to establish the real native boundary.

## Adjacent pages

- [Adding a Model](adding_a_model.md)
- [Model Contract Checklist](model_contract_checklist.md)
- [Workflow Architecture](../workflow/architecture.md)
