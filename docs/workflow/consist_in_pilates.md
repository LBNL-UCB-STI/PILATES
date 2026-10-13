---
title: Consist in PILATES
summary: Ownership and native step execution at the PILATES–Consist boundary.
---

# Consist in PILATES

[Consist](https://lbnl-ucb-sti.github.io/consist/latest/) is PILATES's
execution substrate for declared model boundaries. It gives scenarios, runs,
and artifacts durable identity; binds and materializes inputs; persists output
links and provenance; admits cache hits; and returns `RunResult.outputs`.

PILATES remains a stage-oriented model. It decides which semantic roles matter,
when a model phase runs, what typed output is valid, and which restart policy is
safe. Consist does not own PILATES stage semantics or restart policy.

## Native Step Path

Every enabled model phase follows one path:

```text
semantic roles
-> resolver produces ResolvedStepInputs once
-> execute_step(..., StepDefinition)
-> Consist scenario.run(...)
-> RunResult.outputs
-> PILATES archive/refresh/republish when enabled
-> TypedOutputProjector
-> typed PILATES outputs
```

A `StepDefinition` packages a decorated Consist callable, resolver, typed
projector, and any boundary-local output-path, execution, or cache options.
`WorkflowStepSpec` is only the PILATES policy catalog for stage placement,
order, enablement, dependency, provenance, and dynamic semantic families.

Resolvers select values from the current scenario coupler once, freeze the
selected artifacts and destinations, and require completeness before execution.
The projector validates persisted outputs at their declared destinations.
Consequently fresh runs, admitted cache hits, and the committed restart path
produce the same typed handoff from `RunResult.outputs`.

After `scenario.run()` returns, `execute_step()` refreshes PILATES action
evidence. If archive copying is enabled, it archives the completed outputs and
republishes the refreshed artifacts to the scenario coupler before projection.
That is the current PILATES bridge; `scenario.run()` itself does not make this
archive-validation and refreshed-publication guarantee.

## Advanced immutable bindings

Ordinary `BindingResult`/Coupler binding remains the default. A small set of
native definitions uses Consist `ResolvedBinding` only when every selected
execution input is a locally tracked artifact, maps to a named callable
parameter, and can be consumed from the strict run-owned snapshot.

For those definitions, `execute_step()` asks Consist to preflight the final
decorated identity once, the resolver freezes the selected inputs with that
identity, and the same identity is reused for `scenario.run()`. Strict binding
identity contains the step contract, typed artifact identities, and relative
destinations; selection diagnostics and admission explanation remain in
invocation evidence and do not alter cache reuse.

Dynamic workspace-only inputs remain on ordinary binding. This includes BEAM
postprocess/restart closure inputs, raw BEAM configuration inputs, the BEAM
full-skim runner's workspace mount, ActivitySim population-source fallback and
runner workspace inputs, and model adapters that read a workspace path instead
of their callable parameter.
Strict BEAM linkstats admission is also deferred; it does not change ordinary
BEAM warm-start behavior.

## Ownership

| Concern | Owner |
| --- | --- |
| Generic input binding and materialization | Consist |
| Run and artifact identity, provenance, output links, cache admission | Consist |
| Current semantic-role map | PILATES scenario coupler |
| Semantic selection, typed validation, stage ordering, restart policy | PILATES |
| Model-local preparation, execution, and postprocessing | Model adapter |
| Historical committed boundary fact | Snapshot artifact |

`WorkflowState` is durable workflow control state. `Workspace` is mutable
run-local layout. The coupler describes current roles, not historical evidence;
archive roots tell Consist where an existing artifact may be materialized, not
what the next workflow operation should be.

## Cache and Materialization

Consist calculates cache identity from the declared callable, configuration,
and resolved inputs. Its cache policy controls requested-output hydration and
admission. PILATES consumes the result through the same projector regardless
of whether Consist executed the callable or admitted a cache hit.

This distinction matters operationally: cache behavior is generic Consist
behavior, while a restart decision remains PILATES policy. No cache result
creates a second stage sequence or permits a stage to select a historical
artifact outside its declared roles.

All native definitions have `InputContract`s, but that structural coverage is
not a workflow-wide cache-promotion decision. A native execution or ordinary
cache hit is likewise not proof of portable reuse. `beam_preprocess` is the
only complete native boundary with portable fresh-workspace promotion evidence;
the remaining boundaries require their own acceptance evidence. HDF5 promotion
is a separate gate.

## Restart Boundary

The sole committed mid-stage checkpoint is `beam_run_completed` to
`beam_postprocess`. PILATES pins the immediate successor input closure,
flushes queued archival, synchronously archives selected closure sources, and
verifies their archive-visible bytes before publishing the pinned snapshot. On
restart it validates and hydrates that closure to exact current destinations,
then re-resolves and runs native `beam_postprocess`. The in-progress
postprocess mutation gate remains non-restartable.

## Evidence Status

The SFBay structural checker is the formal structural acceptance artifact.
Seattle v1 is supporting runtime evidence only. BEAM-preprocess and the
editable-Consist ActivitySim-run acceptances are boundary-specific; replay with
a released Consist wheel remains to be shown. HDF5 promotion is separate. This
page does not claim an HPC execution path as acceptance evidence.

All other restart behavior follows the normal durable stage/year frontier. In
particular, an interrupted ActivitySim mid-stage operation fails closed.

## Public Artifact Surface

The coupler keys and typed outputs remain PILATES's public semantic surface.
Important examples include `USIM_DATASTORE_CURRENT_H5`, `ZARR_SKIMS`,
`FINAL_SKIMS_OMX`, `BEAM_FULL_SKIMS`, and year-scoped UrbanSim input snapshot
families. Use [Artifact Semantics](artifact_semantics.md), [Artifact Flow](artifact_flow.md), and [Lineage Map](lineage_map.md) for their contract meanings.

## Optional BEAM Linkstats Admission

PILATES can verify a configured staged BEAM warm-start linkstats artifact
against a declared completed Consist-run input before the BEAM container starts.
This optional, model-specific admission is separate from generic cache
admission and restart policy. See the BEAM configuration reference for the
`beam.admission.linkstats` options and report.

## Analysis Surface

PILATES retains the archive-side Consist database, run metadata, output links,
and artifacts for analysis. Analysis reads that persisted evidence; it does not
reconstruct workflow state. See [Consist in Action](../analysis/consist_in_action.md).

## Adjacent Pages

- Read [Workflow Architecture](architecture.md) for the layer map.
- Read [Step Contracts](step_contracts.md) for the executable boundary.
- Read [Restart and Resume](../run/restart_and_resume.md) for checkpoint policy.
