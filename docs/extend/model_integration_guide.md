---
title: Model Integration Guide
summary: Architecture-oriented guide to how model components plug into the PILATES workflow.
---

# Model Integration Guide

## What PILATES Does

PILATES keeps model-specific logic inside model packages and keeps workflow policy in the orchestration layer.

Model packages provide:

- preprocessors, runners, and postprocessors
- typed output classes for each phase
- model-local input/output logging and recovery callbacks

Workflow code provides:

- the catalog entry that names the step and declares its contract
- the enabled workflow surface that decides when that contract is active in the current run shape
- the step factory shell that binds the live coupler and outputs holder
- Consist step metadata for cache identity, facets, config adapters, and
  identity inputs
- binding rules that resolve coupler values, explicit inputs, and fallback
  sources into one step input plan
- validation that the step returned the expected typed output class
- publication of the typed outputs into the holder and coupler

For model integration work, that means enablement and restart-sensitive contract decisions belong in workflow code. Model modules should not rebuild their own view of runtime flags or active stages.

## Preferred Integration Path

For new models, start with `scripts/new_model_scaffold.py`.

The scaffold is the maintained contributor path for the current architecture. It generates:

- typed output shells
- `WorkflowStepSpec`-aligned step scaffolding
- `StandardStepSpec` / `build_standard_step()` step modules
- `SCHEMA_STEP_BUILDERS` registrations
- stage snippets that assume `WorkflowRuntimeContext`, `StageRunner`, and `build_binding_plan(...)`
- a follow-up checklist under `docs/checklists/`

That is preferable to copying an older model by hand, because the scaffold tracks the current surface-first runtime conventions.

## Current Developer Mental Model

Think of a model integration as four layers, in this order:

1. **Adapter layer**: `pilates/<model>/` knows how to prepare files, run the model, and interpret its local outputs.
2. **Typed boundary layer**: `pilates/<model>/outputs.py` tells the workflow which outputs are public enough to validate, log, and hand off.
3. **Workflow contract layer**: `pilates/workflows/catalog.py` and `pilates/workflows/steps/<model>.py` declare and execute the tracked step boundary.
4. **Consist and binding layer**: `pilates/workflows/step_consist_meta.py` and
   `pilates/workflows/binding.py` declare cache identity, config reference
   policy, and runtime source precedence.
5. **Stage layer**: `pilates/workflows/stages/` decides when the tracked step runs in the year or iteration loop.

The launcher should not need to know model-local details. A reader should be
able to open `pilates/runtime/launcher.py`, understand the lifecycle, and then
follow the stage call into the model-specific workflow files.

## What Not To Copy From Older Integrations

Some established models still carry historical shape. For new work, avoid
copying these patterns unless there is a concrete reason:

- broad `getattr(...)` probing where typed config fields are guaranteed
- stage-local reconstructions of model enablement flags
- direct filesystem discovery when a workflow key or binding plan should provide the source
- untyped dictionaries for outputs that later steps or restart logic consume
- treating the coupler as an archive of historical versions rather than the
  current semantic-role map
- adding archive or recovery-root metadata when the workflow actually needs a
  new snapshot artifact key
- launcher changes for model-specific behavior

The preferred replacement is explicit: typed config access, enabled-surface
decisions, binding plans, typed outputs, and stage-local sequencing only.

## Integration Shape

### ModelFactory

`ModelFactory` maps lowercased model names to concrete component classes. The current registry includes:

- `activitysim`
- `activitysim_compile`
- `beam`
- `beam_full_skim`
- `atlas`
- `urbansim`

`get_preprocessor()`, `get_runner()`, and `get_postprocessor()` instantiate the corresponding component class for the model name. `get_components()` returns the three instances together.

### Step factories

The step modules under `pilates/workflows/steps/` build the runtime callables. A factory binds the current coupler, outputs holder, and model-local callbacks into one step function. The factory does not expose a bare `WorkflowStepSpec`; it returns an executable step decorated with Consist metadata.

`build_standard_step()` applies the shared Consist decoration. The lazy metadata
builders in `pilates/workflows/step_consist_meta.py` provide per-step adapters,
scalar `config`, query `facet`, `facet_index`, `facet_schema_version`, and
`identity_inputs`.

### Binding owns source precedence

When a step needs inputs from coupler state, explicit upstream outputs, or fallbacks, use binding.

In practice that means:

- stage code builds a `BindingPlan`
- catalog-derived defaults are enough for ordinary required and optional keys
- custom `ArtifactBindingRule` entries live in `pilates/workflows/binding.py`
  when a semantic key needs special precedence, preferred keys, or fallback
  providers
- the enabled workflow surface tells binding which inputs are active for the current run shape
- model modules do not manually recreate fallback chains with ad hoc `coupler.get(...)` logic

This keeps input precedence, optionality, and restart-sensitive behavior in one place.

### Consist metadata owns cache identity

Use `pilates/workflows/step_consist_meta.py` when a step needs more than the
default scalar metadata. ActivitySim and BEAM are the reference examples:
ActivitySim config roots are fingerprinted by an ActivitySim config adapter,
and BEAM config references use `BeamReferencePolicy` entries to mark paths as
delegated to workflow artifacts, ignored optional examples, or runtime outputs.

Do not log model config trees as ordinary artifacts just to make cache identity
work. If configuration changes should invalidate a step cache, put that policy
behind the config adapter or explicit `identity_inputs`.

### Coupler, snapshots, and storage metadata

The coupler carries the current semantic role values that later steps resolve.
It is not the historical record of every prior version of a file. When a model
boundary creates a state that future restart/replay needs to identify, publish a
snapshot artifact key through the step contract.

Archive roots and recovery roots are different: they describe where the bytes
for an existing artifact can be found or recovered. They do not create a new
semantic role. Use [Artifact Semantics](../workflow/artifact_semantics.md) and
[Consist in PILATES](../workflow/consist_in_pilates.md) before adding new
artifact families.

### Stages own sequencing

Stages decide when a step runs, not whether it is enabled in the abstract and not how its inputs should be prioritized.

The normal orchestration shape is:

- build or receive a `WorkflowRuntimeContext`
- create a `StageRunner`
- create a `BindingPlan`
- run a `StepRef`

That boundary is intentional. It keeps stage modules readable and stops policy from scattering across model packages.

### Launcher stays model-agnostic

`pilates/runtime/launcher.py` owns the run lifecycle:

- prepare settings, state, surface, tracker, workspace, and storage roots
- run restart preflight and bootstrap
- enter the Consist scenario context
- declare the coupler schema and seed bootstrap artifacts
- call the major stage modules in year order
- snapshot and shut down archive workers

Model-specific fallback logic should not be added there. If a fallback belongs
to ATLAS, BEAM, ActivitySim, UrbanSim, or postprocessing, put it in the model
package or the stage that owns that boundary and pass it in explicitly.

### Typed outputs and publication

Model code returns a typed outputs object that extends `StepOutputsBase`. The step factory then:

- validates required output paths and semantic validators
- stores the typed outputs in `StepOutputsHolder`
- publishes record-store entries or coupler artifacts as the step logic requires

### Why the model module still owns callbacks

`build_standard_step()` stays thin. The model modules still provide the callbacks for input logging, output logging, replay, and recovery because those callbacks close over model-local behavior and a live coupler. That keeps the shared shell reusable without hiding model-specific wiring.

## Reading Path

- Read [Workflow Primer](../workflow/workflow_primer.md) and [Step Contracts](../workflow/step_contracts.md) first.
- Read [Consist in PILATES](../workflow/consist_in_pilates.md) for the
  Consist-owned lifecycle and cache/replay boundary.
- Read [Model Boundaries](../reference/model_boundaries.md) when you need the per-model I/O and handoff picture before touching code.
- Continue to [Adding a Model](adding_a_model.md) for the hands-on path.
- Use [Output Validation](output_validation.md) for runtime and startup guardrails.

## How To Trace An Existing Model Boundary

When you are new to a model family, use this order:

1. read the relevant section in [Model Boundaries](../reference/model_boundaries.md)
2. inspect the static contract in `pilates/workflows/catalog.py`
3. inspect the step factory in `pilates/workflows/steps/*.py`
4. inspect `pilates/workflows/step_consist_meta.py` for cache, facet, and
   config-adapter metadata
5. inspect `pilates/workflows/binding.py` or the input-selection helper
6. inspect the stage module that actually runs the step
7. inspect the model adapter module only after the workflow boundary is clear

That sequence answers these questions cleanly:

- what the workflow says the step needs
- what the typed execution boundary looks like
- what Consist uses for cache identity and query facets
- how runtime source precedence is resolved
- where the step runs in the stage loop
- what later stages consume next
- what the model adapter does internally after the workflow has selected inputs

## Evidence Basis

- Model dispatch: `pilates/generic/model_factory.py`
- Runtime context: `pilates/runtime/context.py`
- Shared step shell: `pilates/workflows/steps/shared.py`
- Consist step metadata and config adapter policy: `pilates/workflows/step_consist_meta.py`
- Binding and source precedence: `pilates/workflows/binding.py`
- Catalog and contract metadata: `pilates/workflows/catalog.py`
- Surface-driven enablement: `pilates/workflows/surface.py`
- Model-specific step wiring under `pilates/workflows/steps/`
- Registry and factory tests: `tests/test_generic_modules.py`, `tests/test_coupler_key_invariants.py`, `tests/test_standard_step_builder.py`
