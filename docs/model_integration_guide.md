# Model Integration Guide

This guide explains the current model-integration architecture in PILATES. It
is intended for developers who need to understand where model code belongs,
what the workflow contract looks like, and how caching/provenance/restart fit
into that design.

Use it with:

- `docs/adding_a_model.md` for the implementation checklist
- `docs/workflow_primer.md` for the end-to-end runtime flow

## Core Rule

The public workflow contract in PILATES is:

- typed step outputs
- path-based artifacts
- explicit coupler publication
- stage-owned orchestration

`RecordStore` still exists and is still useful, but it is no longer the right
mental model for the live workflow boundary. Use it as a model-local assembly
tool when it helps, not as the public contract between steps.

## Architecture In One View

PILATES currently has five integration layers:

1. `run.py`
   Thin CLI entrypoint only.
2. `pilates/runtime/launcher.py`
   Owns runtime assembly: config loading, restart/bootstrap setup, Consist
   scenario creation, yearly stage sequencing, and failure/restart guidance.
3. `pilates/workflows/stages/*.py`
   Own stage orchestration: step order, iteration loops, bindings, manifests,
   and stage-specific fallback logic.
4. `pilates/workflows/steps/*.py`
   Own step execution: component lookup, upstream typed handoff, validation,
   coupler publication, and Consist step decoration.
5. Contract layer
   Implemented primarily in:
   - `pilates/workflows/catalog.py`
   - `pilates/workflows/steps/shared.py`
   - `pilates/workflows/outputs_base.py`
   - `pilates/workflows/orchestration.py`

The key responsibility split is:

- runtime decides the scenario lifecycle
- stages decide when work runs
- step modules decide how a model phase runs
- typed outputs decide what downstream code may rely on

## The Standard Model Pattern

Most models follow the same package shape:

- `preprocessor.py`
- `runner.py`
- `postprocessor.py`
- `outputs.py`

Those components inherit from the generic bases:

- `GenericPreprocessor`
- `GenericRunner`
- `GenericPostprocessor`

The public methods are:

- `preprocess(workspace, previous_records=...)`
- `run(store, workspace, ...)`
- `postprocess(raw_outputs, workspace, ...)`

Each method should return a model-specific output type. In the current codebase,
that usually means a dataclass inheriting from `StepOutputsBase`.

## Typed Outputs Are The Workflow Boundary

Typed outputs are the main integration seam between steps.

They matter because they are used for:

- validation
- coupler publication
- manifest persistence
- cache-hit recovery
- step-to-step in-memory handoff
- onboarding readability

Current examples:

- `ActivitySimPreprocessOutputs`
- `ActivitySimRunOutputs`
- `BeamRunOutputs`
- `UrbanSimPostprocessOutputs`
- `AtlasRunOutputs`

Good typed outputs are:

- small
- explicit
- path-based
- stable across reruns and resume

Poor typed outputs are:

- raw tables or large in-memory objects
- mixed bags of arbitrary values
- thin wrappers over ad hoc filesystem conventions

## `RecordStore`: Still Useful, But Secondary

Some model code still produces or consumes `RecordStore` internally. That is
acceptable when it reduces local boilerplate around copying files or collecting
dynamic file sets.

Use that pattern only if all of the following are true:

- the `RecordStore` stays inside the model package or inside an output adapter
- the step module ultimately exposes typed outputs
- the workflow-facing semantics still live in the typed output class

Examples in the current codebase:

- output dataclasses with `from_record_store(...)`
- model-local conversion helpers for dynamic outputs

Do not design a new integration around "the step returns a `RecordStore` and
everything else figures it out later." That produces weak contracts and poor
restart behavior.

## `ModelFactory` Is The Component Registry

`pilates/generic/model_factory.py` is the registry that maps a workflow model
name to the preprocessor/runner/postprocessor classes.

This is the current pattern:

```python
"beam": {
    "preprocessor": BeamPreprocessor,
    "runner": BeamRunner,
    "postprocessor": BeamPostprocessor,
}
```

The factory also handles model variants such as:

- `activitysim_compile`
- `beam_full_skim`

If a new model or variant is not registered here, the workflow cannot execute
it, regardless of what the docs say.

## Workflow Metadata Lives In The Catalog

`pilates/workflows/catalog.py` is the static contract registry for tracked
workflow steps.

Each `WorkflowStepSpec` defines:

- step identity (`step_name`, `phase`)
- stage ownership (`stage_name`, `order`)
- typed output class (`outputs_class`)
- expected inputs and outputs
- step dependencies and holder dependencies
- optional/dynamic input-output families
- enablement flags
- provenance builder family

Current identity rule:

- `step_name` is the canonical catalog identity and matches the Consist
  `model=...` value on the decorated step
- older helper names may still say "model name", but in the catalog that is a
  compatibility alias for `step_name`, not a second independent key

This metadata is consumed by planning, validation, schema declaration, and
contract enforcement.

If a step is tracked, keep these aligned:

1. the `WorkflowStepSpec`
2. the typed outputs class
3. the `StepOutputsHolder` field
4. the step factory setter/getter logic

## `StepOutputsHolder` Is The In-Memory Handoff

`StepOutputsHolder` in `pilates/workflows/steps/shared.py` is the typed,
in-process handoff object between workflow steps.

It is not optional for tracked steps. The runtime expects step factories to
store outputs there so downstream steps can consume them without re-querying
the coupler or rebuilding state from disk.

Examples of current fields:

- `activitysim_preprocess`
- `beam_run`
- `urbansim_postprocess`
- `atlas_run`

The holder is also used for runtime dependency checks via
`validate_step_ready(...)`.

## What A Step Module Owns

Model-specific step factories live in `pilates/workflows/steps/*.py`.

The current step modules do real model-specific work. They are not thin wrappers.
Typical responsibilities are:

1. resolve the right component from `ModelFactory`
2. fetch typed upstream outputs from `StepOutputsHolder`
3. call the public component method
4. validate the returned outputs
5. publish cross-step artifacts to the coupler
6. store the outputs back on `StepOutputsHolder`
7. expose cache-recovery or output-replay hooks when needed
8. decorate the step for Consist execution

### The Standard Path: `StandardStepSpec` + `build_standard_step(...)`

Most tracked typed steps now use the narrow shared builder in
`pilates/workflows/steps/shared.py`.

That path looks like:

1. define model-local callbacks for execution and optional logging/recovery
2. package them into `StandardStepSpec`
3. return `build_standard_step(...)`

The important design constraint is that the builder stays narrow:

- it removes repeated wrapper boilerplate
- it does not centralize model-specific logging policy
- it does not replace model-local replay/recovery code
- it does not make stages generic

That is why step modules still expose `make_*_step(...)` functions instead of
raw spec objects. The factories close over a live coupler and model-local
callbacks, and schema/runtime assembly intentionally build distinct step
instances with different couplers.

Current standard examples:

- `activitysim_preprocess`
- `activitysim_run`
- `activitysim_postprocess`
- `beam_preprocess`
- `beam_run`
- `beam_postprocess`
- `beam_full_skim`
- `urbansim_preprocess`
- `urbansim_run`
- `urbansim_postprocess`
- `atlas_preprocess`
- `atlas_run`
- `atlas_postprocess`

### The Custom Path Still Exists

Not every step should use `StandardStepSpec`.

Keep a custom step when the runtime contract is meaningfully different from the
standard tracked typed-step shell. Current examples are:

- `activitysim_compile`
- `postprocessing`

Good reference modules:

- `pilates/workflows/steps/activitysim.py`
- `pilates/workflows/steps/beam.py`
- `pilates/workflows/steps/urbansim_atlas.py`
- `pilates/workflows/steps/postprocessing.py`

## What A Stage Module Owns

Stage modules in `pilates/workflows/stages/` own orchestration, not model logic.

In the current design, a stage is responsible for:

- building bindings with `build_binding_plan(...)`
- creating `StepRef`s
- choosing output-path providers or replay hooks when needed
- executing through `StageRunner` or `run_workflow(...)`
- managing loops and restart-sensitive sequencing
- consuming typed outputs from `StepOutputsHolder`

Examples:

- `land_use.py`
- `supply_demand.py`
- `vehicle_ownership.py`
- `postprocessing.py`

Do not move stage concerns into the model package. If you need iteration logic,
manifest checkpoints, or resume behavior, the stage layer is the place for it.

## Schema Steps Versus Runtime Steps

PILATES builds schema-validation steps and runtime-execution steps from the
same `make_*_step(...)` factories, but they are intentionally different
instances.

- schema assembly uses `SchemaCoupler` and the `SCHEMA_STEP_BUILDERS` registry
  in `pilates/workflows/steps/__init__.py`
- runtime assembly uses a live workflow coupler

This is why the step-builder registry remains separate from the workflow
catalog:

- the catalog is static contract metadata
- the registry is executable builder wiring

## Coupler Ownership

The coupler is the workflow-wide artifact namespace for cross-step handoffs and
restart-relevant outputs.

Use it for:

- canonical handoff files
- warm-start artifacts
- durable stage-boundary outputs
- artifacts that downstream steps may need after a restart

Do not use it as a dumping ground for internal state.

When a new artifact becomes part of the workflow contract:

1. add or reuse a key in `pilates/workflows/artifact_keys.py`
2. describe it in `pilates/workflows/coupler_schema.py`
3. publish it from the producing step
4. request it through bindings or input-resolution in the consumer

## Consist Ownership

PILATES uses Consist for provenance, cache identity, and step execution.

Important ownership rules:

- runtime code owns scenario creation and bootstrap execution
- step factories own Consist step callables and metadata
- model components should not create nested workflow lifecycles

In practice, model code may:

- prepare files in the workspace
- run containers via `GenericRunner.run_container(...)`
- build typed outputs
- attach metadata needed for downstream correctness

Model code should not:

- start its own scenario lifecycle
- bypass the shared workflow logging/publication boundary
- depend on ambient `cwd` assumptions instead of the mounted workspace

## Restart, Cache Recovery, And Manifests

The current runtime is restart-aware.

Important pieces:

- `pilates/runtime/bootstrap.py` handles initialization and bootstrap caching
- `pilates/workflows/orchestration.py` handles `StepRef` execution and cache
  recovery
- stage modules write manifests under `.workflow/`
- typed outputs are serialized/deserialized for replay and recovery

Integration consequence:

Your outputs contract must be reconstructable after a cache hit or resume. If a
step succeeds only when some undocumented side effect survives in memory, the
integration is brittle.

## Recommended Design Heuristics

Prefer:

- explicit typed paths
- stable artifact names
- model-local file logic inside the model package
- workflow logic in the stage and step layers
- validators for semantic correctness

Avoid:

- returning implicit filesystem state as the contract
- hiding required artifacts behind undocumented naming conventions
- using a new abstraction layer when existing step helpers already fit
- introducing duplicate artifact keys for the same semantic output

## Current Workflow Surface

These are the main tracked steps today:

- `urbansim_preprocess`
- `urbansim_run`
- `urbansim_postprocess`
- `atlas_preprocess`
- `atlas_run`
- `atlas_postprocess`
- `activitysim_preprocess`
- `activitysim_compile`
- `activitysim_run`
- `activitysim_postprocess`
- `beam_preprocess`
- `beam_run`
- `beam_postprocess`
- `beam_full_skim`
- `postprocessing`

Not every run executes every step. Enablement depends on the configured models
and flags in the loaded settings.

## If You Are Modifying An Existing Integration

Before changing code, identify which layer you are actually changing:

- component behavior
- typed output contract
- step publication behavior
- stage orchestration
- restart/cache behavior

Then update the matching docs and tests in the same PR. Most integration bugs
in this codebase come from contract drift between those layers rather than from
one bad line of model code.
