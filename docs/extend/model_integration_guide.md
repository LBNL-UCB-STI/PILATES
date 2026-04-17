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

### Binding owns source precedence

When a step needs inputs from coupler state, explicit upstream outputs, or fallbacks, use binding.

In practice that means:

- stage code builds a `BindingPlan`
- the enabled workflow surface tells binding which inputs are active for the current run shape
- model modules do not manually recreate fallback chains with ad hoc `coupler.get(...)` logic

This keeps input precedence, optionality, and restart-sensitive behavior in one place.

### Stages own sequencing

Stages decide when a step runs, not whether it is enabled in the abstract and not how its inputs should be prioritized.

The normal orchestration shape is:

- build or receive a `WorkflowRuntimeContext`
- create a `StageRunner`
- create a `BindingPlan`
- run a `StepRef`

That boundary is intentional. It keeps stage modules readable and stops policy from scattering across model packages.

### Typed outputs and publication

Model code returns a typed outputs object that extends `StepOutputsBase`. The step factory then:

- validates required output paths and semantic validators
- stores the typed outputs in `StepOutputsHolder`
- publishes record-store entries or coupler artifacts as the step logic requires

### Why the model module still owns callbacks

`build_standard_step()` stays thin. The model modules still provide the callbacks for input logging, output logging, replay, and recovery because those callbacks close over model-local behavior and a live coupler. That keeps the shared shell reusable without hiding model-specific wiring.

## Reading Path

- Read [Workflow Primer](../workflow/workflow_primer.md) and [Step Contracts](../workflow/step_contracts.md) first.
- Read [Model Boundaries](../reference/model_boundaries.md) when you need the per-model I/O and handoff picture before touching code.
- Continue to [Adding a Model](adding_a_model.md) for the hands-on path.
- Use [Output Validation](output_validation.md) for runtime and startup guardrails.

## How To Trace An Existing Model Boundary

When you are new to a model family, use this order:

1. read the relevant section in [Model Boundaries](../reference/model_boundaries.md)
2. inspect the static contract in `pilates/workflows/catalog.py`
3. inspect the step factory in `pilates/workflows/steps/*.py`
4. inspect the binding or input-selection helper
5. inspect the stage module that actually runs the step

That sequence answers five different questions cleanly:

- what the workflow says the step needs
- what the typed execution boundary looks like
- how runtime source precedence is resolved
- where the step runs in the stage loop
- what later stages consume next

## Evidence Basis

- Model dispatch: `pilates/generic/model_factory.py`
- Runtime context: `pilates/runtime/context.py`
- Shared step shell: `pilates/workflows/steps/shared.py`
- Binding and source precedence: `pilates/workflows/binding.py`
- Catalog and contract metadata: `pilates/workflows/catalog.py`
- Surface-driven enablement: `pilates/workflows/surface.py`
- Model-specific step wiring under `pilates/workflows/steps/`
- Registry and factory tests: `tests/test_generic_modules.py`, `tests/test_coupler_key_invariants.py`, `tests/test_standard_step_builder.py`
