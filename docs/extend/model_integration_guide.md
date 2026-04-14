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
- the step factory shell that binds the live coupler and outputs holder
- validation that the step returned the expected typed output class
- publication of the typed outputs into the holder and coupler

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

### Typed outputs and publication

Model code returns a typed outputs object that extends `StepOutputsBase`. The step factory then:

- validates required output paths and semantic validators
- stores the typed outputs in `StepOutputsHolder`
- publishes record-store entries or coupler artifacts as the step logic requires

### Why the model module still owns callbacks

`build_standard_step()` stays thin. The model modules still provide the callbacks for input logging, output logging, replay, and recovery because those callbacks close over model-local behavior and a live coupler. That keeps the shared shell reusable without hiding model-specific wiring.

## Reading Path

- Read [Workflow Primer](../workflow/workflow_primer.md) and [Step Contracts](../workflow/step_contracts.md) first.
- Continue to [Adding a Model](adding_a_model.md) for the hands-on path.
- Use [Output Validation](output_validation.md) for runtime and startup guardrails.

## Evidence Basis

- Model dispatch: `pilates/generic/model_factory.py`
- Shared step shell: `pilates/workflows/steps/shared.py`
- Catalog and contract metadata: `pilates/workflows/catalog.py`
- Model-specific step wiring under `pilates/workflows/steps/`
- Registry and factory tests: `tests/test_generic_modules.py`, `tests/test_coupler_key_invariants.py`, `tests/test_standard_step_builder.py`
