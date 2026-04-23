---
title: Adding a Model
summary: Hands-on checklist for wiring a new model or model phase into PILATES.
---

# Adding a Model

## Reading Path

- Read [Model Integration Guide](model_integration_guide.md) first.
- Pair this with [Model Contract Checklist](model_contract_checklist.md).
- Finish with [Testing New Integrations](testing_new_integrations.md).

## Practical Order

For most contributors, the least confusing order is:

1. start from `scripts/new_model_scaffold.py`
2. define the typed outputs and public workflow keys
3. add or update the catalog entry
4. wire the step factory and holder publication
5. connect the stage that will call the step
6. add validation, guardrail coverage, and docs before calling the integration done

That order is intentional: the scaffold and the workflow catalog own the
contract surface, while model-local modules own implementation details.
If you are wondering "where do I add X?", use this split:

- model-local behavior lives under `pilates/<model>/`
- public workflow keys and typed outputs live in `pilates/<model>/outputs.py`
- execution metadata lives in `pilates/workflows/catalog.py`
- stage sequencing stays in `pilates/workflows/stages/`
- step shells stay in `pilates/workflows/steps/`

## Preferred Starting Point

Use `scripts/new_model_scaffold.py` first unless you are touching an existing model.

The scaffold gives you the current expected shape:

- typed outputs classes
- `WorkflowStepSpec` catalog entries
- `StandardStepSpec` / `build_standard_step()` workflow step shells
- `SCHEMA_STEP_BUILDERS` registration
- stage snippets built around `WorkflowRuntimeContext`, `StageRunner`, and `build_binding_plan(...)`
- a checklist artifact under `docs/checklists/`

That path is intentionally more declarative than wiring a model by hand across multiple modules.

## Architecture Rules for Contributors

- Enablement comes from the enabled workflow surface, not from reconstructing runtime flags or `WorkflowProfile` in model code.
- Runtime source precedence and fallback belong in binding.
- Stages own sequencing only; they should not become miniature input-resolution engines.
- Typed outputs own boundary validation and publication shape.
- Restart-sensitive contract decisions belong in workflow/runtime code, not model-local modules.

## Minimal Wiring Map

For a new model, the normal set of files to touch is:

- `pilates/<model>/outputs.py`
- `pilates/<model>/preprocessor.py`, `runner.py`, and/or `postprocessor.py`
- `pilates/generic/model_factory.py`
- `pilates/workflows/catalog.py`
- `pilates/workflows/steps/<model>.py`
- `pilates/workflows/steps/__init__.py`
- one stage module under `pilates/workflows/stages/`
- focused tests under `tests/`

If your change needs more than that, pause and check whether workflow policy is leaking into the model package.

## One-Hop Checklist

Use this quick checklist after the scaffold is in place:

1. Define the typed outputs and confirm the public artifact keys are explicit.
2. Register the model classes in `pilates/generic/model_factory.py`.
3. Add or update the `WorkflowStepSpec` entry in `pilates/workflows/catalog.py`.
4. Build the step shell in `pilates/workflows/steps/<model>.py` and export it from `pilates/workflows/steps/__init__.py`.
5. Wire exactly one stage boundary that calls the step.
6. Add focused contract tests for the stage boundary and any restart-sensitive outputs.
7. Update the relevant docs page so the new boundary is discoverable without code spelunking.
