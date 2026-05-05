---
title: Testing New Integrations
summary: Tests and acceptance criteria for new workflow and model integration work.
---

# Testing New Integrations

## Minimum Test Layers

For a new model or workflow integration, cover all of these layers before treating the work as complete:

1. contract tests for typed outputs, holder publication, and coupler keys
2. stage/runtime tests for the orchestration path that calls the model
3. architecture guardrail coverage when the change touches workflow seams
4. focused docs/scaffold review so the contributor path matches the code

## What To Prove

- The typed outputs declare the paths and record keys the step actually publishes.
- The catalog metadata matches the real runtime contract.
- Binding resolves the expected inputs for the active surface shape.
- Stage code uses the current orchestration path (`WorkflowRuntimeContext`, `StageRunner`, `build_binding_plan(...)`) instead of older ad hoc resolution patterns.
- No new profile-era authority path is introduced.

## Recommended Test Targets

- `tests/test_stage_contracts.py`
- `tests/test_step_contract_validator.py`
- `tests/test_workflow_binding.py`
- `tests/test_workflow_step_metadata.py`
- `tests/test_architecture_guardrails.py`
- any model-specific tests for the new integration

## Architecture Rules for Contributors

- If your change requires a new exception to the surface-first runtime rules, update the guardrail tests in the same change.
- Do not add tests that normalize old `WorkflowProfile`-driven behavior back into production paths.
- Prefer explicit surface/context fixtures over raw-settings stubs when testing workflow runtime behavior.

## Scaffold and Docs Check

If you used `scripts/new_model_scaffold.py`, review the generated artifacts before merging:

- step module uses `StandardStepSpec` / `build_standard_step()`
- stage snippets use `WorkflowRuntimeContext`, `StageRunner`, and `build_binding_plan(...)`
- checklist still points at the right stage, tests, and docs
- contributor docs under `docs/extend/` still describe the same mental model your code implements

## Adjacent Pages

- Pair this with [Adding a Model](adding_a_model.md).
- Read [Model Integration Guide](model_integration_guide.md) for the architecture rules.
- Use [Model Contract Checklist](model_contract_checklist.md) as the completion gate.
- Use [Output Validation](output_validation.md) for contract-specific checks.
- Use [Operations Overview](../operations/overview.md) for preserved test-output workflows.
