---
title: Testing New Integrations
summary: Tests and acceptance criteria for new workflow and model integration work.
---

# Testing New Integrations

## Minimum Test Layers

For a new model or workflow integration, cover all of these layers before treating the work as complete:

1. contract tests for semantic input roles, typed outputs, and coupler keys
2. stage/runtime tests for the native execution path that calls the model
3. architecture guardrail coverage when the change touches workflow seams
4. focused docs/scaffold review so the contributor path matches the code

## What To Prove

- The typed outputs declare the paths and record keys the step actually publishes.
- The catalog metadata matches the decorated callable and `StepDefinition`.
- The resolver selects each semantic role once and returns `ResolvedStepInputs`.
- Stage code executes the `StepDefinition` through `execute_step(...)`, then sequences its typed projected outputs into the next step.
- No second binding pass, in-memory holder, or profile-era authority path is introduced.

## Recommended Test Targets

- `tests/test_stage_contracts.py`
- `tests/test_step_execution.py`
- `tests/test_step_definitions.py`
- `tests/test_step_execution_architecture.py`
- `tests/test_architecture_guardrails.py`
- any model-specific tests for the new integration

## Architecture Rules for Contributors

- If your change needs a new semantic role or output, update the resolver, decorated callable, projector, and `StepDefinition` together.
- Keep the native sequence explicit: define semantic roles, resolve them once, execute through Consist, project `RunResult.outputs` into typed PILATES outputs, then sequence the next step.
- Prefer direct `execute_step(...)` fixtures over wrappers that reconstruct inputs or outputs.

## Scaffold and Docs Check

If you used `scripts/new_model_scaffold.py`, review the generated artifacts before merging:

- step module contains a decorated callable, resolver, projector, and `StepDefinition`
- stage snippets use explicit `execute_step(...)` calls and typed projected handoffs
- checklist still points at the right stage, tests, and docs
- contributor docs under `docs/extend/` still describe the same mental model your code implements

## Adjacent Pages

- Pair this with [Adding a Model](adding_a_model.md).
- Read [Model Integration Guide](model_integration_guide.md) for the architecture rules.
- Use [Model Contract Checklist](model_contract_checklist.md) as the completion gate.
- Use [Output Validation](output_validation.md) for contract-specific checks.
- Use [Operations Overview](../operations/overview.md) for preserved test-output workflows.
