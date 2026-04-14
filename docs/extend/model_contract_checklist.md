---
title: Model Contract Checklist
summary: Short contributor checklist for new tracked workflow integrations.
---

# Model Contract Checklist

## What PILATES Needs Before a Step Is Ready

- A typed outputs class that declares the paths and record keys it publishes.
- A `WorkflowStepSpec` entry that names the step, stage, inputs, outputs, and dependencies.
- A step factory that binds a live coupler and `StepOutputsHolder`.
- A `ModelFactory` registry entry for the model name or phase alias.
- A test that proves the declared outputs, runtime outputs, and coupler keys stay aligned.

## Checklist

### Output class

- Declare the output paths as `StepOutputsBase` fields.
- Set `record_keys` for any artifacts that downstream steps or replay logic consume.
- Set `required_path_fields` for paths that must exist after the step runs.
- Add semantic validators only when the contract needs cross-field or cross-step checks.

### Catalog entry

- Add or update the `WorkflowStepSpec`.
- Keep `input_keys`, `optional_input_keys`, `output_keys`, and `optional_output_keys` aligned with the real step behavior.
- Keep `depends_on` and `holder_inputs` aligned with the runtime ordering that the step actually needs.
- Add dynamic families only when the step publishes or consumes dynamically named artifacts.

### Step factory

- Use `build_standard_step()` unless the step needs custom orchestration.
- Bind the live coupler and `StepOutputsHolder` from the factory call site.
- Keep model-local logging and recovery callbacks in the model module.
- Make sure the step returns the expected typed outputs class.

### Registry and tests

- Register the model component classes in `ModelFactory`.
- Add or update contract tests that pin the output keys, holder publication, and coupler keys.
- Run the workflow contract validator tests before treating the integration as complete.

## Adjacent Pages

- Read [Model Integration Guide](model_integration_guide.md) first.
- Use [Adding a Model](adding_a_model.md) for implementation order.
- Use [Output Validation](output_validation.md) for guardrails.

## Evidence Basis

- Contract metadata: `pilates/workflows/catalog.py`
- Typed outputs and validation: `pilates/workflows/outputs_base.py`
- Step shell and publication: `pilates/workflows/steps/shared.py`
- Registry and adapter dispatch: `pilates/generic/model_factory.py`
- Contract drift tests: `tests/test_step_contract_validator.py`, `tests/test_workflow_catalog_contracts.py`, `tests/test_workflow_binding.py`, `tests/test_coupler_key_invariants.py`
