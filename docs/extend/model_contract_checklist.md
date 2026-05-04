---
title: Model Contract Checklist
summary: Short contributor checklist for new tracked workflow integrations.
---

# Model Contract Checklist

## What PILATES Needs Before a Step Is Ready

- A typed outputs class that declares the paths and record keys it publishes.
- A `WorkflowStepSpec` entry that names the step, stage, inputs, outputs, and dependencies.
- Consist step metadata for cache/facet identity and any model config adapter.
- Binding rules for custom source precedence, optional fallbacks, or restart/replay sources.
- A step factory that binds a live coupler and `StepOutputsHolder`.
- A `ModelFactory` registry entry for the model name or phase alias.
- A test that proves the declared outputs, runtime outputs, and coupler keys stay aligned.

## Checklist

### Output class

- Declare the output paths as `StepOutputsBase` fields.
- Set `record_keys` for any artifacts that downstream steps or replay logic consume.
- Set `required_path_fields` for paths that must exist after the step runs.
- Add semantic validators only when the contract needs cross-field or cross-step checks.
- Use a new snapshot artifact key only when the workflow needs a new semantic
  boundary state. Archive/recovery locations are metadata for an existing
  artifact, not new keys by themselves.

### Catalog entry

- Add or update the `WorkflowStepSpec`.
- Keep `input_keys`, `optional_input_keys`, `output_keys`, and `optional_output_keys` aligned with the real step behavior.
- Keep `depends_on` and `holder_inputs` aligned with the runtime ordering that the step actually needs.
- Add dynamic families only when the step publishes or consumes dynamically named artifacts.

### Consist metadata

- Update `pilates/workflows/step_consist_meta.py` when the step needs a config
  adapter, scalar cache identity, facets, facet indexing, or identity inputs.
- Use config adapters for model configuration trees that should affect cache
  identity.
- Declare config reference policies when config paths are delegated to workflow
  artifacts, optional/dormant, or runtime outputs.
- Keep archive and recovery-root metadata out of step identity unless the bytes
  themselves are intended cache inputs.

### Step factory

- Use `build_standard_step()` unless the step needs custom orchestration.
- Bind the live coupler and `StepOutputsHolder` from the factory call site.
- Keep model-local logging and recovery callbacks in the model module.
- Make sure the step returns the expected typed outputs class.

### Stage wiring

- Use `WorkflowRuntimeContext` at the stage/orchestration boundary when the stage needs `settings`, `state`, `surface`, and `workspace` together.
- Let `StageRunner` own step execution plumbing.
- Use `build_binding_plan(...)` when the step needs resolved runtime inputs or fallback behavior.
- Register custom `ArtifactBindingRule` behavior in `pilates/workflows/binding.py`
  instead of repeating precedence chains in stages or model modules.
- Keep enablement decisions on the enabled workflow surface, not in stage-local boolean checks copied from raw settings.

### Architecture rules

- Do not import or rebuild `WorkflowProfile` in production model integration code.
- Do not call `build_enabled_workflow_surface(...).profile` as a shortcut for booleans.
- Keep runtime flag initialization in the approved workflow/runtime entry points only.
- Treat the coupler as the current semantic-role map, not a historical artifact store.
- Treat architecture guardrail tests as part of the integration contract, not optional lint.

### Registry and tests

- Register the model component classes in `ModelFactory`.
- Add or update contract tests that pin the output keys, holder publication, and coupler keys.
- Add or update Consist metadata/config adapter tests when changing
  `step_consist_meta.py`.
- Run the workflow contract validator tests before treating the integration as complete.
- If the integration introduces a new allowed seam, update the architecture guardrail tests explicitly.

## Adjacent Pages

- Read [Model Integration Guide](model_integration_guide.md) first.
- Use [Adding a Model](adding_a_model.md) for implementation order.
- Pair this with [Step Contracts](../workflow/step_contracts.md) and
  [Artifact Semantics](../workflow/artifact_semantics.md).
- Use [Output Validation](output_validation.md) for guardrails.

## Evidence Basis

- Contract metadata: `pilates/workflows/catalog.py`
- Consist metadata and config adapter policy: `pilates/workflows/step_consist_meta.py`
- Binding rules: `pilates/workflows/binding.py`
- Typed outputs and validation: `pilates/workflows/outputs_base.py`
- Step shell and publication: `pilates/workflows/steps/shared.py`
- Registry and adapter dispatch: `pilates/generic/model_factory.py`
- Contract drift tests: `tests/test_step_contract_validator.py`, `tests/test_workflow_catalog_contracts.py`, `tests/test_workflow_binding.py`, `tests/test_coupler_key_invariants.py`
