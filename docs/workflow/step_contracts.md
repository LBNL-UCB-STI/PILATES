---
title: Step Contracts
summary: Semantic workflow contract layer built from step specs, typed outputs, and artifact keys.
---

# Step Contracts

## What PILATES Does

### WorkflowStepSpec

`WorkflowStepSpec` is the static catalog entry for one workflow step. It stores the step identity and contract metadata, including:

- `step_name`, `phase`, `stage_name`, and `order`
- `input_keys` and `optional_input_keys`
- `output_keys` and `optional_output_keys`
- `dynamic_input_families` and `dynamic_output_families`
- `depends_on`, `holder_inputs`, and `upstream_step_inputs`
- optional enablement flags and provenance builder keys

The catalog keeps `step_name` as the canonical identifier. `model_name` remains only a compatibility alias.

### Typed outputs

`StepOutputsBase` is the base class for typed step outputs. A subclass declares:

- `record_keys` for the workflow keys it publishes into a `RecordStore`
- `required_path_fields` and `optional_path_fields` for existence checks
- `dict_path_fields` for nested path maps
- `declared_outputs`, `required_outputs`, and `required_output_families` for contract metadata
- semantic `validators` for cross-field or cross-step checks

`declared_output_keys()` resolves the published output keys. `required_output_keys(state)` resolves the strict runtime set, expanding any state-driven families when present.

### Validation and publication

`validate()` checks that required paths exist, optional paths exist when present, and dict path entries exist. If validators are attached, it passes a `ValidationContext` that can include the current settings, state, workspace, step name, and a snapshot of upstream holder outputs.

`to_record_store()` turns the typed output object into a `RecordStore` for downstream publication.

`step_output_mapping()` is the path-only view of a step output object and is lossy by design. `step_output_handoff_mapping()` is the runtime handoff view: it prefers the coupler-published artifact value for a key and falls back to the filesystem path when the coupler has no value.

### Key matching

`workflow_step_key_match()` canonicalizes raw keys before checking whether they are declared. It compares the canonical key against:

- declared input keys
- optional input keys
- dynamic input families
- declared output keys
- optional output keys
- dynamic output families

This is the logic that keeps alias keys, dynamic families, and declared workflow keys aligned.

### StandardStepSpec and build_standard_step

`StandardStepSpec` is the declarative shell used by the step factories in `pilates/workflows/steps/*.py`. It describes the step name, model name, phase, outputs class, component getter, component executor, logging hooks, and contract metadata.

`build_standard_step()` binds a live coupler plus a `StepOutputsHolder` into a callable step. It:

- instantiates `ModelFactory`
- gets the model component for the step
- runs the model-local executor
- coerces `RecordStore` results into typed outputs when needed
- validates the typed outputs
- stores the result in the holder
- publishes the step output replayer hook

The default holder slot is the step name unless the spec overrides `outputs_holder_key`.

## Adjacent Pages

- Read [Stages and Steps](stages_and_steps.md) first.
- Continue to [Model Integration Guide](../extend/model_integration_guide.md) and [Output Validation](../extend/output_validation.md).
- Pair this with [Artifact Semantics](artifact_semantics.md).

## Evidence Basis

- Static step metadata and key matching: `pilates/workflows/catalog.py`
- Typed output base and validation: `pilates/workflows/outputs_base.py`
- Standard step shell and holder publication: `pilates/workflows/steps/shared.py`
- Contract alignment tests: `tests/test_step_contract_validator.py`, `tests/test_workflow_catalog_contracts.py`, `tests/test_standard_step_builder.py`, `tests/test_output_validation_backbone.py`
