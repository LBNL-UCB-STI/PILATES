---
title: Output Validation
summary: Startup contract validation and runtime output validation for workflow boundaries.
---

# Output Validation

PILATES validates workflow outputs in two places:

1. **Startup contract validation** checks that the static workflow catalog,
   typed output classes, and Consist step metadata agree before a run starts.
2. **Runtime typed-output validation** checks the concrete object returned by a
   step after model execution.

Keep those responsibilities separate. Startup validation catches wiring drift.
Runtime validation catches missing files, malformed handoffs, and semantic
boundary mistakes in the outputs produced by the current run.

## Startup Contract Validation

Startup validation is run from the workflow step surface in
`pilates/workflows/steps/shared.py`. The important inputs are:

- the workflow catalog entry for the step
- the step's `StepOutputsBase` subclass from `STEP_OUTPUTS_CLASSES`
- the Consist step metadata attached by the step decorator

For tracked steps, the validator compares the canonical typed output keys with
the metadata keys. It uses `required_outputs_for_step_outputs_class()` and
`declared_outputs_for_step_outputs_class()` from
`pilates/workflows/outputs_base.py`.

Use this layer for static questions such as:

- "Does this step publish the output keys the catalog says it publishes?"
- "Did a metadata override drift from the typed output class?"
- "Are strict output-path contracts still aligned for the step?"

Do not put file-existence or cross-step runtime checks here. At startup, there
is no produced output object yet.

## Runtime Typed-Output Validation

Runtime validation happens on the object returned by the step executor. The
standard step builder in `pilates/workflows/steps/shared.py` coerces legacy
`RecordStore` results into the declared typed output class when possible, then
calls:

```python
step_outputs.validate(
    context=ValidationContext(
        settings=settings,
        state=state,
        workspace=workspace,
        step_name=step_name,
        upstream_outputs=...,
    )
)
```

`StepOutputsBase.validate()` first performs path checks:

- every field in `required_path_fields` must be present and exist
- every present field in `optional_path_fields` must exist
- every path value in each `dict_path_fields` mapping must exist

After path checks, it runs any semantic validators declared on the class.

## StepOutputsBase Fields

Output classes should inherit from `StepOutputsBase` when they represent a
workflow boundary that publishes outputs for downstream steps, Consist
metadata, restart, or replay.

The common class fields are:

- `declared_outputs`: full semantic output key set the class can publish.
- `required_outputs`: strict output key set expected to materialize at runtime.
- `required_output_families`: format strings expanded from state, for dynamic
  required keys such as year- or iteration-scoped artifacts.
- `optional_outputs`: declared keys that are allowed to be absent.
- `record_keys`: mapping from dataclass field name to semantic artifact key.
- `record_descriptions`: optional descriptions used when producing
  `FileRecord`s.
- `default_description`: fallback description for records.
- `required_path_fields`: dataclass fields whose paths must exist.
- `optional_path_fields`: dataclass fields that may be `None`, but must exist
  when set.
- `dict_path_fields`: dataclass fields containing `dict[key, Path]` values.
- `validators`: semantic `OutputValidator` instances for checks that path
  existence alone cannot express.

The dataclass instance fields should hold concrete `Path` values, optional
`Path` values, path dictionaries, or small metadata needed to reconstruct the
published records.

## Declared Versus Required Outputs

`declared_outputs` is the schema/catalog surface. It says which semantic keys
the output type may publish.

`required_outputs` is the strict runtime surface. It says which keys must be
available for Consist output tracking and cache/restart checks.

If `required_outputs` is absent, `required_outputs_for_step_outputs_class()`
falls back to expanded `required_output_families`, then to declared outputs
minus `optional_outputs`.

`pilates/workflows/orchestration.py` uses this split for tracked steps:

- `required_outputs` becomes the strict Consist `outputs` contract.
- `declared_outputs` remains available for schema/catalog publication.
- optional declared outputs are tolerated when absent.

Use this pattern when a step has optional artifacts. For example,
`BeamPreprocessOutputs` declares warmstart and vehicle handoff keys, but marks
only plans, households, and persons as required.

## Record Keys And Path Fields

`record_keys` connects Python fields to workflow artifact keys:

```python
record_keys = {
    "land_use_table": ASIM_LAND_USE_IN,
    "households_table": ASIM_HOUSEHOLDS_IN,
}
```

Path field declarations control validation, not publication. A field can be
required for local execution without being published, and a published key can be
optional when the field is `None`.

`StepOutputsBase._iter_record_items()` iterates `record_keys`, skips `None`
values, and yields `(short_name, path, description)` tuples. Override it when
the output shape is dictionary-based, as BEAM outputs do for prepared inputs and
raw outputs.

Keep these aligned:

- every required published singleton path usually needs both a `record_keys`
  entry and a `required_path_fields` entry
- optional singleton paths usually need `record_keys` plus
  `optional_path_fields`
- dictionary outputs usually need `dict_path_fields` plus a custom
  `_iter_record_items()`

## Semantic Validators

Use semantic validators when a valid filesystem path is not enough. Examples
include:

- an ActivitySim boundary table must live under the mutable data directory
- a BEAM skim artifact is required only for particular settings
- an output must be consistent with an upstream output already in the holder

A validator implements the `OutputValidator` protocol:

```python
class MyValidator:
    name = "my_output_contract"
    level = "error"  # or "warning"

    def validate(self, outputs, context):
        if ...:
            return [ValidationResult("explain the contract problem")]
        return []
```

Return `ValidationResult` objects with a concise message and optional metadata.
Raise exceptions only for truly unexpected validator failures; `validate()`
wraps validator crashes as `AssertionError`s that name the validator and step.

## ValidationContext

`ValidationContext` carries runtime facts into semantic validators:

- `settings`: current `PilatesConfig`
- `state`: current `WorkflowState` or `AtlasSubState`
- `workspace`: current `Workspace`
- `step_name`: canonical workflow step name
- `upstream_outputs`: snapshot of upstream holder outputs

Validators should treat the context as read-only. Use it to decide whether an
output is required, to find the active forecast year or iteration, or to compare
against an upstream typed output.

## Warning And Error Behavior

`OutputValidator.level` controls behavior:

- `"warning"` logs `OUTPUT VALIDATION WARNING` and allows the run to continue.
- `"error"` collects failures and raises one `AssertionError` after all
  validators run.

Unsupported levels are logged as warnings and treated as errors. Empty validator
messages are ignored. Error messages include the validator name, step name,
output class, and metadata when provided.

Use warnings for migration visibility or non-blocking diagnostics. Use errors
for contracts that downstream execution, cache correctness, restart, or replay
depends on.

## RecordStore Conversion

`to_record_store()` converts typed outputs into `FileRecord`s using
`_iter_record_items()`. This is the path to downstream publication and legacy
`RecordStore` interoperability.

Implement `from_record_store(record_store, workspace)` when a legacy runner,
cache replay, or test may return a `RecordStore` instead of the typed output
class. The standard step builder will call it automatically when the executor
returns a `RecordStore`.

Prefer explicit conversion when records need workspace-relative path
resolution, content hashes, artifact-to-path unwrapping, or derived fields such
as a mutable data directory. Simple record-key based conversion is available as
a fallback in the standard step builder, but explicit `from_record_store()`
makes the contract clearer and easier to test.

## Output Mappings

`step_output_mapping(outputs, warn_lossy=True)` returns a plain
`dict[key, path_string]` from typed outputs. It sanitizes keys, drops duplicate
keys after the first value, and is intentionally lossy because artifact identity
and content hashes are reduced to strings. Use it for path-only diagnostics,
serialization, replay checks, or tests.

`step_output_handoff_mapping(outputs, coupler=...)` is the runtime handoff
helper. For each typed output key, it prefers the current coupler-published
value and falls back to the raw filesystem path. Use this for step-to-step,
stage-to-stage, or iteration-to-iteration handoffs where Consist artifact
identity should be preserved.

Do not use `step_output_mapping()` for live workflow handoffs unless the caller
explicitly wants path-only values.

## Recommended Tests

Add focused tests near the contract being changed:

- path validation: required fields fail when missing; optional fields pass when
  `None` and fail when set to a missing path
- dictionary path validation: missing keyed paths produce actionable failures
- semantic validators: warning validators log without failing; error validators
  raise and include metadata
- context-sensitive validators: pass `ValidationContext` with settings, state,
  workspace, step name, and upstream outputs
- `declared_outputs` and `required_outputs`: assert the resolved class helpers
  return the intended keys, including optional and state-expanded cases
- `to_record_store()` and `from_record_store()`: round-trip representative
  records, including content hashes or artifact-path conversion when relevant
- mapping helpers: assert `step_output_mapping()` matches the path-only
  `RecordStore` view, and use `step_output_handoff_mapping()` in handoff tests
  when coupler values matter
- startup contract validation: update catalog/metadata alignment tests when a
  typed output class changes its canonical keys

Existing examples live in `tests/test_output_validation_backbone.py`,
`tests/test_activitysim_run_output_contract.py`,
`tests/test_step_contract_validator.py`, and
`tests/test_workflow_catalog_derivations.py`. For broader typed-output
examples, also see `tests/test_recordstore_dependency_guards.py` and
`tests/test_urbansim_atlas_typed_contracts.py`.

## Adjacent Pages

- Read [Step Contracts](../workflow/step_contracts.md) first.
- Pair this with [Adding a Model](adding_a_model.md).
- Use [Testing New Integrations](testing_new_integrations.md) for the acceptance path.

## Evidence Basis

- Typed output contracts: `pilates/workflows/outputs_base.py`
- Standard step runtime validation: `pilates/workflows/steps/shared.py`
- Consist output-policy wiring: `pilates/workflows/orchestration.py`
- Model-specific examples: `pilates/activitysim/outputs.py`,
  `pilates/beam/outputs.py`, `pilates/urbansim/outputs.py`,
  `pilates/atlas/outputs.py`
