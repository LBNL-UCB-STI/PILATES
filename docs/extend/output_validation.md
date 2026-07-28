---
title: Output Validation
summary: Typed-output validation at native PILATES–Consist step boundaries.
---

# Output Validation

PILATES validates a workflow boundary in two complementary places:

1. **Definition validation** checks the committed `StepDefinition`, its
   decorated Consist callable, and the policy catalog before a run starts.
2. **Typed-output validation** checks the concrete values projected from
   `RunResult.outputs` after Consist has executed a step or admitted a cache
   hit.

Definition validation detects wiring drift. Typed-output validation detects a
missing destination, malformed handoff, or model-specific semantic error in the
current boundary result. Neither layer selects historical outputs or rebuilds a
previous execution.

## Native boundary path

Every enabled model phase follows one path:

```text
semantic roles
-> ResolvedStepInputs
-> execute_step(..., StepDefinition)
-> Consist RunResult.outputs
-> TypedOutputProjector
-> validated typed PILATES output
```

The resolver selects the current coupler artifacts once and records their exact
destinations. `execute_step()` requires that selection to be complete, passes it
to Consist, and gives the persisted output links to the projector. A projector
must validate the declared current destinations and return the typed object the
following stage consumes.

The same path applies to fresh execution, an admitted cache hit, and the sole
committed mid-stage restart (`beam_run_completed -> beam_postprocess`).

## Typed outputs

Output classes inherit from `StepOutputsBase` when they represent a public
workflow handoff. Their dataclass fields hold concrete `Path` values, optional
paths, path dictionaries, and any small metadata needed by semantic validators.

The common class fields are:

- `declared_outputs`: semantic keys the output type can publish.
- `required_outputs`: keys required at the boundary.
- `required_output_families`: state-expanded required-key families.
- `optional_outputs`: declared keys allowed to be absent.
- `record_keys`: mapping from dataclass field name to semantic artifact key.
- `required_path_fields`, `optional_path_fields`, and `dict_path_fields`:
  filesystem paths that the output object must validate.
- `validators`: semantic checks that go beyond path existence.

Keep a key declaration and its path validation aligned. A required published
singleton normally has both a `record_keys` entry and a
`required_path_fields` entry. Dictionary-shaped outputs normally declare
`dict_path_fields` and customize their record iteration.

`declared_outputs` describes the schema surface; `required_outputs` describes
the strict boundary surface. Optional declared outputs may be absent, but a
required projected output must be linked by Consist and validate at its
declared destination.

## Semantic validators

Use a semantic validator when a filesystem path alone is insufficient. Examples
include an ActivitySim table that must live under the mutable data directory, a
skim required only for particular settings, or a value that must agree with an
explicit upstream typed result.

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

`ValidationContext` supplies the active settings, state, workspace, and
canonical step name. Treat it as read-only. It exists for a validator to assess
the boundary result, not to perform another input resolution.

Warnings are logged and allow the run to continue. Errors are collected and
reported together as an `AssertionError`. Raise directly only for an unexpected
validator failure.

## Handoff mappings

`step_output_mapping(outputs)` is a lossy key-to-path diagnostic view. Do not
use it for a live workflow handoff because it discards artifact identity.

`step_output_handoff_mapping(outputs, coupler=...)` is the path-oriented helper
for a boundary that must preserve a current coupler artifact when one exists.
Native steps still receive their persisted output contract through
`RunResult.outputs` and the projector; a mapping is not a second execution or
restart mechanism.

## Recommended tests

Add focused tests next to the boundary being changed:

- required path fields fail when missing; optional fields pass when absent and
  fail when set to a missing path;
- dictionary path fields report the missing key and path;
- semantic validator warnings log without failing, while errors identify the
  validator and step;
- a projector rejects a missing Consist output link or a wrong final
  destination;
- declared, required, optional, and state-expanded output keys resolve as
  intended;
- a fresh run, admitted cache hit, and supported checkpoint hand back the same
  typed output contract.

Existing examples include `tests/test_output_validation_backbone.py`,
`tests/test_activitysim_run_output_contract.py`,
`tests/test_step_contract_validator.py`, and
`tests/test_urbansim_atlas_typed_contracts.py`.

## Adjacent pages

- Read [Step Contracts](../workflow/step_contracts.md) first.
- Pair this with [Adding a Model](adding_a_model.md).
- Use [Testing New Integrations](testing_new_integrations.md) for the acceptance path.

## Evidence basis

- Native execution: `pilates/workflows/step_execution.py`
- Contract type: `pilates/workflows/step_definition.py`
- Output projector protocol: `pilates/workflows/output_projection.py`
- Typed-output base: `pilates/workflows/outputs_base.py`
- Model definitions: `pilates/workflows/steps/`
