# Model Integration Guide

This guide describes the current model-integration architecture in PILATES.
It is intentionally aligned with the simplified workflow seam, not the older
`RecordStore`-centric design.

If this guide and code disagree, code wins. Update this file in the same change
that updates the integration surface.

## Core Rule

The public workflow contract is:

- typed step outputs
- path-based artifacts
- explicit coupler publication
- explicit stage wiring

Do not design a new model around `RecordStore` as the cross-step contract.
Model-local internals may still use `RecordStore` as a temporary working
container, but that should not cross the live step-module boundary.

## Mental Model

PILATES has four layers:

1. `run.py`
   Owns scenario lifecycle, startup validation, schema-step registration,
   yearly stage sequencing, bootstrap, and restart orchestration.
2. `pilates/workflows/stages/*.py`
   Own orchestration logic for a stage: control flow, loops, `StepRef`
   assembly, input resolution, and fallback decisions.
3. `pilates/workflows/steps/*.py`
   Own step execution: model component lookup, typed output validation,
   logging, and coupler publication.
4. Contract layer
   Implemented in:
   - `pilates/workflows/catalog.py`
   - `pilates/workflows/steps/shared.py`
   - `pilates/workflows/outputs_base.py`
   - `pilates/workflows/orchestration.py`

The key division of responsibility is:

- stages decide when work runs
- step modules decide how a model phase runs
- typed outputs decide what downstream code may rely on

## Where New Model Work Lives

Most new integrations touch these areas:

1. `pilates/<model>/`
   - `preprocessor.py`
   - `runner.py`
   - `postprocessor.py`
   - `outputs.py`
2. `pilates/generic/model_factory.py`
3. `pilates/workflows/steps/<module>.py`
4. `pilates/workflows/steps/shared.py`
   For `StepOutputsHolder` and shared validation/logging utilities.
5. `pilates/workflows/catalog.py`
   For `WorkflowStepSpec` entries.
6. `pilates/workflows/stages/<stage>.py`
7. `run.py`
   Only if schema-step registration or stage sequencing changes.
8. `pilates/workflows/artifact_keys.py`
9. `pilates/workflows/coupler_schema.py`
10. tests under `tests/`

## Public Component Contract

Use the existing preprocessor/runner/postprocessor split, but follow the
current public contract:

- `preprocess(...) -> <Model>PreprocessOutputs`
- `run(...) -> <Model>RunOutputs`
- `postprocess(...) -> <Model>PostprocessOutputs`

These outputs should be typed dataclasses whose fields are usually:

- `Path`
- `dict[str, Path]`
- small metadata fields such as hashes, scenario labels, or canonical keys

Do not load large model outputs into memory just to satisfy the typed contract.
The goal is explicit structure, not dataframe transport.

### Internal `RecordStore` Use

Internal helper code may still use `RecordStore` when it genuinely simplifies
local file assembly. That is acceptable if:

- it stays inside the model package
- it does not become the public return type of the live step path
- typed outputs remain the step boundary

If a bridge helper such as `from_record_store(...)` exists only for tests or a
local helper path, keep it clearly secondary to the typed public method.

## Typed Outputs

Define one typed output dataclass per tracked step phase in `outputs.py`.

Typical requirements:

- inherit from `StepOutputsBase` for tracked workflow steps
- declare stable output keys with `declared_outputs`
- map dataclass fields to artifact keys with `record_keys`
- mark required vs optional path fields
- add semantic validators when path existence is not enough

Example shape:

```python
@dataclass
class FreightRunOutputs(StepOutputsBase):
    primary_output_attr: ClassVar[str] = "output_dir"
    declared_outputs: ClassVar[Tuple[str, ...]] = ("freight_events",)
    record_keys: ClassVar[Dict[str, str]] = {"events_path": "freight_events"}
    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_dir",)
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("events_path",)

    output_dir: Path
    events_path: Optional[Path] = None
```

Guidelines:

- prefer explicit fields over opaque “bag of outputs” mappings
- keep metadata explicit when downstream logic depends on it
- keep coupler keys stable; do not rename for aesthetics

## `StepOutputsHolder`

`StepOutputsHolder` in `pilates/workflows/steps/shared.py` is the typed
in-memory handoff between steps. It is not an optional pattern and not a design
question for new integrations.

When you add a tracked step, add a corresponding holder field and keep it
aligned with the catalog entry for that step.

## Catalog Metadata

`pilates/workflows/catalog.py` is the source of truth for tracked workflow-step
metadata. Each `WorkflowStepSpec` defines:

- step name
- model name
- phase
- stage name
- order
- output class
- dependency metadata
- enablement metadata
- provenance builder family

Keep the following aligned:

1. `WorkflowStepSpec.outputs_class`
2. `StepOutputsHolder` field name
3. step factory output setter
4. actual typed output dataclass

Startup validation via `validate_workflow_step_contracts(...)` is expected to
fail fast if those drift.

## Step Factories

Model-specific step factories live in `pilates/workflows/steps/*.py`.

The common pattern is:

1. resolve the component through `ModelFactory`
2. execute the component public method
3. require a typed output object
4. validate it with `ValidationContext`
5. store it on `StepOutputsHolder`
6. log/publish artifacts through shared helpers
7. decorate the function for Consist

The active ActivitySim, BEAM, and UrbanSim/ATLAS step modules are the current
templates:

- `pilates/workflows/steps/activitysim.py`
- `pilates/workflows/steps/beam.py`
- `pilates/workflows/steps/urbansim_atlas.py`

Do not copy old designs that relied on a generic `step_exec.py` executor
surface. That shim is gone.

## Stages And `StepRef`

Stage modules assemble `StepRef`s and call the orchestration helpers.

A stage is responsible for:

- deciding step order
- resolving explicit inputs and fallback inputs
- passing `input_keys` when coupler resolution is required
- handling iteration loops and restart-specific decisions

Use the input-resolution helpers instead of hand-rolling precedence. The stage
should consume upstream typed outputs from `StepOutputsHolder` and only fall
back to coupler or explicit paths when that is genuinely required by the stage
contract.

## Coupler Ownership

The coupler is for published cross-step artifacts, not for arbitrary hidden
runtime state.

Keep coupler logic in the existing boundaries:

- `pilates/workflows/input_resolution.py`
- `pilates/utils/coupler_helpers.py`
- step-module output loggers
- limited orchestration recovery logic in `pilates/workflows/orchestration.py`

Do not make stage modules into ad hoc coupler mutation layers.

When a new artifact becomes a real cross-step dependency:

1. add or reuse an artifact key in `pilates/workflows/artifact_keys.py`
2. add schema description in `pilates/workflows/coupler_schema.py`
3. publish it from the producing step
4. request it from the consuming stage/step

## Manifest And Recovery

Manifest persistence and cache-hit recovery are typed-output based.

That means:

- tracked outputs must serialize cleanly as typed outputs
- recovery should rebuild the same typed output object shape
- step-local replay hooks should preserve logging/publication parity

If your model needs special recovery behavior, add it at the step layer or in
orchestration recovery hooks. Do not introduce a parallel legacy path.

## Consist Ownership

Run lifecycle belongs to `run.py` and Consist scenario/step contexts.

Model code must not:

- start nested Consist runs
- directly own workflow-stage orchestration
- bypass shared logging/publication boundaries

Model code may:

- perform model-local file preparation
- run containers
- build typed outputs
- attach explicit metadata needed for downstream correctness

## Recommended Integration Sequence

1. Scaffold the model package and step module if that saves time.
2. Implement typed outputs in `pilates/<model>/outputs.py`.
3. Implement public component methods that return typed, path-based outputs.
4. Register model components in `pilates/generic/model_factory.py`.
5. Add tracked step metadata in `pilates/workflows/catalog.py`.
6. Add `StepOutputsHolder` fields in `pilates/workflows/steps/shared.py`.
7. Implement step factories in `pilates/workflows/steps/<module>.py`.
8. Wire the stage in `pilates/workflows/stages/<stage>.py`.
9. Add coupler keys/schema only for true cross-step artifacts.
10. Add focused tests for:
   - typed public method behavior
   - step-factory wiring
   - coupler publication
   - cache/recovery parity if relevant

## What Good Integrations Look Like

Good new integrations have these properties:

- public methods return typed outputs directly
- step modules reject wrong upstream types early
- coupler publication is explicit and minimal
- restart/cache behavior preserves typed-output parity
- model-local helpers stay local
- no dual-path step execution

## Common Failure Modes

Avoid these:

- using `RecordStore` as the public cross-step return type
- rebuilding old generic executor abstractions
- hiding required metadata in workspace side caches
- mutating coupler state from stage code without shared helpers
- documenting a contract that the runtime does not actually enforce
- broadening a model integration into unrelated workflow cleanup

## References

- `docs/adding_a_model.md`
- `docs/workflow_primer.md`
- `pilates/workflows/catalog.py`
- `pilates/workflows/steps/shared.py`
- `pilates/workflows/orchestration.py`
