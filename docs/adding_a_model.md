# Adding a Model to PILATES

This guide is the practical checklist for adding a new model or a new model
variant to the current PILATES workflow.

Use it with:

- `docs/model_integration_guide.md` for architecture and design constraints
- `docs/workflow_primer.md` for the runtime and orchestration model

If this guide and the code disagree, the code wins. Update this file in the
same change that alters the integration surface.

## What "adding a model" means in PILATES

In the current codebase, a model integration usually means all of the following:

1. A model package under `pilates/<model>/` with a preprocessor, runner,
   postprocessor, and typed output classes.
2. Factory registration in `pilates/generic/model_factory.py`.
3. Workflow step metadata in `pilates/workflows/catalog.py`.
4. A `StepOutputsHolder` field in `pilates/workflows/steps/shared.py`.
5. One or more step factories in `pilates/workflows/steps/<model>.py`.
6. Stage wiring in `pilates/workflows/stages/<stage>.py`.
7. Artifact-key and coupler-schema additions for any new cross-step handoffs.
8. Tests that cover the typed contract and at least one realistic execution path.

If you are only adding a variant of an existing model, you may not need all of
those pieces. For example, `activitysim_compile` and `beam_full_skim` reuse the
main model package and add only a variant runner plus workflow wiring.

## Before You Start

Decide these up front:

1. Which major stage owns the model?
   Current major stages come from `WorkflowState.Stage` and the runtime stage
   runners:
   - `land_use`
   - `vehicle_ownership_model`
   - `supply_demand_loop`
   - `postprocessing`
2. Which workflow phases does the model need?
   The standard split is:
   - `preprocess`
   - `run`
   - `postprocess`
   Add an extra step only when it has a distinct contract or caching identity.
3. Which outputs are true cross-step artifacts?
   Only those outputs need stable artifact keys and coupler publication.
4. What should the typed output contract be?
   Prefer a small dataclass of `Path` values plus a few explicit metadata
   fields over opaque dicts or raw in-memory tables.
5. Does the model run in a container?
   If yes, plan how inputs, output paths, and mount roots will be resolved from
   `Workspace` rather than from `cwd`.

## Files You Will Usually Touch

Most integrations change these files:

1. `pilates/<model>/preprocessor.py`
2. `pilates/<model>/runner.py`
3. `pilates/<model>/postprocessor.py`
4. `pilates/<model>/outputs.py`
5. `pilates/generic/model_factory.py`
6. `pilates/workflows/catalog.py`
7. `pilates/workflows/steps/shared.py`
8. `pilates/workflows/steps/<model>.py`
9. `pilates/workflows/steps/__init__.py`
10. `pilates/workflows/stages/<stage>.py`
11. `pilates/workflows/artifact_keys.py`
12. `pilates/workflows/coupler_schema.py`
13. tests under `tests/`

You only need to touch `pilates/runtime/launcher.py`, `workflow_state.py`, or
`pilates/workflows/stages/__init__.py` if you are introducing a brand-new
top-level stage or changing the yearly stage order.

## Recommended Implementation Order

### 1. Implement the model package

Start in `pilates/<model>/`.

Follow the standard component pattern:

- `GenericPreprocessor.preprocess(...) -> <Model>PreprocessOutputs`
- `GenericRunner.run(...) -> <Model>RunOutputs`
- `GenericPostprocessor.postprocess(...) -> <Model>PostprocessOutputs`

These public methods should return typed outputs. Internals can still use
`RecordStore` where it simplifies local file handling, but the live workflow
boundary should be typed outputs.

### 2. Define typed outputs in `outputs.py`

Each tracked workflow step should have a typed dataclass, usually inheriting
from `StepOutputsBase`.

Typical design rules:

- use explicit `Path` fields for important artifacts
- use `dict[str, Path]` only when the output family is open-ended
- declare stable workflow-facing keys in `declared_outputs`
- map dataclass fields to coupler-facing keys with `record_keys` when useful
- define `required_path_fields`, `optional_path_fields`, and validators

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

Use existing outputs modules such as:

- `pilates/activitysim/outputs.py`
- `pilates/beam/outputs.py`
- `pilates/urbansim/outputs.py`
- `pilates/atlas/outputs.py`

as the templates, not older docs or deprecated patterns.

### 3. Register components in `ModelFactory`

Add the model to `pilates/generic/model_factory.py`.

Typical registration:

```python
"freight": {
    "preprocessor": FreightPreprocessor,
    "runner": FreightRunner,
    "postprocessor": FreightPostprocessor,
}
```

Add separate entries for real variants with different runtime contracts, as the
code already does for `activitysim_compile` and `beam_full_skim`.

### 4. Add workflow catalog metadata

Add one `WorkflowStepSpec` per step in `pilates/workflows/catalog.py`.

For each step, define:

- `step_name`
- `model_name`
- `phase`
- `stage_name`
- `order`
- `outputs_class`
- `input_keys` and `optional_input_keys`
- `output_keys` and `optional_output_keys`
- `depends_on`
- `holder_inputs`
- enablement fields
- provenance builder metadata if the step should use an existing builder family

Treat the catalog as the static contract for the step. The runtime uses it for
validation, schema declaration, planning, and documentation.

### 5. Add a `StepOutputsHolder` field

Add the step output field to `pilates/workflows/steps/shared.py`:

```python
freight_preprocess: Optional[FreightPreprocessOutputs] = None
```

Keep the holder field name aligned with:

1. the `step_name` in the catalog
2. the typed outputs class
3. the step factory setter

`validate_workflow_step_contracts(...)` is supposed to fail early when these
drift.

### 6. Implement step factories

Add model-specific factories in `pilates/workflows/steps/<model>.py`.

The current examples to copy are:

- `pilates/workflows/steps/activitysim.py`
- `pilates/workflows/steps/beam.py`
- `pilates/workflows/steps/urbansim_atlas.py`
- `pilates/workflows/steps/postprocessing.py`

Your step module should usually do four things:

1. resolve the component through `ModelFactory`
2. execute the model component using the correct typed upstream outputs
3. validate and store the returned outputs on `StepOutputsHolder`
4. publish any cross-step artifacts to the coupler

Prefer the shared helpers in `pilates/workflows/steps/shared.py` over inventing
new execution shims.

### 7. Wire the owning stage

Update the appropriate stage module in `pilates/workflows/stages/`.

In the current runtime, stages:

- build bindings with `build_binding_plan(...)`
- create `StepRef`s
- execute them through `StageRunner` / `run_workflow(...)`
- consume typed outputs from `StepOutputsHolder`

Do not make a stage module into a hidden state layer that manually mutates the
coupler in ad hoc ways. Publish workflow-facing artifacts from the producing
step unless there is a clear stage-boundary durability reason not to.

### 8. Add artifact keys and schema entries only when needed

If downstream code needs a stable artifact from your step, add it in:

- `pilates/workflows/artifact_keys.py`
- `pilates/workflows/coupler_schema.py`

Good candidates:

- a latest datastore handoff
- warm-start skims
- a canonical plans/persons/households handoff
- a durable manifest or archive artifact used for restart

Bad candidates:

- temporary files used only within one component
- debug outputs that no downstream step reads
- duplicate names for the same semantic artifact

## How To Choose A Good Output Contract

Prefer these patterns:

- `output_dir: Path` plus a few explicit named outputs
- `raw_outputs: Dict[str, Path]` for dynamic file families
- metadata only when downstream logic actually depends on it

Avoid:

- returning `RecordStore` as the public step contract
- giant untyped dictionaries with mixed strings, paths, and tables
- encoding workflow control flow in the outputs object

The goal is not "maximum typing." The goal is a stable, inspectable boundary
that supports validation, caching, restart, and onboarding.

## Testing Checklist

At minimum, add tests for:

1. output dataclass validation and serialization behavior
2. catalog/holder alignment if you added tracked steps
3. step factory behavior for required upstream outputs
4. a happy-path integration test or stubbed workflow test
5. any restart or cache-recovery behavior unique to the new model

Useful starting points in the current suite:

- `tests/test_initialization.py`
- `tests/test_database_components.py`
- `tests/test_stub_provenance_flow.py`

Use the repo-preferred interpreter for local runs:

```bash
/Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest
```

## Common Mistakes

- Adding a model package but forgetting `ModelFactory` registration.
- Adding a `WorkflowStepSpec` but not adding the matching
  `StepOutputsHolder` field.
- Returning a raw `RecordStore` across the live workflow boundary.
- Publishing unstable or duplicate coupler keys.
- Depending on `cwd` instead of `Workspace` paths.
- Starting new Consist lifecycle contexts inside model code.
- Putting orchestration logic in the model component instead of the stage.
- Describing a contract in docs that is not enforced in code.

## Definition Of Done

An integration is in good shape when:

1. A new developer can find the model package, step module, catalog entry, and
   stage wiring without guesswork.
2. The public step boundary is typed and path-based.
3. The model's workflow-facing artifacts have stable names.
4. Startup validation catches contract drift.
5. The integration works in fresh runs and restart scenarios.
6. The docs were updated in the same change.
