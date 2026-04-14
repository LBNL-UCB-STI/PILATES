# Adding a Model to PILATES

This guide is the practical checklist for adding a new model or a new model
variant to the current PILATES workflow.

Use it with:

- `docs/model_integration_guide.md` for architecture and design constraints
- `docs/workflow_primer.md` for the runtime and orchestration model

If this guide and the code disagree, the code wins. Update this file in the
same change that alters the integration surface.

## Fast Path

If you are adding a normal tracked workflow step family, the shortest current
path is:

1. add typed outputs in `pilates/<model>/outputs.py`
2. register the model components in `pilates/generic/model_factory.py`
3. add `WorkflowStepSpec` entries in `pilates/workflows/catalog.py`
4. add `StepOutputsHolder` fields in `pilates/workflows/steps/shared.py`
5. implement `make_*_step(...)` factories in `pilates/workflows/steps/<model>.py`
   using `build_standard_step(...)` and `StandardStepSpec`
6. export/register the step builders in `pilates/workflows/steps/__init__.py`
7. wire the owning stage in `pilates/workflows/stages/<stage>.py`
8. add tests for contract alignment, step behavior, and at least one realistic
   execution path

If your new step behaves like `activitysim_compile` or `postprocessing`, it is
not a standard tracked typed step. In that case, reuse the current custom
patterns instead of forcing it through `StandardStepSpec`.

## Contract Surfaces

For tracked steps, keep the contract layers in this order:

1. `WorkflowStepSpec` is the semantic source of truth
2. `StandardStepSpec` is the standard step-factory construction surface
3. model components own stable path knowledge through
   `expected_inputs(settings, state, workspace)` and
   `expected_outputs(settings, state, workspace)` when they can
4. restart/replay hooks are exceptional and should be added only when the
   default replay-first path is not enough
5. when a step needs provenance fingerprints, read Consist artifacts through
   `artifact.hash`; do not probe legacy hash field names
6. when a step logs selected HDF5 child artifacts, use
   `log_h5_container(..., child_specs=..., child_selection="include_only")`
   and keep only the workflow-specific table-selection logic local

If a step uses requested replay/restart staging, publish canonical destination
paths through `input_paths`. Do not make requested staging depend on local file
existence.

## If Restart Does Not Matter

There are two simpler paths when you do not care about custom restart behavior.

### Option A: Keep a normal tracked typed step, but skip custom restart hooks

This is the best default when downstream code still wants a typed in-memory
handoff through `StepOutputsHolder`.

Use the normal standard-step path:

- `WorkflowStepSpec` in the catalog
- `StepOutputsHolder` field
- `make_*_step(...)` using `StandardStepSpec` + `build_standard_step(...)`

But do not add:

- `output_recoverer`
- `output_replayer`

That still gives you:

- typed outputs
- normal coupler publication
- catalog/startup validation
- manifest restore for straightforward serializable outputs

You only need custom restart hooks when the step cannot be reconstructed
cleanly from its serialized outputs and default coupler republish behavior.

### Option B: Make it a lighter custom/untracked step

This is the lighter path when the step does not need typed downstream handoff
or special runtime reconstruction.

This is the current pattern used by steps such as `activitysim_compile`.

Typical shape:

- custom `make_*_step(...)` factory
- no tracked typed outputs
- no `StepOutputsHolder` field
- direct coupler publication for any downstream artifacts
- catalog entry marked as untracked when appropriate

Choose this path only when all of the following are true:

- downstream steps do not need a typed output object from this step
- the step can be safely rerun on resume
- the public contract is small and artifact-based rather than rich and typed

Do not use this path just to avoid writing output classes. If the step is a
real workflow boundary with non-trivial downstream behavior, keep it typed.

## What "adding a model" means in PILATES

In the current codebase, a model integration usually means all of the following:

1. A model package under `pilates/<model>/` with a preprocessor, runner,
   postprocessor, and typed output classes.
2. Factory registration in `pilates/generic/model_factory.py`.
3. Workflow step metadata in `pilates/workflows/catalog.py`.
4. A `StepOutputsHolder` field in `pilates/workflows/steps/shared.py`.
5. One or more step factories in `pilates/workflows/steps/<model>.py`.
6. Builder export/registration in `pilates/workflows/steps/__init__.py`.
7. Stage wiring in `pilates/workflows/stages/<stage>.py`.
8. Artifact-key and coupler-schema additions for any new cross-step handoffs.
9. Tests that cover the typed contract and at least one realistic execution path.

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
- use `optional_outputs` for conditional stable artifacts
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

Important current detail:

- `step_name` is the canonical workflow-step identity in the catalog.
- Older helper names may still talk about a "model name", but for catalog
  entries that is now just a compatibility alias for `step_name`.

Treat the catalog as the static contract for the step. The runtime uses it for
validation, schema declaration, planning, restart query derivation, and
documentation.

Keep the catalog aligned with the step metadata and provider surfaces. If
catalog semantics, `input_paths`/`output_paths`, and typed outputs disagree,
startup validation should fail.

### 5. Add a `StepOutputsHolder` field

Add the step output field to `pilates/workflows/steps/shared.py`:

```python
freight_preprocess: Optional[FreightPreprocessOutputs] = None
```

Keep the holder field name aligned with:

1. the `step_name` in the catalog
2. the typed outputs class
3. the step factory storage key

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

For most tracked typed steps, the current default pattern is:

1. write model-local callbacks for:
   - component lookup
   - execution
   - optional input logging
   - optional output logging
   - optional output replay/recovery
2. wrap those callbacks in `StandardStepSpec`
3. return `build_standard_step(coupler=..., outputs_holder=..., spec=...)`

When a model component already knows the canonical artifact layout, point
`input_paths` and `output_paths` at the model-owned `expected_*` providers
instead of re-deriving those paths inside the step module.

Use the current standard examples first:

- `make_activitysim_preprocess_step(...)`
- `make_activitysim_run_step(...)`
- `make_beam_run_step(...)`
- `make_urbansim_run_step(...)`
- `make_atlas_preprocess_step(...)`

Do not overgeneralize the callbacks into shared declarative policy. The builder
is intentionally narrow; model-specific logging, warm-start, replay, and
recovery logic should stay near the model code.

Use a custom step instead of `StandardStepSpec` when the step is not a normal
tracked typed workflow step. Current examples are:

- `activitysim_compile`
- `postprocessing`

If you need a custom recoverer or replayer, write down the semantic reason in
the step module. “Historical compatibility” by itself is not a good enough
reason unless you can name the workflow boundary being preserved.

### 7. Register the schema-step builder

If the step should participate in schema declaration and startup validation,
register its factory in `pilates/workflows/steps/__init__.py`.

Today this means:

1. export the `make_*_step(...)` symbol from the module import block
2. add the `step_name -> make_*_step` mapping to `SCHEMA_STEP_BUILDERS`

That registry is intentionally separate from the workflow catalog:

- the catalog is static contract metadata
- `SCHEMA_STEP_BUILDERS` is the executable builder surface

Keep both in sync in the same change.

### 8. Wire the owning stage

Update the appropriate stage module in `pilates/workflows/stages/`.

In the current runtime, stages:

- build bindings with `build_binding_plan(...)`
- create `StepRef`s
- execute them through `StageRunner` / `run_workflow(...)`
- consume typed outputs from `StepOutputsHolder`

Do not make a stage module into a hidden state layer that manually mutates the
coupler in ad hoc ways. Publish workflow-facing artifacts from the producing
step unless there is a clear stage-boundary durability reason not to.

### 9. Add artifact keys and schema entries only when needed

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
3. standard-step metadata and builder behavior if you used `StandardStepSpec`
4. step factory behavior for required upstream outputs
5. a happy-path integration test or stubbed workflow test
6. any restart or cache-recovery behavior unique to the new model

Useful starting points in the current suite:

- `tests/test_workflow_catalog_derivations.py`
- `tests/test_workflow_step_metadata.py`
- `tests/test_standard_step_builder.py`
- `tests/test_step_contract_validator.py`
- `tests/test_stage_contracts.py`
- `tests/test_golden_stub_workflow.py`

Use the repo-preferred interpreter for local runs:

```bash
/Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest
```

## Common Mistakes

- Adding a model package but forgetting `ModelFactory` registration.
- Adding a `WorkflowStepSpec` but not adding the matching
  `StepOutputsHolder` field.
- Treating catalog `step_name` and "model name" as separate identities for a
  new tracked step.
- Adding a tracked step factory but forgetting to register it in
  `SCHEMA_STEP_BUILDERS`.
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
