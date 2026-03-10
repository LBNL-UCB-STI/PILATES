# Adding a Model to PILATES

This guide is the practical checklist for adding a new model to the current
PILATES workflow.

Use it with:

- `docs/model_integration_guide.md` for architecture
- `docs/workflow_primer.md` for orchestration details

If this guide and the code disagree, the code wins.

## Short Version

For a new model, the safe path is:

1. scaffold the package and step module if helpful
2. define typed outputs in `pilates/<model>/outputs.py`
3. implement public `preprocess/run/postprocess` methods that return typed,
   path-based outputs
4. register the model in `pilates/generic/model_factory.py`
5. add tracked step metadata in `pilates/workflows/catalog.py`
6. add `StepOutputsHolder` fields in `pilates/workflows/steps/shared.py`
7. implement step factories in `pilates/workflows/steps/<module>.py`
8. wire stage orchestration in `pilates/workflows/stages/<stage>.py`
9. add coupler keys/schema only for true cross-step artifacts
10. add focused tests and run the scoped workflow gates

## Before You Start

Make these decisions first:

1. Which stage owns the model?
   Current stage names in the catalog are:
   - `land_use`
   - `vehicle_ownership_model`
   - `activity_demand`
   - `traffic_assignment`
   - `postprocessing`
2. Does the model need all three phases?
   The normal pattern is:
   - `<model>_preprocess`
   - `<model>_run`
   - `<model>_postprocess`
   Add a separate compile/prepare step only if it has a real distinct contract.
3. Which artifacts are real cross-step dependencies?
   Those are the only ones that should become coupler keys.
4. What is the typed output shape for each phase?
   Default answer: explicit `Path` fields plus small metadata fields.

## Scaffold

If the scaffold script saves time, start there:

```bash
python scripts/new_model_scaffold.py <model_slug> --major-stage <stage>
```

Useful flags:

- `--dry-run`
- `--force`
- `--class-prefix <Prefix>`
- `--step-module <module_name>`
- `--catalog-stage <stage_name>`
- `--stage-patch-plan`

What the scaffold gives you:

- model package boilerplate under `pilates/<model>/`
- step module boilerplate under `pilates/workflows/steps/`
- catalog and holder insertion points
- stage-template snippets/checklists

What still needs judgment:

- the final typed output shape
- coupler publication scope
- stage input resolution
- tests
- cleanup of any scaffolded bridge helpers you do not actually need

Treat the scaffold as a starting point, not architectural truth.

## Files You Will Usually Touch

1. `pilates/<model>/preprocessor.py`
2. `pilates/<model>/runner.py`
3. `pilates/<model>/postprocessor.py`
4. `pilates/<model>/outputs.py`
5. `pilates/generic/model_factory.py`
6. `pilates/workflows/steps/<module>.py`
7. `pilates/workflows/steps/__init__.py`
8. `pilates/workflows/steps/shared.py`
9. `pilates/workflows/catalog.py`
10. `pilates/workflows/stages/<stage>.py`
11. `pilates/workflows/artifact_keys.py`
12. `pilates/workflows/coupler_schema.py`
13. tests under `tests/`

If you are adding a brand-new top-level stage, also update:

1. `workflow_state.py`
2. `pilates/workflows/stages/__init__.py`
3. `run.py`

## Step 1: Define Typed Outputs

Create typed outputs in `pilates/<model>/outputs.py`.

Tracked steps should normally use `StepOutputsBase`.

Example:

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

- prefer explicit fields over giant untyped mappings
- keep outputs path-based
- add explicit metadata fields when correctness depends on them
- use stable artifact keys
- only keep `from_record_store(...)` / `to_record_store()` helpers if they are
  genuinely useful for tests or model-local internals

## Step 2: Implement Public Component Methods

New model components should follow the current public contract:

- `preprocess(...) -> <Model>PreprocessOutputs`
- `run(...) -> <Model>RunOutputs`
- `postprocess(...) -> <Model>PostprocessOutputs`

That public boundary should be typed and path-based.

Internal helpers may still use `RecordStore` if it truly simplifies local file
assembly, but do not expose it as the live cross-step contract.

Good pattern:

```python
class FreightRunner(GenericRunner[FreightPreprocessOutputs, FreightRunOutputs]):
    def run(
        self,
        inputs: FreightPreprocessOutputs,
        workspace: Workspace,
    ) -> FreightRunOutputs:
        return self._run(inputs, workspace)
```

Bad pattern:

```python
class FreightRunner(...):
    def run(...) -> RecordStore:
        ...
```

## Step 3: Register The Model

Add the model to `pilates/generic/model_factory.py`.

You need registrations for the phases you actually use. Typical entry:

```python
"freight": {
    "preprocessor": FreightPreprocessor,
    "runner": FreightRunner,
    "postprocessor": FreightPostprocessor,
}
```

If a compile/full-skim style variant exists, register that variant explicitly.

## Step 4: Add Catalog Metadata

Add `WorkflowStepSpec` entries in `pilates/workflows/catalog.py`.

For each tracked step, define:

- `step_name`
- `model_name`
- `phase`
- `stage_name`
- `order`
- `outputs_class`
- `depends_on`
- `holder_inputs`
- enablement attrs
- provenance builder family if applicable

Keep naming consistent with the actual step factory and holder field.

## Step 5: Update `StepOutputsHolder`

Add holder fields in `pilates/workflows/steps/shared.py`.

These are the in-memory typed handoff points between steps. They must stay
aligned with:

1. the catalog entry
2. the output class
3. the step factory setter

`validate_workflow_step_contracts(...)` is expected to catch drift here.

## Step 6: Implement Step Factories

Add model-specific factories in `pilates/workflows/steps/<module>.py`.

The active examples to copy are:

- `pilates/workflows/steps/activitysim.py`
- `pilates/workflows/steps/beam.py`
- `pilates/workflows/steps/urbansim_atlas.py`

The usual responsibilities are:

1. fetch the component from `ModelFactory`
2. call the public component method
3. enforce the typed return type
4. validate outputs with `ValidationContext`
5. store outputs on `StepOutputsHolder`
6. log and publish outputs
7. decorate the callable for Consist

Do not rebuild the old generic executor shim or design a new abstraction just
to avoid writing a small amount of local step code.

## Step 7: Wire The Stage

In the owning stage module:

1. resolve inputs with the input-resolution helpers
2. build `StepRef`s
3. pass explicit `inputs` and `input_keys` as needed
4. call `run_workflow(...)` or `run_manifested_steps(...)`
5. consume typed outputs from `StepOutputsHolder`

Do not hand-roll input precedence if `resolve_step_inputs(...)` or related
helpers already cover it.

Do not make the stage module a hidden coupler mutation layer.

## Step 8: Add Coupler Keys And Schema

Only add a coupler key if another step or stage genuinely consumes that
artifact.

For each real cross-step artifact:

1. add or reuse a constant in `pilates/workflows/artifact_keys.py`
2. document it in `pilates/workflows/coupler_schema.py`
3. publish it from the producing step via shared helpers
4. resolve it from the consuming step/stage

Do not publish everything just because it exists on disk.

## Step 9: Keep Recovery And Manifest Behavior Honest

Tracked outputs must work with:

- manifest serialization
- cache-hit recovery
- restart rehydration
- step-local replay logging

If your model requires special metadata for correctness, carry it on the typed
output object. Do not hide it in a workspace side cache.

## Step 10: Test It

At minimum, add:

1. typed public-method tests for preprocess/run/postprocess
2. step-factory wiring tests
3. coupler publication tests for real published artifacts
4. cache/recovery parity tests if the step participates in restart or cache-hit
   recovery

Useful existing patterns:

- `tests/test_activitysim_compile_run_handshake.py`
- `tests/test_beam_runner_outputs.py`
- `tests/test_urbansim_atlas_typed_contracts.py`
- `tests/test_step_shared_executor_helpers.py`
- `tests/test_manifest_cache_parity.py`
- `tests/test_cache_hit_recovery.py`
- `tests/test_stage_contracts.py`

## Recommended Validation Order

1. `ruff check` on the touched files
2. targeted `pytest` for the new model slice
3. broader workflow parity tests if the new step participates in recovery or
   stage-level fallback logic
4. `ty check --python /Users/zaneedell/miniforge3/envs/PILATES/bin/python`
   as a signal on the touched workflow scope

## Integration Checklist

Use this as the real definition of done:

1. typed outputs exist and validate correctly
2. public component methods return typed/path-based outputs
3. model is registered in `ModelFactory`
4. catalog metadata exists and is consistent
5. `StepOutputsHolder` fields exist
6. step factories are wired
7. stage `StepRef`s are wired
8. coupler keys/schema are added only where needed
9. recovery behavior is preserved
10. targeted tests pass
11. scoped `ruff` passes
12. any new `ty` issues in touched workflow code are fixed or explicitly tracked

## Common Mistakes

Avoid these:

- documenting a model around the old `steps.py` / `step_exec.py` layout
- returning `RecordStore` from the live public step path
- adding broad abstraction instead of a small local step helper
- publishing artifacts that no downstream code consumes
- using workspace side caches for correctness-critical metadata
- letting holder fields, catalog entries, and step factories drift apart
- leaving a compile/prepare exception undocumented

## References

- `docs/model_integration_guide.md`
- `docs/workflow_primer.md`
- `pilates/workflows/catalog.py`
- `pilates/workflows/steps/shared.py`
- `scripts/new_model_scaffold.py`
