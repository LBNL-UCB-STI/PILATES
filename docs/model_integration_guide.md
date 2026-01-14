# Model Integration Guide

This guide explains how to design, implement, and wire a new or existing model
into the PILATES workflow framework with Consist-first step granularity.

## Goals

- Make model execution explicit as preprocess/run/postprocess steps.
- Keep orchestration in `run.py`, not inside model components.
- Pass data with typed StepOutputs for safety and clear contracts.
- Use Consist coupler keys for provenance, caching, and hydration.

## Where Things Live

- Orchestration: `run.py`
- Step orchestration + dataclasses: `pilates/workflows/steps.py`
- Pure execution helpers: `pilates/workflows/step_exec.py`
- Inputs/outputs declaration: component `expected_inputs/expected_outputs`
- Coupler helpers: `pilates/utils/coupler_helpers.py`
- Coupler schema: `pilates/workflows/coupler_schema.py`
- Manifest persistence: `pilates/utils/step_manifest.py`

## Step Granularity Pattern

Each model is split into explicit Consist steps:

1. `<model>_preprocess`
2. `<model>_run`
3. `<model>_postprocess`

Optional compile/prepare steps should be separate if they have distinct inputs
and outputs (for example, `activitysim_compile`).

## Component Expectations

Every model should adhere to the preprocessor/runner/postprocessor pattern.
Each component returns a `RecordStore` and does not start Consist steps.

- Preprocessor: `preprocess(workspace) -> RecordStore`
- Runner: `run(input_store, workspace) -> RecordStore`
- Postprocessor: `postprocess(raw_outputs, workspace) -> RecordStore`

Declare expected inputs and outputs on each component:

```python
@staticmethod
def expected_inputs(settings, state, workspace) -> Dict[str, Any]:
    return {"some_input": workspace.get_some_input()}

@staticmethod
def expected_outputs(settings, state, workspace) -> Dict[str, Any]:
    return {"some_output": workspace.get_some_output()}
```

## StepOutputs Dataclasses

Define typed outputs per step in `pilates/workflows/steps.py`:

```python
@dataclass
class ExamplePreprocessOutputs:
    primary_output_attr: ClassVar[str] = "mutable_data_dir"
    mutable_data_dir: Path
    some_table: Path

    def validate(self) -> None:
        assert self.mutable_data_dir.exists()
        assert self.some_table.exists()
```

Guidelines:

- Use `Path` fields for outputs.
- Provide `.validate()` to fail early when outputs are missing.
- Optionally implement `from_record_store()` and `to_record_store()`.
- Keep these dataclasses internal to PILATES (not logged to Consist).

## StepOutputsHolder

Add fields to `StepOutputsHolder` so steps can share outputs safely:

```python
@dataclass
class StepOutputsHolder:
    example_preprocess: Optional[ExamplePreprocessOutputs] = None
    example_run: Optional[ExampleRunOutputs] = None
    example_postprocess: Optional[ExamplePostprocessOutputs] = None
```

## Step Factories

Create step functions in `pilates/workflows/steps.py` using the generic
factory pattern to avoid repeated boilerplate:

```python
def make_example_preprocess_step(coupler, outputs_holder):
    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="example",
        phase="preprocess",
        outputs_class=ExamplePreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "example", state, WorkflowState.Stage.some_stage
        ),
        component_executor=_execute_preprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "example_preprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )
```

Input and output logging should be handled via `log_and_set_input()` and
`log_and_set_output()` helper functions to keep Consist behavior centralized.

## Coupler Keys + Schema

When you add a new input or output that should be tracked by Consist:

1. Add a key to `PILATES_COUPLER_SCHEMA`.
2. Log inputs/outputs in step functions using coupler helpers.

Example:

```python
log_and_set_output(
    key="example_output_dir",
    path=str(outputs.output_dir),
    description="Example output directory",
    coupler=coupler,
)
```

## Wiring in run.py

The orchestration layer runs steps explicitly and manages manifests.
Follow this high-level flow:

1. Load `current_stage.yaml` (existing behavior).
2. Load `.workflow/year_{year}_iteration_{iteration}.yaml` manifest.
3. Reconstruct `StepOutputsHolder` from manifest.
4. For each step in order:
   - Skip if in manifest.
   - Validate dependencies.
   - Run via `scenario.run()`.
   - Persist outputs to manifest.

Example pattern:

```python
outputs_holder = StepOutputsHolder()
manifest = load_step_manifest(manifest_path) or {}

for step_key in steps:
    if step_key in manifest:
        continue
    validate_step_ready(step_key, outputs_holder)
    result = scenario.run(...)
    outputs = outputs_holder.get_attribute(step_key)
    manifest[step_key] = {
        "completed_at": datetime.now().isoformat(),
        "cache_hit": bool(result.cache_hit),
        "outputs": serialize_step_outputs(outputs),
    }
    save_step_manifest(manifest, manifest_path)
```

## Dependency Validation

Define dependencies in `STEP_DEPENDENCIES` and validate before execution:

```python
STEP_DEPENDENCIES = {
    "example_run": {
        "depends_on": ["example_preprocess"],
        "holder_inputs": ["example_preprocess"],
    },
}
```

This prevents out-of-order execution.

## Testing Expectations

Add or update tests in `tests/`:

- Unit tests for each step function.
- Integration test for preprocess->run->postprocess sequence.
- Cache behavior tests if applicable.

Keep tests using fixtures in `tests/fixtures/` to avoid large inputs.

## Checklist for Adding a Model

1. Create/verify preprocessor, runner, postprocessor components.
2. Define StepOutputs dataclasses and holder fields.
3. Implement step factories with input/output logging.
4. Add coupler schema keys for new artifacts.
5. Wire steps in `run.py` with manifest + dependency validation.
6. Add tests and validate provenance/logging.

## Common Pitfalls

- Starting Consist steps inside model code (only `run.py` should do this).
- Passing outputs via raw dicts instead of StepOutputs.
- Forgetting to log inputs to the coupler (breaks lineage/caching).
- Skipping manifest updates (breaks restart recovery).
- Relying on `cwd` for paths instead of `Workspace`.

## References

- `docs/step_granularity_architecture.md` for the full specification.
- `pilates/workflows/steps.py` for the ActivitySim implementation template.
- `pilates/utils/coupler_helpers.py` for Consist input/output logging helpers.
