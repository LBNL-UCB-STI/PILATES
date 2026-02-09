# Adding a Model to PILATES

This guide is the source of truth for how model integration works in the current
PILATES workflow. It covers:

1. How the workflow executes today.
2. The exact integration points for adding a model.
3. Consist/provenance/caching requirements you need to satisfy.

If this doc and code disagree, code wins. Update this file in the same PR as
workflow integration changes.

## 1) Current Workflow Architecture

PILATES orchestration is Consist-first and step-based.

- Entrypoint: `run.py`
- Stage runners:
  - `pilates/workflows/stages/land_use.py`
  - `pilates/workflows/stages/vehicle_ownership.py`
  - `pilates/workflows/stages/supply_demand.py`
  - `pilates/workflows/stages/postprocessing.py`
- Step definitions/factories: `pilates/workflows/steps.py`
- Stage execution engine: `pilates/workflows/orchestration.py`

High-level execution in `run.py`:

1. Parse settings and initialize `WorkflowState`.
2. Create a Consist tracker with mounted roots (`inputs`, `workspace`, `scratch`).
3. Create a Consist scenario context and declared coupler schema.
4. Run one-time initialization copy step (if not restarting).
5. Loop years and run stages in order:
   - Land use
   - Vehicle ownership
   - Supply/demand loop (iterative)
   - Postprocessing

Important runtime design points:

- Run lifecycle is owned by Consist `scenario(...)` and `scenario.run(...)`.
- Model code should not create nested run contexts.
- Coupler is the artifact handoff mechanism across steps.
- Restart/checkpoint behavior is driven by manifests for selected steps.

## 2) What "Adding a Model" Means in This Codebase

A model integration is not one change; it is a contract across multiple layers.

You usually need to touch:

1. `pilates/<model>/` package (preprocessor/runner/postprocessor/outputs)
2. `pilates/generic/model_factory.py`
3. `pilates/workflows/steps.py`
4. `pilates/workflows/stages/<stage>.py` (existing stage or new stage)
5. `run.py` (if new top-level stage)
6. `pilates/utils/consist_config.py`
7. `pilates/workflows/artifact_keys.py` and `pilates/workflows/coupler_schema.py`
8. Tests under `tests/`

## 3) Component Contract (Model Package)

Each model follows preprocessor/runner/postprocessor components and returns
`RecordStore` objects.

Base classes:

- `pilates/generic/preprocessor.py` -> `GenericPreprocessor`
- `pilates/generic/runner.py` -> `GenericRunner`
- `pilates/generic/postprocessor.py` -> `GenericPostprocessor`

Rules:

1. `_preprocess`, `_run`, and `_postprocess` return `RecordStore`.
2. Keep workflow/provenance orchestration out of model internals.
3. Use `Workspace` path helpers, never hardcoded `cwd` assumptions.
4. Expose static `expected_inputs(...)` / `expected_outputs(...)` where possible.

`expected_inputs/expected_outputs` are consumed by
`pilates/workflows/step_io.py` to merge model-declared contracts into stage
inputs and expected output declarations.

## 4) Register the Model in ModelFactory

File: `pilates/generic/model_factory.py`

Add the model to `_registry` with three components:

- `preprocessor`
- `runner`
- `postprocessor`

Without this, step factories cannot instantiate your model components.

## 5) Define Typed Step Outputs

PILATES converts `RecordStore` outputs into typed dataclasses per step.

Base: `pilates/workflows/outputs_base.py` -> `StepOutputsBase`

Per-model output types typically live in `pilates/<model>/outputs.py`.

Required patterns:

1. Use `Path` fields (and `Dict[str, Path]` for collections).
2. Implement `from_record_store(...)` (or rely on generic conversion).
3. Set `required_path_fields`, `optional_path_fields`, `dict_path_fields`.
4. Provide `_iter_record_items()` if key/path/description mapping is custom.

These output dataclasses are used by:

- dependency validation
- step-to-step handoff via in-memory holder
- manifest serialize/deserialize on restart

## 6) Wire Step Outputs into `steps.py`

File: `pilates/workflows/steps.py`

You must update all three maps/holders:

1. `StepOutputsHolder` fields
2. `STEP_OUTPUTS_CLASSES`
3. `STEP_DEPENDENCIES`

If one is missing, you will get runtime failures on dependency checks,
cache-hit recovery, or manifest hydration.

## 7) Implement Step Factories in `steps.py`

Step factories are the canonical integration surface.

Pattern:

- `make_<model>_preprocess_step(...)`
- `make_<model>_run_step(...)`
- `make_<model>_postprocess_step(...)`

Most should use `_make_generic_step_function(...)` with:

- `component_getter`: from `ModelFactory`
- `component_executor`: preprocess/run/postprocess executor
- `outputs_holder_setter`: stores typed outputs on `StepOutputsHolder`
- optional `input_logger` and `output_logger`

Logging and provenance expectations:

1. Log material inputs and outputs with coupler helpers in
   `pilates/utils/coupler_helpers.py`:
   - `log_and_set_input`
   - `log_input_only`
   - `log_and_set_output`
   - `log_output_only`
2. Include schema profiling metadata only where useful.
3. Include structured facets for query-heavy artifact families.

Step metadata is attached via `_decorate_step_with_consist(...)`, which wraps
`consist.define_step(...)` and adds dynamic metadata from
`pilates/workflows/step_consist_meta.py`.

## 8) Stage Wiring (Existing or New Stage)

Stages are declared as ordered `StepRef` lists and executed through
`run_workflow(...)` or `run_manifested_steps(...)`.

Engine file: `pilates/workflows/orchestration.py`

Each `StepRef` can carry:

- `inputs` (explicit key->artifact/path mapping)
- `input_keys` (read from coupler)
- `output_paths` (expected outputs)
- cache policy options (`cache_mode`, `cache_hydration`, `load_inputs`)
- explicit `model`, `year`, `iteration`, `phase`

For expensive iterative/restart-sensitive flows, use manifest checkpoints.
Current example: ActivitySim portions of supply/demand in
`pilates/workflows/stages/supply_demand.py`.

If adding a completely new top-level stage:

1. Create `pilates/workflows/stages/<your_stage>.py`
2. Export it in `pilates/workflows/stages/__init__.py`
3. Call it in `run.py` within the yearly loop and stage gating logic

## 9) Coupler Keys and Schema

Canonical keys live in `pilates/workflows/artifact_keys.py`.

Coupler schema is assembled in `pilates/workflows/coupler_schema.py` from:

1. static key descriptions (`PILATES_COUPLER_SCHEMA`)
2. step-declared schema via `collect_step_schema`
3. dynamic extras (for example, deterministic ATLAS static input keys)

When introducing new cross-step artifacts:

1. Add or reuse a constant in `artifact_keys.py`.
2. Add description in `coupler_schema.py`.
3. Ensure producing step logs and sets the key.
4. Ensure consuming step references it via `input_keys` or explicit `inputs`.

If migrating key formats, check
`pilates/workflows/artifact_key_migrations.py`.

## 10) Consist Config/Facet/Hash Integration

File: `pilates/utils/consist_config.py`

Every step model should have deterministic Consist metadata:

- `config` (cache identity)
- `facet` (queryable metadata)
- `hash_inputs` when file content should affect identity
- `facet_schema_version`

How to add a new model family:

1. Implement a `ConsistConfigBuilder`.
2. Register it in `_CONFIG_BUILDERS`.
3. Decide whether `workspace_path` is required for hash inputs.
4. Return stable `facet_schema_version` strings.

`pilates/workflows/step_consist_meta.py` resolves these dynamically at runtime
from `StepContext`, so step functions stay clean.

## 11) Artifact Facets: Recommended Pattern

For artifacts you plan to query by dimensions (year, iteration, scenario,
sub-iteration, etc.), log structured scalar facets.

You can set facets through:

1. `FileRecord.metadata` fields (`facet`, `facet_schema_version`, `facet_index`)
2. step logging meta passed to `log_output_only` / `log_and_set_output`
3. batch RecordStore logging path in `pilates/generic/model.py` (`_log_record_store`)

Current examples in workflow:

- BEAM linkstats and phys-sim sub-iteration linkstats facets
- ActivitySim output facets by year/iteration/family
- UrbanSim and ATLAS output/input family facets

Guidelines:

1. Use scalar fields only (indexing/query expects scalar values).
2. Keep schema versions explicit (for example, `"v1"`).
3. Keep artifact keys human-readable, but do not rely on key parsing as the
   long-term semantic contract.

## 12) Minimal Integration Checklist

Use this in PRs.

1. Add model component classes under `pilates/<model>/`.
2. Register model in `ModelFactory._registry`.
3. Add typed outputs dataclasses and validation.
4. Update `StepOutputsHolder`, `STEP_OUTPUTS_CLASSES`, `STEP_DEPENDENCIES`.
5. Add `make_<model>_*_step` factories with proper input/output logging.
6. Wire into an existing stage or add a new stage module.
7. If new stage: export in `stages/__init__.py` and call from `run.py`.
8. Add/update artifact keys + coupler schema descriptions.
9. Add Consist config builder and `_CONFIG_BUILDERS` entry.
10. Add facets for query-relevant artifacts.
11. Add tests (unit + integration + restart/caching expectations).
12. Update docs (this file + other relevant docs).

## 13) Testing Expectations

At minimum, cover:

1. `expected_inputs/expected_outputs` contract for your model components.
2. Step dependency enforcement (`validate_step_ready`).
3. `RecordStore` -> typed outputs conversion and `.validate()` checks.
4. Coupler key updates for outputs needed downstream.
5. Facet metadata presence for query-critical artifacts.
6. Restart behavior if manifest-backed steps are involved.

Useful nearby tests:

- `tests/test_expected_inputs_contracts.py`
- `tests/test_coupler_helpers.py`
- `tests/test_beam_artifact_facets.py`
- `tests/test_consist_migration_phase3_4.py`

## 14) Common Failure Modes

1. Model added to package but not registered in `ModelFactory`.
2. Step factories created but not referenced in any stage.
3. Stage created but never invoked from `run.py`.
4. Missing holder/dependency map entries in `steps.py`.
5. New coupler keys not added to schema/constants.
6. Over-encoding semantics in key names instead of facets.
7. Using filesystem paths directly instead of coupler artifacts for step inputs.
8. Missing `hash_inputs` for config-driven steps where mutable config files
   should invalidate cache.

## 15) Practical Template (Copy/Paste Starting Point)

```python
# steps.py

def make_example_preprocess_step(*, coupler, outputs_holder):
    def _log_outputs(outputs, settings, state, workspace, holder):
        for key, path, desc in outputs._iter_record_items():
            log_and_set_output(
                key=key,
                path=str(path),
                description=desc,
                coupler=coupler,
            )

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
        output_logger=_log_outputs,
    )
```

```python
# stages/example.py

def run_example_stage(...):
    outputs_holder = StepOutputsHolder()
    steps = [
        StepRef(
            name="example_preprocess",
            step_func=make_example_preprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            inputs=example_inputs,
        ),
        StepRef(
            name="example_run",
            step_func=make_example_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            input_keys=["example_input_key"],
        ),
        StepRef(
            name="example_postprocess",
            step_func=make_example_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
        ),
    ]
    run_workflow(..., steps=steps, outputs_holder=outputs_holder)
```

## 16) Notes Specific to Current PILATES Workflow

1. `run.py` declares a coupler output schema with `warn_undefined=True`.
   Undeclared outputs are a warning signal; treat them as schema drift.
2. Supply/demand iteration recovery relies on manifest files under `.workflow/`.
3. BEAM warmstart should use linkstats artifacts; do not substitute unrelated
   event-derived link tables.
4. ATLAS static inputs are scenario/year filtered and key-sanitized; preserve
   deterministic key generation when extending ATLAS input catalogs.

## 17) Related Docs

- `docs/model_integration_guide.md`
- `docs/artifact_facet_catalog.md`
- `docs/consist_migration_checklist.md`
- `docs/step_granularity_architecture.md`
