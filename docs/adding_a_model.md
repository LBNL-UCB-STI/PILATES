# Adding a Model to PILATES

This guide documents how model integration works in the current codebase and is
intended to be the implementation playbook for new model wiring.

If this doc and code disagree, code wins. Update this file in the same PR as
workflow integration changes.

## 1) Current Workflow Structure

PILATES is orchestrated as stage functions that assemble ordered step calls,
executed under a Consist scenario context.

Primary entrypoints:

- `run.py` (workflow orchestration + scenario lifecycle)
- `pilates/workflows/stages/` (stage assembly)
- `pilates/workflows/steps/` (step factory implementations)
- `pilates/workflows/orchestration.py` (`StepRef`, `run_workflow`, manifest flow)

Current stage order per simulation year in `run.py`:

1. Land use (`run_land_use_stage`)
2. Vehicle ownership (`run_vehicle_ownership_stage`)
3. Supply/demand loop (`run_supply_demand_stage`)
4. Postprocessing (`run_postprocessing_stage`)

The scenario/coupler lifecycle is owned by `run.py` through Consist
(`tracker.scenario(...)`, `scenario.run(...)`, `scenario.step(...)`). Model code
must not create nested run contexts.

## 2) Where Model Integration Lives

A full integration usually spans all of the following:

1. `pilates/<model>/` (preprocessor, runner, postprocessor, outputs)
2. `pilates/generic/model_factory.py` (registry)
3. `pilates/workflows/steps/` (step factory wiring)
4. `pilates/workflows/stages/<stage>.py` (stage-level `StepRef` assembly)
5. `pilates/workflows/artifact_keys.py` and `pilates/workflows/coupler_schema.py`
6. `pilates/utils/consist_config.py` (step identity/facet/hash-input mapping)
7. tests under `tests/`
8. docs (this file + any model-specific docs)

If you are adding a new top-level stage, also update:

- `pilates/workflows/stages/__init__.py`
- `run.py` yearly execution loop

## 3) Step Module Layout (Current)

Step factories are split by domain:

- `pilates/workflows/steps/urbansim_atlas.py`
- `pilates/workflows/steps/activitysim.py`
- `pilates/workflows/steps/beam.py`
- `pilates/workflows/steps/postprocessing.py`
- shared step infrastructure in `pilates/workflows/steps/shared.py`

Public exports are in `pilates/workflows/steps/__init__.py`.

Important: legacy wrapper modules were removed. Add new wiring in the package
above, not in historical `steps_*` wrapper files.

## 4) Model Component Contract

Model components follow preprocessor/runner/postprocessor and return
`RecordStore` data.

Base classes:

- `pilates/generic/preprocessor.py` (`GenericPreprocessor`)
- `pilates/generic/runner.py` (`GenericRunner`)
- `pilates/generic/postprocessor.py` (`GenericPostprocessor`)

Requirements:

1. `_preprocess`, `_run`, `_postprocess` should emit `RecordStore` outputs.
2. Use `Workspace` helpers for paths (no `cwd` assumptions).
3. Keep orchestration/provenance lifecycle outside model internals.
4. Provide `expected_inputs(...)` and `expected_outputs(...)` where feasible.

Expected input/output declarations are merged via
`pilates/workflows/step_io.py`.

## 5) Step Outputs and Dependency Contracts

Typed step outputs are registered in `pilates/workflows/steps/shared.py`.

Core structures:

- `StepOutputsHolder`
- `STEP_OUTPUTS_CLASSES`
- `STEP_DEPENDENCIES`
- `validate_step_ready(...)`

When introducing step names, update all relevant registries or runtime recovery,
validation, and hydration will break.

## 6) Canonical Input Resolution (Use This)

PILATES now has one canonical input-resolution path in
`pilates/workflows/input_resolution.py`.

Use these helpers when assembling `StepRef.inputs` / `StepRef.input_keys`:

- `resolve_step_inputs(...)`
- `resolve_preferred_step_input(...)`
- `first_resolved_key(...)`
- `resolved_value_for_key(...)`

Canonical per-key precedence is:

1. explicit input
2. coupler key
3. fallback input

Do not hand-roll precedence with custom `if/elif` chains in stage or model
input wiring.

Why this matters:

- eliminates hidden behavior drift between stages
- keeps cache identity and run behavior predictable
- avoids coupler-vs-fallback surprises

## 7) `StepRef` Assembly Rules

`StepRef` (in `pilates/workflows/orchestration.py`) is the canonical execution
unit.

Common fields:

- `name`, `step_func`
- `inputs` (explicit mapping)
- `input_keys` (coupler-resolved keys)
- `output_paths` (expected outputs)
- cache controls: `cache_mode`, `cache_hydration`, `load_inputs`

Guidelines:

1. Prefer `input_resolution` helpers when producing `inputs/input_keys`.
2. Keep required input checks explicit (`required_keys` + clear errors).
3. Reuse `outputs_holder` in-memory outputs first; use coupler for cross-step
   handoff, not extra database passes.

## 8) Coupler Keys and Schema

Canonical keys live in `pilates/workflows/artifact_keys.py`.

Coupler schema is built in `pilates/workflows/coupler_schema.py` from:

1. static schema entries (`PILATES_COUPLER_SCHEMA`)
2. step-declared schema (`collect_step_schema`)
3. dynamic extras (for example deterministic ATLAS static input keys)

When adding cross-step artifacts:

1. add/reuse key constant
2. add schema description
3. ensure producing step logs/sets it
4. ensure consuming step requests it by `input_keys` or explicit `inputs`

For key migrations, use `pilates/workflows/artifact_key_migrations.py`.

## 9) Provenance, Facets, and Consist Config

Step-level Consist metadata is built through
`pilates/utils/consist_config.py` and applied by step decorators.

Per-model builders define:

- `config` (cache identity)
- `facet` (queryable metadata)
- `hash_inputs` (when needed)
- `facet_schema_version`

Artifact-level facet guidance:

1. use scalar facet fields for indexed querying
2. keep schema version explicit (for example `"v1"`)
3. keep keys human-readable but avoid encoding the full semantic contract in
   key names

For query-heavy families (for example BEAM linkstats variants), log facets on
all relevant artifacts and use structured param queries downstream.

## 10) Registering a New Model in `ModelFactory`

Add entries to `ModelFactory._registry` in
`pilates/generic/model_factory.py` for:

- `preprocessor`
- `runner`
- `postprocessor`

If you add a compile variant, register it explicitly (example:
`activitysim_compile`).

## 11) Minimal Integration Checklist

1. Implement model components in `pilates/<model>/`.
2. Register the model in `ModelFactory._registry`.
3. Add/update typed output dataclasses in `pilates/<model>/outputs.py`.
4. Update `StepOutputsHolder`, `STEP_OUTPUTS_CLASSES`, and dependencies.
5. Add `make_<model>_<phase>_step` factory functions under
   `pilates/workflows/steps/`.
6. Wire the step sequence in an existing or new stage module.
7. Use canonical `input_resolution` helpers for all stage input precedence.
8. Add artifact keys + coupler schema descriptions for new couplings.
9. Add/extend Consist config builder behavior as needed.
10. Add/extend artifact facets for queryable outputs.
11. Add tests for contracts, wiring, and restart/cache behavior.
12. Update docs.

## 12) Testing Expectations

At minimum, add or update coverage for:

1. expected input/output contracts (`step_io` merge behavior)
2. step dependency enforcement (`validate_step_ready`)
3. `RecordStore` to typed output conversion and validation
4. coupler output propagation for downstream consumers
5. facet metadata for query-critical artifacts
6. canonical input precedence behavior
7. manifest/restart behavior for manifested steps

Useful tests in this repo include:

- `tests/test_input_resolution.py`
- `tests/test_stage_contracts.py`
- `tests/test_expected_inputs_contracts.py`
- `tests/test_activitysim_config_adapter_wiring.py`
- `tests/test_beam_config_adapter_wiring.py`
- `tests/test_beam_artifact_facets.py`
- `tests/test_cache_hit_recovery.py`

## 13) Common Failure Modes

1. Model registered incompletely in `ModelFactory`.
2. Step factory exists but is never called from stage assembly.
3. Stage wired but not called from `run.py`.
4. Missing `StepOutputsHolder`/dependency map updates.
5. New coupler key not added to schema/constants.
6. Hand-rolled input precedence diverges from canonical resolver.
7. Overloading key names with semantics that should be facets.
8. Missing Consist hash inputs for config/files that should invalidate cache.

## 14) Practical Skeleton

```python
from pilates.workflows.input_resolution import resolve_step_inputs
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.steps import StepOutputsHolder, make_example_run_step


def run_example_stage(*, scenario, state, settings, workspace, coupler):
    outputs_holder = StepOutputsHolder()

    resolved = resolve_step_inputs(
        keys=["example_input"],
        coupler=coupler,
        explicit_inputs={"example_input": "/path/override"},
        fallback_inputs={"example_input": "/path/fallback"},
        required_keys=["example_input"],
    )
    if resolved.missing_required:
        raise RuntimeError("example_input is required")

    steps = [
        StepRef(
            name="example_run",
            step_func=make_example_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
            ),
            inputs=resolved.stepref_inputs(),
            input_keys=resolved.stepref_input_keys(),
        )
    ]

    run_workflow(
        stage_name="example",
        steps=steps,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=str(state.year),
        iteration=getattr(state, "iteration", 0),
    )
```

## 15) Related Files

- `run.py`
- `pilates/workflows/stages/*.py`
- `pilates/workflows/steps/*.py`
- `pilates/workflows/input_resolution.py`
- `pilates/workflows/orchestration.py`
- `pilates/workflows/coupler_schema.py`
- `pilates/workflows/artifact_keys.py`
- `pilates/utils/consist_config.py`
- `docs/consist_migration_checklist.md`
