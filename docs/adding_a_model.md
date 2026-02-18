# Adding a Model to PILATES

This guide is the canonical workflow for integrating a model into the current
PILATES architecture.

If this guide and code disagree, code wins. Update this file in the same PR as
integration changes.

## Quick Start

If you want the shortest safe path:

1. Generate scaffolding:

```bash
python scripts/new_model_scaffold.py <model_slug> --major-stage <stage>
```

2. Complete the generated checklist:

`docs/checklists/add_model_<model_slug>.md`

3. Wire the new step factories into an existing stage (or new stage), then add
   those factories to `run.py::_build_schema_steps()`.
4. Add/adjust coupler keys, schema descriptions, and Consist config builders.
5. Run focused tests (examples in [Testing and Validation](#testing-and-validation)).

You can safely preview scaffolding first:

```bash
python scripts/new_model_scaffold.py <model_slug> --major-stage <stage> --dry-run
```

Supported scaffold stage choices are:

- `land_use`
- `vehicle_ownership_model`
- `supply_demand_loop`
- `postprocessing`

## Mental Model

Model integration is straightforward when these three layers stay aligned:

1. Stage assembly (`pilates/workflows/stages/*.py`) builds ordered `StepRef`s.
2. Step factories (`pilates/workflows/steps/*.py`) run model phases and publish
   outputs.
3. Shared contracts (`StepOutputsHolder`, coupler keys/schema, Consist metadata)
   define what downstream steps can rely on.

Per simulation year, `run.py` currently executes:

1. `run_land_use_stage`
2. `run_vehicle_ownership_stage`
3. `run_supply_demand_stage`
4. `run_postprocessing_stage`

Consist lifecycle ownership is in `run.py` (`tracker.scenario(...)`,
`scenario.run(...)`, `scenario.trace(...)`). Model components must not start
nested runs.

## Hard Boundaries

PILATES intentionally separates ownership:

1. `StepOutputsHolder`: typed in-memory handoff between steps.
2. Coupler: published cross-step artifacts and keys.
3. Manifest state: restart durability for manifested workflows.

Coupler ownership contract:

1. Coupler read/write logic belongs in:
   `pilates/workflows/input_resolution.py`,
   `pilates/utils/coupler_helpers.py`,
   and limited orchestration diagnostics in
   `pilates/workflows/orchestration.py`.
2. Stage modules should assemble `StepRef`s, not call coupler mutation APIs.
3. Step modules should publish via shared helpers, not direct ad hoc coupler writes.

This is enforced by `tests/test_coupler_ownership_contracts.py`.

## Files You Will Touch

Most integrations touch all of these:

1. `pilates/<model>/`:
   `preprocessor.py`, `runner.py`, `postprocessor.py`, `outputs.py`
2. `pilates/generic/model_factory.py`
3. `pilates/workflows/steps/<module>.py` and `pilates/workflows/steps/__init__.py`
4. `pilates/workflows/steps/shared.py` (tracked outputs/dependencies)
5. `pilates/workflows/stages/<stage>.py`
6. `run.py` (`_build_schema_steps()` and, if needed, stage loop/filtering)
7. `pilates/workflows/artifact_keys.py`
8. `pilates/workflows/coupler_schema.py`
9. `pilates/utils/consist_config.py`
10. Tests under `tests/`
11. Docs, including this guide

If creating a new top-level stage, also update:

1. `workflow_state.py` (`WorkflowState.Stage`, enabled-stage logic, ordering)
2. `pilates/workflows/stages/__init__.py`
3. `run.py` yearly stage loop

## Step-By-Step Integration

### 1) Scaffold (recommended)

Command:

```bash
python scripts/new_model_scaffold.py freight --major-stage supply_demand_loop
```

Useful flags:

- `--dry-run`: preview writes only
- `--force`: overwrite scaffold targets
- `--class-prefix Freight`: override generated class names
- `--step-module freight_steps`: custom step module filename

The scaffold auto-creates:

1. model package boilerplate in `pilates/<model>/`
2. step factory module in `pilates/workflows/steps/<step_module>.py`
3. updates in:
   `pilates/generic/model_factory.py`,
   `pilates/workflows/steps/__init__.py`,
   `pilates/workflows/steps/shared.py`
4. checklist in `docs/checklists/add_model_<model>.md`

Still manual:

1. stage-level `StepRef` assembly
2. schema-step registration in `run.py::_build_schema_steps()`
3. coupler keys/schema and Consist config/hash/facet wiring
4. tests and docs

### 2) Implement model components

Use the standard component pattern:

- `GenericPreprocessor._preprocess(...) -> RecordStore`
- `GenericRunner._run(...) -> RecordStore`
- `GenericPostprocessor._postprocess(...) -> RecordStore`

Rules:

1. Return `RecordStore` artifacts; do not mutate orchestration state directly.
2. Use `Workspace` path getters; do not rely on `cwd`.
3. Keep run lifecycle/provenance orchestration outside components.
4. Add `expected_inputs(...)` / `expected_outputs(...)` where possible.
   Stage wiring merges these via `pilates/workflows/step_io.py`.

### 3) Define typed outputs (`outputs.py`)

Each step phase should have a `StepOutputsBase` dataclass. Prefer explicit
`declared_outputs` when output keys are stable.

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

Output contract precedence at runtime:

1. Step metadata outputs (`__consist_step__.outputs`)
2. `StepOutputs` declared outputs
3. `StepRef.required_outputs` override

When required outputs are present, orchestration defaults to strict policy:
`output_missing="error"` and `output_mismatch="error"` unless overridden.

### 4) Build step factories

Preferred path is `_make_generic_step_function(...)` in
`pilates/workflows/steps/shared.py`. It:

1. executes the model component
2. converts `RecordStore` to typed outputs
3. validates outputs
4. logs/publishes artifacts
5. decorates the callable with `define_step(...)` metadata

Naming convention is `<model>_<phase>` and should match holder/dependency maps.

### 5) Keep shared step registries consistent

For tracked steps, `pilates/workflows/steps/shared.py` must stay aligned:

1. `StepOutputsHolder` field
2. `STEP_OUTPUTS_CLASSES` entry
3. `STEP_DEPENDENCIES` entry

Startup validation (`validate_workflow_step_contracts`) checks consistency and
fails fast if any map drifts.

Note:

- `activitysim_compile` and `postprocessing` are currently allowlisted as
  untracked declared steps via `DEFAULT_UNTRACKED_STEP_NAMES`.
- If you add another declared-but-untracked step, either track it fully or
  update the allowlist intentionally.

### 6) Assemble stage `StepRef`s with canonical input resolution

Use `resolve_step_inputs(...)` and related helpers from
`pilates/workflows/input_resolution.py`. Per-key precedence is fixed:

1. explicit input
2. coupler
3. fallback input

Do not hand-roll precedence.

Example:

```python
resolved = resolve_step_inputs(
    keys=["freight_input"],
    coupler=coupler,
    explicit_inputs={"freight_input": explicit_path},
    fallback_inputs={"freight_input": fallback_path},
    required_keys=["freight_input"],
)
if resolved.missing_required:
    raise RuntimeError("freight_input is required")

steps = [
    StepRef(
        name="freight_run",
        step_func=make_freight_run_step(
            coupler=coupler,
            outputs_holder=outputs_holder,
        ),
        inputs=resolved.stepref_inputs(),
        input_keys=resolved.stepref_input_keys(),
        # Optional overrides:
        # required_outputs=[...],
        # output_missing="error|warn|allow",
        # output_mismatch="error|warn|allow",
    )
]
```

`StepRef` output contract field is `required_outputs`. There is no `StepRef.outputs`
field.

### 7) Register schema steps in `run.py`

Add your step factories to `run.py::_build_schema_steps()`.

Why this matters:

1. startup contract validation checks declared step metadata against tracked maps
2. coupler schema is built from enabled schema steps plus static extras
3. scenario `require_outputs` is derived from non-optional enabled steps

If your step model prefix is new (not `urbansim`, `atlas`, `activitysim`,
`beam`), update `_filter_schema_steps_for_enabled_models(...)` so it is included
when appropriate.

### 8) Add coupler keys and schema

When a new artifact is consumed across steps:

1. add/reuse a constant in `pilates/workflows/artifact_keys.py`
2. add schema description in `pilates/workflows/coupler_schema.py`
3. ensure producer logs/sets it via shared helpers
4. ensure consumer requests it via `input_keys` or resolved explicit `inputs`

Namespace behavior today:

1. publish helpers try `coupler.view("<namespace>")` when available
2. helpers also write legacy unscoped keys for compatibility
3. resolution prefers namespaced values and falls back to legacy keys

For key migrations/aliases, use `pilates/workflows/artifact_key_migrations.py`.

### 9) Configure Consist identity/facets/hash inputs

`pilates/utils/consist_config.py` maps step model names to per-model builders
for:

1. `config` (cache identity)
2. `facet` (queryable metadata)
3. `hash_inputs` (file/dir digests folded into identity)
4. `facet_schema_version`

If you add a new model family, add a builder and register it in
`_CONFIG_BUILDERS`.

### 10) Stage-specific and top-level wiring

For existing stage integration:

1. add your `StepRef`s to the relevant stage module
2. export/import step factories from `pilates/workflows/steps/__init__.py` if needed
3. ensure stage receives required inputs and forwards `outputs_holder`

For a new top-level stage:

1. add stage function in `pilates/workflows/stages/`
2. export it in `pilates/workflows/stages/__init__.py`
3. update `workflow_state.py` stage enum/enablement/order logic
4. call it from `run.py` in yearly loop
5. register schema steps and filter behavior

## Testing and Validation

Use the repo-preferred Python for local tests:

`/Users/zaneedell/miniforge3/envs/PILATES/bin/python`

Fast validation set:

```bash
/Users/zaneedell/miniforge3/envs/PILATES/bin/python -m pytest \
  tests/test_step_contract_validator.py \
  tests/test_stage_contracts.py \
  tests/test_input_resolution.py \
  tests/test_run_schema_filtering.py \
  tests/test_coupler_ownership_contracts.py -v
```

Add model-specific tests near related modules. Useful existing suites:

- `tests/test_expected_inputs_contracts.py`
- `tests/test_step_io.py`
- `tests/test_cache_hit_recovery.py`
- `tests/test_activitysim_config_adapter_wiring.py`
- `tests/test_beam_config_adapter_wiring.py`
- `tests/test_beam_artifact_facets.py`

If your model adds restart-sensitive manifested steps, include restart/hydration
coverage (see orchestration + supply-demand manifested flows).

## Common Failure Modes

1. Model exists but is not registered in `ModelFactory._registry`.
2. Step factory exists but is not wired into any stage.
3. Step is wired in a stage but missing from `_build_schema_steps()`.
4. `StepOutputsHolder` / `STEP_OUTPUTS_CLASSES` / `STEP_DEPENDENCIES` drift.
5. New cross-step key added to code but not `artifact_keys` + `coupler_schema`.
6. Ad hoc input precedence logic bypasses `input_resolution`.
7. Step declares outputs that typed outputs can never produce.
8. New model family misses `consist_config` builder registration.
9. New prefix step metadata is filtered out by `_filter_schema_steps_for_enabled_models`.

## PR Checklist

Before opening a PR:

1. Generated checklist for the model is complete.
2. Startup contract validation passes.
3. Coupler schema includes all new cross-step keys.
4. Required tests (contract + wiring + model-specific) pass.
5. `docs/adding_a_model.md` and any model-specific docs are updated.
6. If migration work changed, update `docs/consist_migration_checklist.md`.

## Related References

- `run.py`
- `workflow_state.py`
- `pilates/workflows/stages/*.py`
- `pilates/workflows/steps/*.py`
- `pilates/workflows/orchestration.py`
- `pilates/workflows/input_resolution.py`
- `pilates/workflows/artifact_keys.py`
- `pilates/workflows/coupler_schema.py`
- `pilates/utils/consist_config.py`
- `docs/consist_migration_checklist.md`
