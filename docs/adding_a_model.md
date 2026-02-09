# Adding a Model to PILATES

This guide explains how model integration works in the current codebase and how
to add a new model without breaking workflow contracts.

If this doc and code disagree, code wins. Update this file in the same PR as
workflow integration changes.

## 0) Start Here: Mental Model

Before touching files, keep this flow in mind:

1. A **stage** assembles one or more **steps**.
2. Each step runs a model **phase** (`preprocess`, `run`, `postprocess`).
3. Step outputs are logged, written to the **coupler**, and stored in typed
   in-memory objects.
4. Downstream steps resolve inputs in a standard order and continue.

If you keep stage assembly + step contracts + coupler keys aligned, integration
is straightforward.

## 0.05) Ownership Rules (Hard Boundaries)

PILATES uses three distinct ownership layers:

1. `StepOutputsHolder` owns typed in-process handoff between steps.
2. Coupler owns published cross-step artifact references and provenance-facing
   keys.
3. Manifest/checkpoint state owns restart durability.

Code boundaries:

1. Direct coupler reads/writes belong in gateway modules:
   `pilates/workflows/input_resolution.py`,
   `pilates/utils/coupler_helpers.py`,
   with limited diagnostics in `pilates/workflows/orchestration.py`.
2. Stage modules should assemble `StepRef`s and call input-resolution helpers,
   not call coupler methods directly.
3. Step modules should use shared logging/publish helpers, not direct coupler
   mutation.

These boundaries are enforced by
`tests/test_coupler_ownership_contracts.py`.

## 0.1) Jargon Quick Reference

- **Stage**: A major block in the yearly simulation loop (for example land use,
  supply/demand).
- **Step**: A single callable unit inside a stage (usually one model phase).
- **StepRef**: The execution spec for a step (`name`, callable, `inputs`,
  `input_keys`, cache config).
- **Coupler**: Shared cross-step key/value artifact map managed through Consist.
- **RecordStore**: Collection of file artifacts emitted by model components.
- **StepOutputsHolder**: In-memory typed outputs used for intra-workflow handoff
  without re-reading from disk/database.
- **Manifest**: Serialized checkpoint of step progress/outputs for restart.
- **Facet**: Structured metadata on artifacts used for query/filter.
- **Cache identity**: Hash/materialized metadata used to decide cache hits.

## 0.2) Scaffold First (Recommended)

Use the scaffold generator before hand-editing integration files:

```bash
python scripts/new_model_scaffold.py <model_name> --major-stage <stage>
```

Example:

```bash
python scripts/new_model_scaffold.py freight --major-stage supply_demand_loop
```

Useful flags:

- `--dry-run`: preview all file changes without writing
- `--force`: overwrite generated scaffold files if rerunning
- `--class-prefix Freight`: override generated class names
- `--step-module freight_steps`: place step factories in a custom step module

What the scaffold writes automatically:

1. model package boilerplate in `pilates/<model>/`
2. step stub module in `pilates/workflows/steps/<step_module>.py`
3. registry updates in:
   `pilates/generic/model_factory.py`,
   `pilates/workflows/steps/__init__.py`,
   `pilates/workflows/steps/shared.py`
4. a model-specific TODO list in
   `docs/checklists/add_model_<model>.md`

What still requires manual wiring:

1. stage-level `StepRef` assembly in `pilates/workflows/stages/*.py`
2. schema-step declaration updates in `run.py::_build_schema_steps()`
3. artifact keys, coupler schema, and Consist facet/hash policy updates
4. model-specific tests and docs

## 1) Current Workflow Structure

PILATES is orchestrated as stage functions that assemble ordered steps under a
Consist scenario context.

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

A complete integration usually touches all of these layers:

1. `pilates/<model>/` (preprocessor, runner, postprocessor, outputs)
2. `pilates/generic/model_factory.py` (model registry)
3. `pilates/workflows/steps/` (step factory wiring)
4. `pilates/workflows/stages/<stage>.py` (stage-level step assembly)
5. `pilates/workflows/artifact_keys.py` and `pilates/workflows/coupler_schema.py`
6. `pilates/utils/consist_config.py` (step identity/facets/hash inputs)
7. tests under `tests/`
8. docs (this file plus model-specific docs)

If you are adding a new top-level stage, also update:

- `pilates/workflows/stages/__init__.py`
- `run.py` yearly execution loop

## 3) Step Module Layout (Current)

Step factories are split by domain:

- `pilates/workflows/steps/urbansim_atlas.py`
- `pilates/workflows/steps/activitysim.py`
- `pilates/workflows/steps/beam.py`
- `pilates/workflows/steps/postprocessing.py`
- shared infrastructure in `pilates/workflows/steps/shared.py`

Public exports are in `pilates/workflows/steps/__init__.py`.

Important: legacy wrapper modules were removed. Add new wiring in the step
package above, not in historical `steps_*` wrappers.

## 4) Model Component Contract

Model components follow the preprocessor/runner/postprocessor pattern and
exchange artifacts via `RecordStore`.

Base classes:

- `pilates/generic/preprocessor.py` (`GenericPreprocessor`)
- `pilates/generic/runner.py` (`GenericRunner`)
- `pilates/generic/postprocessor.py` (`GenericPostprocessor`)

Requirements:

1. `_preprocess`, `_run`, `_postprocess` should emit `RecordStore` outputs.
2. Use `Workspace` helpers for all paths (no `cwd` assumptions).
3. Keep orchestration/provenance lifecycle outside model internals.
4. Provide `expected_inputs(...)` and `expected_outputs(...)` where feasible.

Expected input/output declarations are merged via
`pilates/workflows/step_io.py`.

## 5) Step Outputs and Dependency Contracts

Typed step outputs and dependency contracts are centralized in
`pilates/workflows/steps/shared.py`.

Core structures:

- `StepOutputsHolder`
- `STEP_OUTPUTS_CLASSES`
- `STEP_DEPENDENCIES`
- `validate_step_ready(...)`
- `validate_workflow_step_contracts(...)`
- `StepOutputsBase.declared_outputs` (canonical strict-output contract source)

Why this matters: restart/hydration logic assumes these registries are
consistent. If you add a step name in one place but not the others, workflow
validation or restoration will fail.

Declared output contract rule:

1. Prefer setting `declared_outputs` directly on each `StepOutputs` dataclass
   when output keys are stable.
2. If `declared_outputs` is omitted, shared step wiring falls back to
   `required_path_fields` + `record_keys`.
3. Orchestration uses step metadata first (`@define_step(outputs=[...])`), then
   `StepOutputs` declared outputs, then `StepRef` overrides.

## 6) Canonical Input Resolution (Use This)

PILATES uses one canonical input-resolution path in
`pilates/workflows/input_resolution.py`.

Use these helpers when assembling `StepRef.inputs` / `StepRef.input_keys`:

- `resolve_step_inputs(...)`
- `resolve_preferred_step_input(...)`
- `first_resolved_key(...)`
- `resolved_value_for_key(...)`

Per-key precedence is always:

1. explicit input
2. coupler key
3. fallback input

Do not hand-roll custom precedence with ad hoc `if/elif` chains.

Why this matters:

- prevents behavior drift between stages
- keeps cache identity and execution behavior predictable
- avoids coupler-vs-fallback surprises during restart

## 7) `StepRef` Assembly Rules

`StepRef` (in `pilates/workflows/orchestration.py`) is the canonical execution
unit passed to `run_workflow(...)`.

Common fields:

- `name`, `step_func`
- `inputs` (explicit key -> value mapping)
- `input_keys` (keys expected to already exist in coupler)
- `output_paths` (declared expected outputs)
- optional output-contract overrides:
  `outputs` (or `StepRef.required_outputs` alias), `output_missing`,
  `output_mismatch`
- cache controls: `cache_mode`, `cache_hydration`, `load_inputs`

Guidelines:

1. Use `input_resolution` helpers when producing `inputs/input_keys`.
2. Keep required input checks explicit (`required_keys` + clear errors).
3. Prefer `outputs_holder` in-memory outputs first; use coupler for cross-step
   handoff rather than extra DB passes unless there is a clear benefit.
4. Preferred output-contract pattern:
   declare step outputs in step metadata (`@define_step(outputs=[...])` or the
   step factory metadata path), and let orchestration infer strict defaults
   (`output_missing="error"`, `output_mismatch="error"`).
5. Use `StepRef.required_outputs` (alias for explicit `outputs`) /
   `StepRef.output_*` only when overriding the default inferred behavior for a
   specific step.
6. For generic step factories, define canonical outputs once on the
   `StepOutputs` class (`declared_outputs`) so decoration and runtime fallback
   share the same contract source.

## 8) Coupler Keys and Schema

Canonical artifact key constants live in
`pilates/workflows/artifact_keys.py`.

Coupler schema is built in `pilates/workflows/coupler_schema.py` from:

1. static schema entries (`PILATES_COUPLER_SCHEMA`)
2. step-declared schema (`collect_step_schema`)
3. dynamic extras (for example deterministic ATLAS static input keys)

When adding a cross-step artifact:

1. add/reuse a key constant
2. add a schema description
3. ensure the producer step logs/sets it
4. ensure the consumer step requests it via `input_keys` or explicit `inputs`

Namespace behavior:

1. Coupler gateway helpers now use `coupler.view("<model>")` when available.
2. During migration, helpers also keep writing legacy unscoped keys for
   compatibility.
3. Input resolution prefers namespaced keys when present and automatically
   falls back to unscoped keys.

For key migrations, use `pilates/workflows/artifact_key_migrations.py`.

## 9) Provenance, Facets, and Consist Config

Step-level Consist metadata is defined through
`pilates/utils/consist_config.py` and applied by step decorators.

Per-model builders define:

- `config` (cache identity)
- `facet` (queryable metadata)
- `hash_inputs` (files/config that should invalidate cache)
- `facet_schema_version`

Artifact-level facet guidance:

1. use scalar facet fields for indexed querying
2. keep schema version explicit (for example `"v1"`)
3. keep artifact keys human-readable, but avoid encoding all semantics in key
   strings

For query-heavy families (for example BEAM linkstats variants), log facets on
all relevant artifacts and query by params instead of parsing key names.

## 10) Registering a New Model in `ModelFactory`

Add entries to `ModelFactory._registry` in
`pilates/generic/model_factory.py` for:

- `preprocessor`
- `runner`
- `postprocessor`

If you add a compile variant, register it explicitly (example:
`activitysim_compile`).

## 11) Practical Integration Checklist

1. Run `python scripts/new_model_scaffold.py <model_name> --major-stage <stage>`.
2. Implement model behavior in generated components under `pilates/<model>/`.
3. Refine typed output dataclasses in `pilates/<model>/outputs.py`.
   Add `declared_outputs` for stable, queryable contract keys.
4. Refine `make_<model>_<phase>_step` factories under
   `pilates/workflows/steps/`.
5. Wire step sequence into an existing or new stage module.
6. Add new step callables to `run.py::_build_schema_steps()` so startup
   contract validation includes them.
7. Add or reuse artifact keys and coupler schema descriptions.
8. Add/extend Consist config hashing and facet policy.
9. Prefer output contracts declared on step metadata and rely on inferred strict
   defaults; use `StepRef.output_*` / explicit `outputs` only for explicit
   overrides.
10. Add tests for contracts, wiring, and restart/cache behavior.
11. Update docs.

## 12) Testing Expectations

At minimum, add or update coverage for:

1. expected input/output contracts (`step_io` merge behavior)
2. step dependency enforcement (`validate_step_ready`)
3. `RecordStore` to typed output conversion and validation
4. coupler output propagation for downstream consumers
5. facet metadata for query-critical artifacts
6. canonical input precedence behavior
7. manifest/restart behavior for manifested steps

Useful tests in this repo:

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
3. Stage is wired but never invoked from `run.py`.
4. `StepOutputsHolder` / dependency maps were not updated consistently.
5. New coupler key not added to schema/constants.
6. Input precedence implemented ad hoc instead of via canonical resolver.
7. Key names overloaded with semantics that should live in facets.
8. Missing hash inputs for config/files that should invalidate cache.

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
