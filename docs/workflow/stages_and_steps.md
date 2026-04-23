---
title: Stages and Steps
summary: Execution model for years, stages, substages, and workflow steps in PILATES.
---

# Stages and Steps

## What PILATES Does

`WorkflowState` is the run-progress object. It stores:

- `current_year`, `current_major_stage`, `current_sub_stage`, and `current_inner_iter`
- `forecast_year` and `sub_stage_progress`
- restart state such as `run_info_path` and `is_restart_run`
- durable state-file values through `write_stage()` and `read_current_stage()`

The stage model in `workflow_state.py` separates the simulation into:

- `land_use`
- `vehicle_ownership_model`
- `supply_demand_loop`
- `postprocessing`

When activity demand is enabled, the supply-demand loop contains the substages `activity_demand` and `traffic_assignment`. When activity demand is disabled but land use is enabled, the runtime uses the direct-from-land-use activity-demand branch instead.

## Stage and Step Boundaries

- A stage owns orchestration across one region of the run lifecycle.
- A step owns one typed execution boundary and one typed output object.
- Steps run in year and iteration context, but the step factory still publishes the result through the same shared holder and coupler contract.

The launcher persists state after progress updates so restart runs can resume from the stored stage, year, iteration, and sub-stage fields.

## Practical Execution Shape

The major stage modules are called from the launcher year loop:

- `run_land_use_stage(...)`
- `run_vehicle_ownership_stage(...)`
- `run_supply_demand_stage(...)`
- `run_postprocessing_stage(...)`

Inside a stage, the normal pattern is:

1. receive a `WorkflowRuntimeContext` with settings, state, workspace, and enabled surface
2. resolve inputs with binding or model-local input helpers
3. create or call a step ref from `pilates/workflows/steps/`
4. let the step factory run the model component through `ModelFactory`
5. validate the returned typed outputs
6. publish outputs into `StepOutputsHolder` and the coupler
7. update `WorkflowState` at the stage or substage boundary

That separation is what keeps a stage readable. A stage should explain ordering
and progress. It should not hide a new public artifact contract, and it should
not duplicate fallback precedence that belongs in binding.

## Holder Versus Coupler

Two handoff surfaces are active during a run:

- `StepOutputsHolder` is in-memory and convenient for adjacent steps in the same Python process.
- The coupler is the workflow-visible artifact surface used across step boundaries, replay, and archive-aware recovery.

If a later step or restart path needs a value, it should be published through a
workflow key, not left as an implicit local path. Scratch files can stay inside
model-local directories only when no later workflow boundary depends on them.

## Practical Reading Rule

If you are reading the code for the first time:

- start in a stage module when you want to know *when* work runs
- start in a step module when you want to know *what boundary* a step publishes
- start in the catalog when you want to know *which keys and dependencies* the workflow declares
- start in a model adapter only after you know which step boundary called it

## Adjacent Pages

- Read [Workflow Primer](workflow_primer.md) first.
- Then read [Simulation Logic by Stage](simulation_logic_by_stage.md) for the logical meaning of each stage.
- Read [Step Contracts](step_contracts.md) for the semantic workflow boundary.

## Evidence Basis

- Stage state and persistence: `workflow_state.py`
- Runtime ordering and orchestration: `pilates/runtime/launcher.py`, `pilates/workflows/orchestration.py`
- Stage execution code: `pilates/workflows/stages/*.py`
- Stage and restart behavior: `tests/test_workflow_invariants.py`, `tests/test_golden_stub_workflow.py`
