# PILATES Architecture

This document is the high-level architecture overview for PILATES.

It is intended to answer two questions quickly:

1. What are the main runtime layers in the current codebase?
2. Which docs should you read next for a specific task?

## System Overview

PILATES is a workflow runtime that coordinates multiple model adapters across a
shared yearly simulation lifecycle.

The main model families in this repository are:

- UrbanSim
- ATLAS
- ActivitySim
- BEAM
- postprocessing utilities

The system combines:

- configuration-driven execution
- a run-local workspace
- typed workflow contracts
- artifact publication through a coupler namespace
- Consist-backed provenance, caching, and restart support

## Main Runtime Layers

### 1. Entry Point And Runtime Assembly

- `run.py`
- `pilates/runtime/launcher.py`

`run.py` is intentionally thin. The real runtime assembly happens in
`pilates/runtime/launcher.py`, which handles:

- CLI/config loading
- `WorkflowState` construction
- `Workspace` creation
- Consist scenario and runtime contract setup
- bootstrap execution
- yearly stage sequencing
- restart logging and failure guidance

### 2. Workflow State And Workspace

- `workflow_state.py`
- `pilates/workspace.py`

`WorkflowState` tracks the current year, stage, iteration, and restart state.

`Workspace` owns run-local filesystem layout and provides canonical paths for:

- UrbanSim
- ActivitySim
- BEAM
- ATLAS

This is the main boundary that keeps model code from depending on ambient
working-directory assumptions.

### 3. Stage Orchestration

- `pilates/workflows/stages/*.py`
- `pilates/workflows/orchestration.py`

Stages own control flow.

They decide:

- which steps run
- in what order
- with which bindings
- under which manifest/restart behavior

Examples:

- `land_use.py`
- `vehicle_ownership.py`
- `supply_demand.py`
- `postprocessing.py`

### 4. Step Execution

- `pilates/workflows/steps/*.py`
- `pilates/workflows/steps/shared.py`

Steps are workflow-aware execution units created by step factories.

The step layer is responsible for:

- resolving model components from `ModelFactory`
- consuming typed upstream outputs
- executing preprocess/run/postprocess methods
- validating outputs
- publishing workflow artifacts
- storing typed outputs on `StepOutputsHolder`

### 5. Model Adapters

- `pilates/<model>/`
- `pilates/generic/`

Each model family generally follows the same pattern:

- `preprocessor.py`
- `runner.py`
- `postprocessor.py`
- `outputs.py`

The generic bases in `pilates/generic/` define the common component interface.

### 6. Contract Layer

- `pilates/workflows/catalog.py`
- `pilates/workflows/outputs_base.py`
- `pilates/workflows/artifact_keys.py`
- `pilates/workflows/coupler_schema.py`

This layer defines the stable workflow contract:

- tracked steps
- expected inputs and outputs
- typed output classes
- artifact-key conventions
- step dependencies

This is the main guardrail against integration drift.

## Core Design Rules

The current architecture is built around a few practical rules:

- runtime owns lifecycle
- stages own orchestration
- steps own execution and publication
- model packages own model-local logic
- typed outputs define the public step boundary

This is why the current code is more explicit than older versions of PILATES.
There are more layers, but each one has a clearer job.

## Provenance, Caching, And Restart

Consist is integrated into the runtime and step layers to support:

- tracked step execution
- cache identity
- artifact logging
- restart-aware recovery

Important runtime pieces include:

- `pilates/runtime/bootstrap.py`
- `pilates/runtime/scenario_runtime.py`
- `pilates/utils/consist_config.py`
- `pilates/utils/consist_analysis.py`

The important architectural consequence is that workflow boundaries must be
stable enough to survive cache hits and restart recovery. That is why typed
outputs and artifact keys matter so much in this repo.

## Main Artifact Boundaries

At a high level, the workflow moves data across these major boundaries:

- bootstrap -> staged workspace inputs
- UrbanSim -> ATLAS
- UrbanSim -> ActivitySim
- ActivitySim -> BEAM
- ATLAS -> BEAM
- BEAM -> ActivitySim iterative skim feedback

For a concise version, see `docs/artifact_flow.md`.
For the fuller map, see `docs/lineage_map.md`.

## Where To Start Depending On Your Task

### If you want to run PILATES

Read:

1. `docs/getting_started.md`
2. `docs/cli_reference.md`
3. `docs/configuration_reference.md`
4. `docs/data_bootstrap.md`

### If you want to understand the workflow runtime

Read:

1. this file
2. `docs/workflow_primer.md`
3. `docs/provenance_and_lineage.md`
4. `docs/artifact_flow.md`

### If you want to add or modify a model integration

Read:

1. `docs/workflow_primer.md`
2. `docs/model_integration_guide.md`
3. `docs/adding_a_model.md`

### If you want to inspect run outputs and the database

Read:

1. `docs/database-setup.md`
2. `docs/database_documentation_guide.md`
3. `docs/database_schema_reference.md`

## Related Docs

### Runtime And Workflow

- `docs/workflow_primer.md`
- `docs/model_integration_guide.md`
- `docs/adding_a_model.md`
- `docs/artifact_flow.md`
- `docs/lineage_map.md`

### User-Oriented Operation

- `docs/getting_started.md`
- `docs/cli_reference.md`
- `docs/configuration_reference.md`
- `docs/data_bootstrap.md`
- `docs/troubleshooting.md`
- `docs/hpc_execution.md`

### Provenance And Database

- `docs/provenance_and_lineage.md`
- `docs/artifact_facet_catalog.md`
- `docs/database-setup.md`
- `docs/database_documentation_guide.md`
- `docs/database_schema_reference.md`

## Maintenance Rule

Keep public docs in `docs/` focused on current behavior.

Historical notes, proposals, and exploratory design material belong in
`docs-internal/` unless they are still active and directly useful to most
contributors.
