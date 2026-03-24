# Artifact Flow Overview

This document summarizes the main workflow-facing artifact handoffs in the
current PILATES runtime.

It is intentionally high-level. For detailed step-by-step outputs, see
`docs/lineage_map.md`. For the runtime structure around these handoffs, see
`docs/workflow_primer.md`.

## What Counts As An "Artifact"

In PILATES, an artifact is a workflow-facing file or directory published under a
stable key so downstream steps, restart logic, or provenance tooling can refer
to it consistently.

Artifacts are defined and handled across four layers:

- artifact keys in `pilates/workflows/artifact_keys.py`
- coupler schema in `pilates/workflows/coupler_schema.py`
- typed step outputs in model `outputs.py` modules
- step publication logic in `pilates/workflows/steps/*.py`

The orchestration layer then consumes those artifacts through bindings and
`StepRef` execution in:

- `pilates/workflows/stages/*.py`
- `pilates/workflows/orchestration.py`
- `pilates/runtime/launcher.py`

## Core Handoffs By Workflow Phase

### Bootstrap -> Model Stages

Initialization and bootstrap seed the workspace and coupler with durable inputs
used by later stages.

Important examples:

- UrbanSim baseline datastore:
  - `usim_datastore_base_h5`
- current UrbanSim datastore handoff:
  - `usim_datastore_h5`
- shared skim inputs:
  - `omx_skims`
- ActivitySim/ATLAS bootstrap-safe copied inputs:
  - model- and region-specific files staged into the workspace

These are prepared during bootstrap/runtime initialization and then reused by
stage code and step bindings.

### UrbanSim -> ATLAS

The vehicle-ownership stage primarily consumes UrbanSim datastore outputs.

Main handoff:

- `usim_datastore_h5`

UrbanSim preprocess/run/postprocess also preserve related datastore roles such
as the baseline/current distinction needed for restart-sensitive workflows.

### UrbanSim -> ActivitySim

ActivitySim preprocess uses the current UrbanSim datastore to derive the core
ActivitySim input tables.

Main handoffs:

- `usim_datastore_h5`
- `asim_land_use_in`
- `asim_households_in`
- `asim_persons_in`
- `asim_omx_skims` when OMX skims are present

The important detail is that downstream workflow code does not need to guess at
table filenames. The handoff is expressed through typed outputs and artifact
keys.

### ActivitySim -> BEAM

ActivitySim postprocess and BEAM preprocess together define the activity-demand
to traffic-assignment boundary.

Important handoffs include:

- ActivitySim run/postprocess outputs such as household/person/plan tables
- BEAM prepared-input artifacts:
  - `beam_plans_in`
  - `beam_households_in`
  - `beam_persons_in`

BEAM preprocess may also publish extra prepared inputs when the scenario
requires them.

### ATLAS -> BEAM

When ATLAS is enabled, its processed vehicle outputs feed BEAM on the first
relevant iteration.

Main handoff:

- `atlas_vehicles2_input`

This is the main workflow-facing vehicle file boundary between the
vehicle-ownership stage and BEAM preprocessing.

### BEAM -> ActivitySim (Iterative Feedback)

During the supply-demand loop, BEAM postprocess provides skim artifacts that can
feed the next ActivitySim iteration.

Main handoffs:

- `zarr_skims`
- `final_skims_omx` when OMX export is enabled
- latest/published `linkstats` and warm-start artifacts where relevant

This is the main feedback channel that closes the ActivitySim <-> BEAM loop.

### BEAM -> Optional Full-Skim Processing

If full-skim generation is enabled, BEAM outputs can feed the optional
full-skim step.

Main handoff:

- `beam_full_skims`

That path is separate from the normal iterative skim handoff and is only used
when the workflow config enables it.

## Where To Inspect The Contract

If you need the authoritative definition of a handoff, inspect these in order:

1. `pilates/workflows/catalog.py`
   This tells you which step declares which workflow inputs and outputs.
2. model output dataclasses such as:
   - `pilates/activitysim/outputs.py`
   - `pilates/beam/outputs.py`
   - `pilates/urbansim/outputs.py`
   - `pilates/atlas/outputs.py`
3. the producing step module in `pilates/workflows/steps/*.py`
   This tells you what actually gets published to the coupler.
4. the consuming stage module in `pilates/workflows/stages/*.py`
   This tells you how the artifact is resolved and passed downstream.

## Important Notes

- Not every file produced by a model is a workflow artifact.
  Only stable cross-step or restart-relevant outputs should be treated that way.
- The coupler is the cross-step namespace, but typed outputs remain the primary
  in-memory contract inside a running workflow.
- Restart logic may reconstruct required artifacts from manifests, cache
  recovery, or durable workspace files. That is one reason stable artifact keys
  matter.
- The runtime is no longer centered on a monolithic `run.py`.
  Use `pilates/runtime/launcher.py`, `pilates/workflows/stages/*.py`, and
  `pilates/workflows/steps/*.py` as the current code references.

## Related Docs

- `docs/workflow_primer.md`
- `docs/model_integration_guide.md`
- `docs/lineage_map.md`
- `docs/artifact_facet_catalog.md`
