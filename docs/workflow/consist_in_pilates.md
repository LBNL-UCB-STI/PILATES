---
title: Consist in PILATES
summary: What Consist owns in PILATES and how replay, caching, staging, and analysis fit together.
---

# Consist in PILATES

## What Consist Owns

- Scenario and step contexts created in `run.py` / `pilates/runtime/launcher.py`.
- Cache identity and queryable facets built by `pilates/utils/consist_config.py`.
- Step metadata, parent run linkage, tags, and artifact records.
- Step-level cache hits, replay results, and archive-friendly run outputs.

## What PILATES Owns

- The run/storage topology: archive run directory versus mutable local workspace.
- Bootstrap hydration, restart preflight, and stage orchestration.
- Coupler seeding for bootstrap artifacts and replay-restored paths.
- The choice to treat some recovery helpers as legacy/manual tools rather than the normal launcher path.

## Current Runtime Shape

`pilates/runtime/launcher.py` initializes runtime flags, restores `WorkflowState`, builds the enabled workflow surface, declares the scenario contract, seeds bootstrap artifacts into the coupler, and then runs the year loop inside Consist scenario/step contexts.

`pilates/utils/consist_config.py` keeps the Consist inputs narrow and explicit:

- `config` carries the cache identity.
- `facet` carries queryable run metadata.
- `identity_inputs` are only added for steps that need file or directory digests.

If you want one file that demonstrates the intended Consist integration
pattern, start there. `pilates/utils/consist_config.py` is the reference for:

- which config fields become cache identity
- which fields become queryable run facets
- which boundaries need `identity_inputs` instead of broad config hashing

For runtime-local code that needs to report the active run, use
`pilates.current_run_id()` or `pilates.utils.consist_runtime.current_run_id()`
instead of probing `consist.current_run()` directly.

The launcher also records run context metadata such as `scenario_id`, `seed`, `archive_run_dir`, `local_run_dir`, and `restart_run`.

The key ownership boundary is:

- the enabled workflow surface decides which stages, step contracts, and restart requirements are active
- Consist executes those declared boundaries, records lineage, and provides replay / cache behavior for them

## Public Artifact Surface

Consist records the tracked workflow publications, but the semantic public surface still comes from PILATES workflow keys and families. The most important current examples are:

- `USIM_DATASTORE_CURRENT_H5` / `USIM_DATASTORE_H5` for the canonical current mutable UrbanSim datastore handle
- `USIM_POPULATION_SOURCE_H5` for the datastore selected for ActivitySim preprocess
- `USIM_INPUT_ARCHIVE_PREFIX` (`usim_input_archive_{year}`) and `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`) for year-scoped UrbanSim snapshot families
- `ZARR_SKIMS` for the shared skims handoff used by ActivitySim and BEAM
- `FINAL_SKIMS_OMX` and `BEAM_FULL_SKIMS` for BEAM-side skim outputs
- `BEAM_INPUT_*_ARCHIVED` for replay-relevant archived BEAM inputs

Use [Artifact Semantics](artifact_semantics.md), [Artifact Flow](artifact_flow.md), and [Lineage Map](lineage_map.md) for the contract-level meaning of those publications.

## Replay, Restart, and Hydration

- Fresh runs execute bootstrap first, then enter the scenario context.
- Restart runs can still run bootstrap pre-scenario so the workspace invariants are rehydrated through the normal cached path.
- When `state.data_initialized` is already true on restart, the launcher skips bespoke restart hydration and relies on scenario replay plus Consist cache hits.
- The launcher emits a `restart_hydration` audit event with `fallback_reason="replay_mode"` in that replay-first case.
- The older hydration helpers in `pilates/runtime/restart.py` remain available for manual tooling and focused tests, but the normal launcher path does not call them.

## Analysis Surface

PILATES keeps the archive-side Consist database and run outputs available for post-run analysis. The analysis helpers read the archived run metadata, run outputs, and artifact facets rather than rebuilding workflow state from scratch.

For a concrete walkthrough, use [Consist in Action](../analysis/consist_in_action.md).

## Adjacent Pages

- Read [Scenario Lifecycle](../run/scenario_lifecycle.md) next.
- Continue to [Artifact Semantics](artifact_semantics.md) and [Lineage Map](lineage_map.md).
- For the analysis-facing view of archived runs, go to [Opening Archives](../analysis/opening_archives.md).
