---
title: Artifact Semantics
summary: Meaning of major workflow artifacts and why semantic names matter even when files overlap.
---

# Artifact Semantics

## Core Semantics

| Key or family | Current role |
| --- | --- |
| `USIM_DATASTORE_BASE_H5` | Static or base UrbanSim datastore handle for the run year. |
| `USIM_DATASTORE_CURRENT_H5` / `USIM_DATASTORE_H5` | Canonical current mutable UrbanSim datastore handle. `USIM_DATASTORE_H5` is an alias, not a separate concept. |
| `USIM_FORECAST_OUTPUT` | Forecast-year UrbanSim output from the runner before postprocess makes it available as the current mutable datastore handle. |
| `USIM_H5_UPDATED` | Current UrbanSim datastore republished after ATLAS postprocess. |
| `USIM_POPULATION_SOURCE_H5` | UrbanSim datastore selected for ActivitySim preprocess. |
| `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`) | Year-suffixed merged UrbanSim datastore snapshot family. |
| `USIM_INPUT_ARCHIVE_PREFIX` (`usim_input_archive_{year}`) | Year-suffixed archived UrbanSim datastore snapshot family. |
| `householdv_{year}`, `vehicles_{year}` | Year-scoped ATLAS output families published at the vehicle-ownership boundary. |
| `ATLAS_VEHICLES2_OUTPUT` | Postprocessed ATLAS vehicle output handed to downstream workflow consumers. |
| `OMX_SKIMS` / `omx_skims` | Default/bootstrap OMX skim source used by UrbanSim, ATLAS, and ActivitySim-compatible staging when no newer BEAM-produced OMX handoff is selected. |
| `ASIM_LAND_USE_IN`, `ASIM_HOUSEHOLDS_IN`, `ASIM_PERSONS_IN` | Staged ActivitySim table inputs derived from the selected population-source datastore. |
| `ASIM_OMX_SKIMS` | ActivitySim OMX skims input staged for the run boundary. |
| `ASIM_SHARROW_CACHE_DIR` | Optional persisted ActivitySim compile cache directory. |
| `ZARR_SKIMS` | Shared skims handoff used by ActivitySim and BEAM. |
| `BEAM_PLANS_IN`, `BEAM_HOUSEHOLDS_IN`, `BEAM_PERSONS_IN` | Staged BEAM inputs derived from the demand-side workflow state. |
| `LINKSTATS` | Canonical BEAM linkstats output publication. |
| `FINAL_SKIMS_OMX` | Final BEAM OMX skims output and the preferred updated OMX handoff for later UrbanSim/ATLAS/ActivitySim preprocessing when available. |
| `BEAM_FULL_SKIMS` | Separate full-skim tracked output family. |
| `LINKSTATS_WARMSTART` | Warm-start linkstats input for BEAM. |
| `BEAM_PLANS_OUT` | Published BEAM plans output from the run boundary. |
| `BEAM_OUTPUT_PLANS_XML`, `BEAM_EXPERIENCED_PLANS_XML`, `BEAM_OUTPUT_EXPERIENCED_PLANS_XML` | BEAM plan/XML publications that can also serve as restart sources. |
| `BEAM_INPUT_*_ARCHIVED` | Restart snapshot keys for BEAM inputs and warm-start artifacts. |

## Why PILATES Separates These Keys

- The same physical file can play different roles at different boundaries. The docs need the role, not just the path.
- Some keys exist to distinguish current mutable state from archived snapshots.
- Some keys exist because restart logic needs to ask for exactly one producer role, even when multiple files could look plausible.
- Some keys are aliases or compatibility handles that keep older code paths and names stable while the runtime converges on a canonical key.

## What To Be Careful About

- Do not infer scientific meaning from a key name unless the code or tests demonstrate it.
- Do not treat every registry entry as equally prominent in the current workflow. This page only describes the keys that the current runtime and tests exercise directly.
- If a key is only present as a compatibility alias or legacy restart artifact, say that explicitly.

## Adjacent Pages

- Read [Artifact Flow](artifact_flow.md) for the current handoff map.
- Pair this with [Lineage Map](lineage_map.md) and [Consist in PILATES](consist_in_pilates.md).
- For the step-by-step contract view, go to [Step Contracts](step_contracts.md).
