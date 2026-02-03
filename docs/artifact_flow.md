# Artifact Flow Overview

This document summarizes how key artifacts flow between steps in the current
PILATES workflow. It is intentionally high-level; for exhaustive mappings and
file lists, see `docs/lineage_map.md`.

## Core Artifact Handoffs

- UrbanSim → ATLAS: `usim_datastore_h5` (mutable HDF5 datastore)
- UrbanSim → ActivitySim: `usim_datastore_h5` (households/persons/land_use)
- ActivitySim → BEAM: `asim_output_dir` (activity plans)
- BEAM → ActivitySim (next iter): `zarr_skims` (skims cache)
- BEAM → ActivitySim (final): `final_skims_omx` (optional OMX export)
- ATLAS → BEAM (iter 0): `atlas_vehicles2_input` (vehicles2_<year>.csv)

## Where Artifacts Are Defined

- Step outputs: model-specific `outputs.py` files.
- Provenance logging: `pilates/workflows/steps.py` input/output loggers.
- Coupler keys: `pilates/workflows/coupler_schema.py`.
- Orchestration: `run.py` stage wiring.
- Full lineage map: `docs/lineage_map.md`.

## Notes

- Some artifacts are still referenced by literal keys (e.g., `usim_datastore_h5`).
  A future improvement is to centralize these keys into a shared constants module.
- For restart scenarios, initialization inputs may be re-derived from mutable
  directories (see ATLAS static input fallback in `run.py`).
