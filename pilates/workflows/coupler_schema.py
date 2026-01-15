"""
Workflow coupler schema for PILATES.

This schema documents coupler keys that are explicitly set during the workflow.
"""

from typing import Dict

from pilates.workflows.artifact_constants import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ASIM_MUTABLE_DATA_DIR,
    ASIM_OUTPUT_DIR,
    ATLAS_OUTPUT_DIR,
    BEAM_HOUSEHOLDS_CSV_GZ,
    BEAM_HOUSEHOLDS_IN,
    BEAM_HOUSEHOLDS_PARQUET,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_OUTPUT_DIR,
    BEAM_PLANS_OUT,
    BEAM_PLANS_CSV_GZ,
    BEAM_PLANS_IN,
    BEAM_PLANS_PARQUET,
    BEAM_PERSONS_CSV_GZ,
    BEAM_PERSONS_IN,
    BEAM_PERSONS_PARQUET,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_H5,
    USIM_MUTABLE_DATA_DIR,
    ZARR_SKIMS,
)


PILATES_COUPLER_SCHEMA: Dict[str, str] = {
    USIM_DATASTORE_H5: "UrbanSim datastore (H5) produced by land use/ATLAS.",
    USIM_MUTABLE_DATA_DIR: "UrbanSim mutable data directory in workspace.",
    ASIM_MUTABLE_DATA_DIR: "ActivitySim mutable data directory from preprocess.",
    ATLAS_OUTPUT_DIR: "ATLAS output directory for the current sub-year.",
    ASIM_OUTPUT_DIR: "ActivitySim output directory for the current iteration.",
    BEAM_OUTPUT_DIR: "BEAM output directory for the current iteration.",
    BEAM_MUTABLE_DATA_DIR: "BEAM mutable data directory populated for the run.",
    ASIM_LAND_USE_IN: "ActivitySim land use input table (from preprocess).",
    ASIM_HOUSEHOLDS_IN: "ActivitySim households input table (from preprocess).",
    ASIM_PERSONS_IN: "ActivitySim persons input table (from preprocess).",
    ASIM_OMX_SKIMS: "ActivitySim compile input skims (OMX).",
    ZARR_SKIMS: "Zarr skims produced by ActivitySim/BEAM.",
    FINAL_SKIMS_OMX: "Final OMX skims produced by BEAM.",
    BEAM_PLANS_IN: "BEAM plans input staged for the runner.",
    BEAM_PLANS_PARQUET: "BEAM plans input (parquet) staged for the runner.",
    BEAM_PLANS_CSV_GZ: "BEAM plans input (csv.gz) staged for the runner.",
    BEAM_HOUSEHOLDS_IN: "BEAM households input staged for the runner.",
    BEAM_HOUSEHOLDS_PARQUET: "BEAM households input (parquet) staged for the runner.",
    BEAM_HOUSEHOLDS_CSV_GZ: "BEAM households input (csv.gz) staged for the runner.",
    BEAM_PERSONS_IN: "BEAM persons input staged for the runner.",
    BEAM_PERSONS_PARQUET: "BEAM persons input (parquet) staged for the runner.",
    BEAM_PERSONS_CSV_GZ: "BEAM persons input (csv.gz) staged for the runner.",
    LINKSTATS_WARMSTART: "BEAM warm-start linkstats input (initial or prior run).",
    LINKSTATS: "BEAM linkstats output for downstream runs.",
    BEAM_PLANS_OUT: "BEAM plans output for downstream runs.",
}
