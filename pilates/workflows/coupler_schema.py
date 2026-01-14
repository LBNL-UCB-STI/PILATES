"""
Workflow coupler schema for PILATES.

This schema documents coupler keys that are explicitly set during the workflow.
"""

from typing import Dict

from pilates.workflows.artifact_constants import (
    ASIM_MUTABLE_DATA_DIR,
    ASIM_OUTPUT_DIR,
    ATLAS_OUTPUT_DIR,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_OUTPUT_DIR,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
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
    ZARR_SKIMS: "Zarr skims produced by ActivitySim/BEAM.",
    FINAL_SKIMS_OMX: "Final OMX skims produced by BEAM.",
    LINKSTATS: "BEAM linkstats output for downstream runs.",
    BEAM_PLANS_OUT: "BEAM plans output for downstream runs.",
}
