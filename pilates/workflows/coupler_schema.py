"""
Workflow coupler schema for PILATES.

This schema documents coupler keys that are explicitly set during the workflow.
"""

PILATES_COUPLER_SCHEMA = {
    "usim_datastore_h5": "UrbanSim datastore (H5) produced by land use/ATLAS.",
    "atlas_output_dir": "ATLAS output directory for the current sub-year.",
    "asim_output_dir": "ActivitySim output directory for the current iteration.",
    "beam_output_dir": "BEAM output directory for the current iteration.",
    "zarr_skims": "Zarr skims produced by ActivitySim/BEAM.",
    "final_skims_omx": "Final OMX skims produced by BEAM.",
    "linkstats": "BEAM linkstats output for downstream runs.",
    "beam_plans_out": "BEAM plans output for downstream runs.",
}
