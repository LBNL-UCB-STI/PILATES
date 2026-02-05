"""
Workflow coupler schema for PILATES.

This schema documents coupler keys that are explicitly set during the workflow.
"""

import os
from typing import Dict

from pilates.atlas.static_inputs import (
    ATLAS_STATIC_INPUTS_BY_SCENARIO,
    ATLAS_STATIC_INPUTS_COMMON,
)
from pilates.generic.records import sanitize_artifact_key
from pilates.utils.path_utils import find_project_root

from pilates.workflows.artifact_constants import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ASIM_MUTABLE_DATA_DIR,
    ATLAS_OUTPUT_DIR,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_HOUSEHOLDS_IN,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_PLANS_OUT,
    BEAM_PLANS_IN,
    BEAM_PERSONS_IN,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_H5,
    USIM_H5_UPDATED,
    USIM_MUTABLE_DATA_DIR,
    ZARR_SKIMS,
)


PILATES_COUPLER_SCHEMA: Dict[str, str] = {
    USIM_DATASTORE_H5: "UrbanSim datastore (H5) produced by land use/ATLAS.",
    USIM_H5_UPDATED: "UrbanSim datastore updated by ATLAS postprocess.",
    USIM_MUTABLE_DATA_DIR: "UrbanSim mutable data directory in workspace.",
    ASIM_MUTABLE_DATA_DIR: "ActivitySim mutable data directory from preprocess.",
    ATLAS_OUTPUT_DIR: "ATLAS output directory for the current sub-year.",
    BEAM_MUTABLE_DATA_DIR: "BEAM mutable data directory populated for the run.",
    ASIM_LAND_USE_IN: "ActivitySim land use input table (from preprocess).",
    ASIM_HOUSEHOLDS_IN: "ActivitySim households input table (from preprocess).",
    ASIM_PERSONS_IN: "ActivitySim persons input table (from preprocess).",
    ASIM_OMX_SKIMS: "ActivitySim compile input skims (OMX).",
    ZARR_SKIMS: "Zarr skims produced by ActivitySim/BEAM.",
    FINAL_SKIMS_OMX: "Final OMX skims produced by BEAM.",
    BEAM_PLANS_IN: "BEAM plans input staged for the runner.",
    BEAM_HOUSEHOLDS_IN: "BEAM households input staged for the runner.",
    BEAM_PERSONS_IN: "BEAM persons input staged for the runner.",
    LINKSTATS_WARMSTART: "BEAM warm-start linkstats input (initial or prior run).",
    LINKSTATS: "BEAM linkstats output for downstream runs.",
    BEAM_PLANS_OUT: "BEAM plans output for downstream runs.",
    BEAM_OUTPUT_PLANS_XML: "BEAM output plans XML (previous run warm-start source).",
    BEAM_EXPERIENCED_PLANS_XML: "BEAM experienced plans XML (previous run warm-start source).",
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML: (
        "BEAM output experienced plans XML (previous run warm-start source)."
    ),
    "hh_size": "UrbanSim household size input CSV.",
    "income_rates": "UrbanSim income rates input CSV.",
    "relmap": "UrbanSim relationship map input CSV.",
    "schools": "UrbanSim schools input CSV.",
    "school_districts": "UrbanSim school districts input CSV.",
    "canonical_zones": "Canonical zones file copied into ActivitySim workspace.",
    "clipped_geoms": "Clipped geometry inputs for ActivitySim if available.",
}

_NAMESPACED_INIT_KEYS = {
    "urbansim": {
        "usim_data_reference": "UrbanSim reference datastore input.",
        "usim_datastore_h5": "UrbanSim datastore in mutable workspace.",
        "omx_skims": "UrbanSim OMX skims input.",
        "hh_size": "UrbanSim household size input CSV.",
        "income_rates": "UrbanSim income rates input CSV.",
        "relmap": "UrbanSim relationship map input CSV.",
        "schools": "UrbanSim schools input CSV.",
        "school_districts": "UrbanSim school districts input CSV.",
    },
    "activitysim": {
        "omx_skims": "ActivitySim OMX skims input.",
    },
}

for model_name, key_map in _NAMESPACED_INIT_KEYS.items():
    for key, description in key_map.items():
        namespaced_key = f"{model_name}/{key}"
        if namespaced_key not in PILATES_COUPLER_SCHEMA:
            PILATES_COUPLER_SCHEMA[namespaced_key] = description


def _atlas_static_key_map() -> Dict[str, str]:
    keys: Dict[str, str] = {}
    all_paths = []
    project_root = find_project_root(start_path=os.path.dirname(__file__))
    if project_root:
        atlas_input_root = os.path.join(project_root, "pilates", "atlas", "atlas_input")
        if os.path.isdir(atlas_input_root):
            for root, _, files in os.walk(atlas_input_root):
                for filename in files:
                    if "readme" in filename.lower():
                        continue
                    full_path = os.path.join(root, filename)
                    relpath = os.path.relpath(full_path, atlas_input_root)
                    all_paths.append(relpath.replace("\\", "/"))

    if not all_paths:
        all_paths = list(ATLAS_STATIC_INPUTS_COMMON)
        for scenario_paths in ATLAS_STATIC_INPUTS_BY_SCENARIO.values():
            all_paths.extend(scenario_paths)

    for relpath in all_paths:
        rel_no_ext, _ = os.path.splitext(relpath)
        key = rel_no_ext.replace("\\", "/")
        key = sanitize_artifact_key(key) or key
        if key not in keys:
            keys[key] = f"ATLAS static input file: {relpath}"
    return keys


PILATES_COUPLER_SCHEMA.update(_atlas_static_key_map())
