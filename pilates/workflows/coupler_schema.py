"""
Workflow coupler schema for PILATES.

This schema documents coupler keys that are explicitly set during the workflow.
"""

import os
from typing import Any, Callable, Dict, Iterable, Optional

from consist.utils import collect_step_schema

from pilates.atlas.inputs import atlas_static_input_relpaths
from pilates.generic.records import sanitize_artifact_key
from pilates.workflows.coupler_namespace import namespaced_alias_for_key

from pilates.workflows.artifact_keys import (
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
    BEAM_FULL_SKIMS,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_H5_UPDATED,
    USIM_MUTABLE_DATA_DIR,
    ZARR_SKIMS,
)


PILATES_COUPLER_SCHEMA: Dict[str, str] = {
    USIM_DATASTORE_BASE_H5: (
        "UrbanSim base datastore (H5): static/exogenous input for the run year."
    ),
    USIM_DATASTORE_CURRENT_H5: (
        "UrbanSim current datastore (H5): latest mutable version produced by workflow steps."
    ),
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
    BEAM_FULL_SKIMS: "BEAM full-skim background skims output.",
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
        "usim_datastore_base_h5": "UrbanSim base datastore in mutable workspace.",
        "usim_datastore_current_h5": "UrbanSim current datastore in mutable workspace.",
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


def _atlas_static_key_map(settings: Optional[Any]) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for relpath in atlas_static_input_relpaths(settings):
        rel_no_ext, _ = os.path.splitext(relpath)
        key = rel_no_ext.replace("\\", "/")
        key = sanitize_artifact_key(key) or key
        if key not in keys:
            keys[key] = f"ATLAS static input file: {relpath}"
    return keys


def build_coupler_schema(
    steps: Iterable[Callable[..., Any]],
    settings: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Build the workflow coupler schema from step metadata plus static extras.

    Parameters
    ----------
    steps : iterable of callables
        Canonical workflow step functions decorated with ``@define_step``.
    settings : Any, optional
        Settings object used to resolve callable schema metadata.

    Returns
    -------
    dict
        Coupler key -> description mapping.
    """
    extras = dict(PILATES_COUPLER_SCHEMA)
    for key, description in list(PILATES_COUPLER_SCHEMA.items()):
        namespaced_key = namespaced_alias_for_key(key)
        if not namespaced_key:
            continue
        extras.setdefault(
            namespaced_key,
            f"{description} (namespaced alias)",
        )
    extras.update(_atlas_static_key_map(settings))
    try:
        return collect_step_schema(steps, settings=settings, extra_keys=extras)
    except Exception:
        return extras
