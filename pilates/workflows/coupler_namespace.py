from __future__ import annotations

from typing import Optional

from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_MUTABLE_DATA_DIR,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_INPUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    BEAM_OUTPUT_DIR,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    USIM_MUTABLE_DATA_DIR,
)

NAMESPACE_SEPARATOR = "/"


_EXPLICIT_NAMESPACE_BY_KEY = {
    # UrbanSim
    USIM_DATASTORE_BASE_H5: "urbansim",
    USIM_DATASTORE_CURRENT_H5: "urbansim",
    USIM_DATASTORE_H5: "urbansim",
    USIM_H5_UPDATED: "urbansim",
    USIM_MUTABLE_DATA_DIR: "urbansim",
    USIM_FORECAST_OUTPUT: "urbansim",
    # ActivitySim
    ASIM_MUTABLE_DATA_DIR: "activitysim",
    ASIM_LAND_USE_IN: "activitysim",
    ASIM_HOUSEHOLDS_IN: "activitysim",
    ASIM_PERSONS_IN: "activitysim",
    ASIM_OMX_SKIMS: "activitysim",
    # BEAM
    BEAM_OUTPUT_DIR: "beam",
    BEAM_MUTABLE_DATA_DIR: "beam",
    BEAM_PLANS_IN: "beam",
    BEAM_HOUSEHOLDS_IN: "beam",
    BEAM_PERSONS_IN: "beam",
    BEAM_PLANS_OUT: "beam",
    LINKSTATS: "beam",
    LINKSTATS_WARMSTART: "beam",
    # ATLAS
    ATLAS_OUTPUT_DIR: "atlas",
    ATLAS_VEHICLES2_INPUT: "atlas",
}

_NAMESPACE_PREFIX_RULES = (
    ("usim_", "urbansim"),
    ("atlas_", "atlas"),
    ("beam_", "beam"),
    ("linkstats", "beam"),
    ("events_", "beam"),
    ("raw_od_skims", "beam"),
    ("path_traversal_links_", "beam"),
    ("plans_beam_", "beam"),
    ("households_beam_", "beam"),
    ("persons_beam_", "beam"),
)


def is_namespaced_key(key: str) -> bool:
    return NAMESPACE_SEPARATOR in key


def infer_namespace_for_key(key: str) -> Optional[str]:
    """
    Infer a model namespace for an unscoped coupler key.
    """
    if not key or is_namespaced_key(key):
        return None
    explicit = _EXPLICIT_NAMESPACE_BY_KEY.get(key)
    if explicit:
        return explicit
    for prefix, namespace in _NAMESPACE_PREFIX_RULES:
        if key.startswith(prefix):
            return namespace
    if "_asim_" in key:
        return "activitysim"
    return None


def qualify_key(namespace: str, key: str) -> str:
    """
    Build a normalized namespaced key.
    """
    normalized_namespace = namespace.strip(NAMESPACE_SEPARATOR)
    if not normalized_namespace:
        return key
    local_key = key.strip(NAMESPACE_SEPARATOR)
    if local_key.startswith(f"{normalized_namespace}{NAMESPACE_SEPARATOR}"):
        return local_key
    return f"{normalized_namespace}{NAMESPACE_SEPARATOR}{local_key}"


def local_key_for_namespace(key: str, namespace: str) -> str:
    """
    Return the local key portion for a namespace view.
    """
    normalized_namespace = namespace.strip(NAMESPACE_SEPARATOR)
    if not normalized_namespace:
        return key
    prefix = f"{normalized_namespace}{NAMESPACE_SEPARATOR}"
    local_key = key.strip(NAMESPACE_SEPARATOR)
    if local_key.startswith(prefix):
        return local_key[len(prefix) :]
    return local_key
