from __future__ import annotations

from typing import Any, Optional, Tuple

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


def namespaced_alias_for_key(key: str) -> Optional[str]:
    """
    Return the namespaced alias for an unscoped key when one can be inferred.
    """
    namespace = infer_namespace_for_key(key)
    if not namespace:
        return None
    alias = qualify_key(namespace, key)
    if alias == key:
        return None
    return alias


def resolve_coupler_value(coupler: Any, key: str) -> Tuple[Any, Optional[str]]:
    """
    Resolve a key from a coupler using canonical namespace-aware lookup order.

    Lookup order:
    1. namespaced view (`coupler.view(namespace).get(local_key)`)
    2. legacy/global key (`coupler.get(key)`)
    3. namespaced global alias (`coupler.get(namespace/key)`)
    """
    if coupler is None:
        return None, None

    namespace = infer_namespace_for_key(key)
    view_fn = getattr(coupler, "view", None)
    if namespace and callable(view_fn):
        try:
            namespaced_view = view_fn(namespace)
            view_get = getattr(namespaced_view, "get", None)
            if callable(view_get):
                local_key = local_key_for_namespace(key, namespace)
                value = view_get(local_key)
                if value is not None:
                    return value, qualify_key(namespace, local_key)
        except Exception:
            pass

    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return None, None

    value = get_value(key)
    if value is not None:
        return value, key

    alias = namespaced_alias_for_key(key)
    if alias is not None:
        value = get_value(alias)
        if value is not None:
            return value, alias

    return None, None


def namespaced_view_target(key: str) -> Optional[Tuple[str, str]]:
    """
    Return ``(namespace, local_key)`` for publishing through a namespace view.
    """
    namespace = infer_namespace_for_key(key)
    if not namespace:
        return None
    return namespace, local_key_for_namespace(key, namespace)
