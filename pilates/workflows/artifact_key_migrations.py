"""
Artifact key alias/migration helpers.

Canonical keys remain defined in ``artifact_keys.py``. This module supports
temporary aliases during migration and lets coupler/provenance interfaces
accept legacy or alternate spellings safely.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from pilates.workflows.artifact_keys import ArtifactKeys as K


ARTIFACT_KEY_ALIASES: Dict[str, str] = {
    # Proposed "asim_*_in" naming to current canonical values.
    "asim_land_use_in": K.ASIM_LAND_USE_IN,
    "asim_households_in": K.ASIM_HOUSEHOLDS_IN,
    "asim_persons_in": K.ASIM_PERSONS_IN,
    "asim_omx_skims": K.ASIM_OMX_SKIMS,
    # UrbanSim key migration aliases.
    "usim_datastore_current_h5": K.USIM_DATASTORE_CURRENT_H5,
    "usim_datastore_h5": K.USIM_DATASTORE_CURRENT_H5,
    "usim_forecast_output": K.USIM_FORECAST_OUTPUT,
    "usim_population_source_h5": K.USIM_POPULATION_SOURCE_H5,
    "usim_h5_updated": K.USIM_DATASTORE_CURRENT_H5,
    "usim_input_next": K.USIM_DATASTORE_CURRENT_H5,
}


def resolve_artifact_key(key: str) -> str:
    """
    Resolve an artifact key alias to its canonical key.
    """
    return ARTIFACT_KEY_ALIASES.get(key, key)


def canonicalize_artifact_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Canonicalize mapping keys using ``resolve_artifact_key``.

    Canonical key entries take precedence over alias entries when both exist.
    """
    canonical: Dict[str, Any] = {}
    for raw_key, value in mapping.items():
        resolved = resolve_artifact_key(raw_key)
        if resolved in canonical and raw_key != resolved:
            continue
        canonical[resolved] = value
    return canonical
