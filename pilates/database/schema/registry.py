from __future__ import annotations

from typing import Dict, Optional, Type

from sqlmodel import SQLModel
from pilates.workflows.artifact_key_migrations import resolve_artifact_key

from pilates.database.schema.activitysim_schema import (
    HouseholdsAsimIn,
    HouseholdsAsimOut,
    LandUseAsimIn,
    PersonsAsimIn,
    PersonsAsimOut,
)
from pilates.database.schema.atlas_schema import (
    AtlasBlocks,
    AtlasGrave,
    AtlasHousehold,
    AtlasJobs,
    AtlasPersons,
    AtlasResidential,
    VehiclesAtlasOut,
)
from pilates.database.schema.beam_schema import (
    BeamLinkstats,
    BeamNetworkFinal,
    PlansBeamIn,
)

# Central mapping of artifact keys to curated schema classes.
# Keep keys in sync with pilates/workflows/artifact_keys.py.
_SCHEMA_BY_KEY: Dict[str, Type[SQLModel]] = {
    "households_asim_in": HouseholdsAsimIn,
    "land_use_asim_in": LandUseAsimIn,
    "persons_asim_in": PersonsAsimIn,
    "households_beam_in": HouseholdsAsimOut,
    "persons_beam_in": PersonsAsimOut,
    "plans_beam_in": PlansBeamIn,
    "beam_network_final": BeamNetworkFinal,
    "linkstats": BeamLinkstats,
    "linkstats_warmstart": BeamLinkstats,
    "atlas_blocks_csv": AtlasBlocks,
    "atlas_grave_csv": AtlasGrave,
    "atlas_households_csv": AtlasHousehold,
    "atlas_jobs_csv": AtlasJobs,
    "atlas_persons_csv": AtlasPersons,
    "atlas_residential_csv": AtlasResidential,
    "atlas_vehicles2_input": VehiclesAtlasOut,
    "atlas_vehicles2_output": VehiclesAtlasOut,
}

_SCHEMA_BY_PREFIX: Dict[str, Type[SQLModel]] = {
    # BEAM linkstats artifacts are commonly iteration/version-scoped keys, e.g.
    # linkstats_parquet_2018_0_sub1 or linkstats_unmodified_parquet__y2018__i0__...
    "linkstats_": BeamLinkstats,
}


def get_consist_schemas() -> list[Type[SQLModel]]:
    """Return unique schema classes to register with Consist Tracker."""
    unique = {schema for schema in _SCHEMA_BY_KEY.values()}
    return sorted(unique, key=lambda cls: cls.__name__)


def get_schema_for_key(key: Optional[str]) -> Optional[Type[SQLModel]]:
    """Look up a curated schema for a given artifact key."""
    if not key:
        return None
    resolved_key = resolve_artifact_key(key)
    exact = _SCHEMA_BY_KEY.get(resolved_key)
    if exact is not None:
        return exact
    for prefix, schema in _SCHEMA_BY_PREFIX.items():
        if resolved_key.startswith(prefix):
            return schema
    return None
