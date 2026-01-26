from __future__ import annotations

from typing import Dict, Optional, Type

from sqlmodel import SQLModel

from pilates.database.schema.activitysim_schema import (
    HouseholdsAsimIn,
    LandUseAsimIn,
    PersonsAsimIn,
)

# Central mapping of artifact keys to curated schema classes.
# Keep keys in sync with pilates/workflows/artifact_constants.py.
_SCHEMA_BY_KEY: Dict[str, Type[SQLModel]] = {
    "households_asim_in": HouseholdsAsimIn,
    "land_use_asim_in": LandUseAsimIn,
    "persons_asim_in": PersonsAsimIn,
}


def get_consist_schemas() -> list[Type[SQLModel]]:
    """Return unique schema classes to register with Consist Tracker."""
    unique = {schema for schema in _SCHEMA_BY_KEY.values()}
    return sorted(unique, key=lambda cls: cls.__name__)


def get_schema_for_key(key: Optional[str]) -> Optional[Type[SQLModel]]:
    """Look up a curated schema for a given artifact key."""
    if not key:
        return None
    return _SCHEMA_BY_KEY.get(key)

