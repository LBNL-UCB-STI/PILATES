from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from pilates.workflows.artifact_keys import (
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
)


POPULATION_TABLE_BY_KEY: Dict[str, str] = {
    USIM_POPULATION_HOUSEHOLDS_TABLE: "households",
    USIM_POPULATION_PERSONS_TABLE: "persons",
    USIM_POPULATION_JOBS_TABLE: "jobs",
    USIM_POPULATION_BLOCKS_TABLE: "blocks",
}


def resolve_usim_h5_table_key(
    store: pd.HDFStore,
    *,
    year: Optional[int],
    table: str,
    allow_root_fallback: bool = True,
    nearest_year_fallback: bool = True,
) -> str:
    """
    Resolve an UrbanSim table path inside a datastore H5.

    Prefer an exact year-scoped table when ``year`` is provided, then fall back
    to the root table and finally to the nearest available year-scoped variant.
    """
    if year is not None:
        year_key = f"/{year}/{table}"
        if year_key in store:
            return year_key
    else:
        year_key = f"/{table}"

    root_key = f"/{table}"
    if allow_root_fallback and (table in store or root_key in store):
        return root_key

    if nearest_year_fallback:
        suffix = f"/{table}"
        year_scoped_candidates = []
        for key in store.keys():
            if not key.endswith(suffix):
                continue
            parts = key.strip("/").split("/")
            if len(parts) != 2:
                continue
            year_token, table_token = parts
            if table_token != table or not year_token.isdigit():
                continue
            year_scoped_candidates.append((int(year_token), key))

        if year_scoped_candidates:
            if year is None:
                return min(year_scoped_candidates, key=lambda entry: entry[0])[1]
            prior_or_equal = [entry for entry in year_scoped_candidates if entry[0] <= year]
            if prior_or_equal:
                return max(prior_or_equal, key=lambda entry: entry[0])[1]
            return min(year_scoped_candidates, key=lambda entry: entry[0])[1]

    return year_key


def resolve_usim_population_table_paths(
    *,
    h5_path: str,
    year: Optional[int],
) -> Dict[str, str]:
    """
    Resolve the exact UrbanSim tables that represent one population slice.
    """
    with pd.HDFStore(h5_path, mode="r") as store:
        resolved = {
            semantic_key: resolve_usim_h5_table_key(
                store,
                year=year,
                table=table_name,
            )
            for semantic_key, table_name in POPULATION_TABLE_BY_KEY.items()
        }
        missing = [table_path for table_path in resolved.values() if table_path not in store]
        if missing:
            available = sorted(store.keys())
            raise KeyError(
                "UrbanSim population source is missing required tables. "
                f"missing={missing} available={available}"
            )
        return resolved
