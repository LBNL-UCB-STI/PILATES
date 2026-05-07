from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd

from pilates.workflows.artifact_keys import (
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
)

logger = logging.getLogger(__name__)


POPULATION_TABLE_BY_KEY: Dict[str, str] = {
    USIM_POPULATION_HOUSEHOLDS_TABLE: "households",
    USIM_POPULATION_PERSONS_TABLE: "persons",
    USIM_POPULATION_JOBS_TABLE: "jobs",
    USIM_POPULATION_BLOCKS_TABLE: "blocks",
}


def allow_root_population_tables_for_target_year_datastore(
    h5_path: str,
    year: Optional[int],
) -> bool:
    """
    Return whether root population tables can represent ``year`` for this file.

    UrbanSim and ATLAS can publish year-specific datastores such as
    ``model_data_2021.h5`` whose population tables live at the H5 root. This
    should not be treated like a stale root fallback from an older datastore.
    """
    if year is None:
        return False
    name = Path(str(h5_path)).name
    return re.search(rf"(?:^|[_-]){int(year)}(?:\.h5$|[_-])", name) is not None


def should_require_exact_population_year_tables(
    *,
    h5_path: str,
    year: Optional[int],
    require_exact_year: bool,
) -> bool:
    """
    Apply ActivitySim's exact-year table policy to one H5 datastore path.
    """
    return bool(require_exact_year) and not (
        allow_root_population_tables_for_target_year_datastore(h5_path, year)
    )


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
            prior_or_equal = [
                entry for entry in year_scoped_candidates if entry[0] <= year
            ]
            if prior_or_equal:
                return max(prior_or_equal, key=lambda entry: entry[0])[1]
            return min(year_scoped_candidates, key=lambda entry: entry[0])[1]

    return year_key


def resolve_usim_population_table_paths(
    *,
    h5_path: str,
    year: Optional[int],
    require_exact_year: bool = False,
) -> Dict[str, str]:
    """
    Resolve the exact UrbanSim tables that represent one population slice.
    """
    if require_exact_year and year is None:
        raise ValueError(
            "Exact-year UrbanSim population table resolution requires a year."
        )
    with pd.HDFStore(h5_path, mode="r") as store:
        resolved = {
            semantic_key: resolve_usim_h5_table_key(
                store,
                year=year,
                table=table_name,
                allow_root_fallback=not require_exact_year,
                nearest_year_fallback=not require_exact_year,
            )
            for semantic_key, table_name in POPULATION_TABLE_BY_KEY.items()
        }
        missing = [
            table_path for table_path in resolved.values() if table_path not in store
        ]
        if missing:
            available = sorted(store.keys())
            raise KeyError(
                "UrbanSim population source is missing required tables. "
                f"h5_path={h5_path} year={year} require_exact_year={require_exact_year} "
                f"missing={missing} available={available}"
            )
        return resolved


def reconcile_usim_population_table_paths(
    *,
    h5_path: str,
    year: Optional[int],
    provided_paths: Optional[Mapping[str, str]] = None,
    require_exact_year: bool = False,
) -> Dict[str, str]:
    """
    Validate optional pre-resolved table paths against the actual H5 contents.

    When stale metadata points at tables that are not present in the bound H5,
    fall back to year-aware resolution for just those missing entries.
    """
    if require_exact_year and year is None:
        raise ValueError(
            "Exact-year UrbanSim population table reconciliation requires a year."
        )
    normalized_provided = {
        semantic_key: (
            table_path
            if str(table_path).startswith("/")
            else f"/{str(table_path).lstrip('/')}"
        )
        for semantic_key, table_path in (provided_paths or {}).items()
        if table_path
    }
    with pd.HDFStore(h5_path, mode="r") as store:
        resolved: Dict[str, str] = {}
        for semantic_key, table_name in POPULATION_TABLE_BY_KEY.items():
            provided_path = normalized_provided.get(semantic_key)
            exact_year_path = f"/{year}/{table_name}" if year is not None else None
            if provided_path and provided_path in store:
                if (
                    exact_year_path is not None
                    and provided_path != exact_year_path
                    and exact_year_path in store
                ):
                    logger.warning(
                        "Ignoring stale pre-resolved UrbanSim population table "
                        "path for %s: provided=%s selected=%s year=%s",
                        semantic_key,
                        provided_path,
                        exact_year_path,
                        year,
                    )
                    resolved[semantic_key] = exact_year_path
                    continue
                if require_exact_year and exact_year_path is not None:
                    resolved[semantic_key] = exact_year_path
                    continue
                resolved[semantic_key] = provided_path
                continue
            resolved[semantic_key] = resolve_usim_h5_table_key(
                store,
                year=year,
                table=table_name,
                allow_root_fallback=not require_exact_year,
                nearest_year_fallback=not require_exact_year,
            )

        missing = [
            table_path for table_path in resolved.values() if table_path not in store
        ]
        if missing:
            available = sorted(store.keys())
            raise KeyError(
                "UrbanSim population source is missing required tables. "
                f"h5_path={h5_path} year={year} require_exact_year={require_exact_year} "
                f"missing={missing} available={available}"
            )
        return resolved
