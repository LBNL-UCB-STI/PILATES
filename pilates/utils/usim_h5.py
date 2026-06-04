from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import h5py
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


def ensure_usim_population_year_table_aliases(
    *,
    h5_path: str,
    year: int,
) -> Dict[str, list[str]]:
    """
    Link root-level UrbanSim population tables into a year-scoped namespace.

    UrbanSim forecast outputs can expose the current population slice at root
    keys such as ``/households``. Downstream exact-year consumers expect
    ``/<year>/households``. HDF5 hard links let the population-source snapshot
    satisfy that contract without duplicating table data.
    """
    created: list[str] = []
    existing: list[str] = []
    missing_root: list[str] = []
    with h5py.File(h5_path, "r+") as handle:
        year_group = handle.require_group(str(year))
        for table_name in POPULATION_TABLE_BY_KEY.values():
            year_key = f"/{year}/{table_name}"
            if table_name in year_group:
                existing.append(year_key)
                continue
            if table_name not in handle:
                missing_root.append(f"/{table_name}")
                continue
            year_group[table_name] = handle[table_name]
            created.append(year_key)
    return {
        "created": created,
        "existing": existing,
        "missing_root": missing_root,
    }


def _year_from_usim_model_data_filename(h5_path: str) -> Optional[int]:
    match = re.fullmatch(r"model_data_(\d{4})\.h5", Path(h5_path).name)
    if match is None:
        return None
    return int(match.group(1))


def _validate_usim_h5_filename_year(*, h5_path: str, year: Optional[int]) -> None:
    filename_year = _year_from_usim_model_data_filename(h5_path)
    if filename_year is None or year is None or filename_year == year:
        return
    raise ValueError(
        "UrbanSim population source H5 filename/year mismatch. "
        f"h5_path={h5_path} filename_year={filename_year} requested_year={year}. "
        f"A year-named model_data_<year>.h5 file must be resolved with its "
        f"matching population year; this usually means the binding layer selected "
        f"the wrong UrbanSim datastore."
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


def fingerprint_usim_h5_table(
    *,
    h5_path: str,
    year: Optional[int],
    table: str,
    allow_root_fallback: bool = True,
    nearest_year_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Return a stable semantic fingerprint for one UrbanSim H5 table.

    The fingerprint is deliberately table-scoped instead of whole-file scoped:
    ATLAS cache identity depends on the selected population table, not on
    unrelated H5 metadata or tables that ATLAS does not consume for that
    decision point.
    """
    with pd.HDFStore(h5_path, mode="r") as store:
        table_path = resolve_usim_h5_table_key(
            store,
            year=year,
            table=table,
            allow_root_fallback=allow_root_fallback,
            nearest_year_fallback=nearest_year_fallback,
        )
        if table_path not in store:
            available = sorted(store.keys())
            raise KeyError(
                "UrbanSim H5 table fingerprint target is missing. "
                f"h5_path={h5_path} year={year} table={table} "
                f"resolved_table_path={table_path} available={available}"
            )
        frame = store[table_path]

    metadata = {
        "fingerprint_version": "usim_h5_table_v1",
        "requested_year": year,
        "table": table,
        "resolved_table_path": table_path,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "index_name": None if frame.index.name is None else str(frame.index.name),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy(
        dtype="uint64",
        copy=False,
    )
    digest.update(row_hashes.tobytes())
    metadata["sha256"] = digest.hexdigest()
    return metadata


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
    _validate_usim_h5_filename_year(h5_path=h5_path, year=year)
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
    _validate_usim_h5_filename_year(h5_path=h5_path, year=year)
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
