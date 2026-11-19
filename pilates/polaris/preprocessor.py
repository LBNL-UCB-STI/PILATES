"""Reusable POLARIS supply-database preprocessing helpers."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)

POLARIS_EMPLOYMENT_COLUMNS: tuple[str, ...] = (
    "employment_retail",
    "employment_government",
    "employment_manufacturing",
    "employment_services",
    "employment_industrial",
    "employment_other",
    "employment_total",
)

SECTOR_TO_POLARIS_EMPLOYMENT_COLUMN: dict[str, str] = {
    "11": "employment_industrial",
    "21": "employment_industrial",
    "22": "employment_industrial",
    "23": "employment_industrial",
    "31": "employment_manufacturing",
    "32": "employment_manufacturing",
    "33": "employment_manufacturing",
    "31-33": "employment_manufacturing",
    "42": "employment_retail",
    "44": "employment_retail",
    "45": "employment_retail",
    "44-45": "employment_retail",
    "48": "employment_industrial",
    "49": "employment_industrial",
    "48-49": "employment_industrial",
    "51": "employment_services",
    "52": "employment_services",
    "53": "employment_services",
    "54": "employment_services",
    "55": "employment_services",
    "56": "employment_other",
    "61": "employment_government",
    "62": "employment_other",
    "71": "employment_other",
    "72": "employment_services",
    "81": "employment_services",
    "92": "employment_government",
}


def aggregate_zonal_employment(
    jobs_df: pd.DataFrame,
    block_to_zone: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Aggregate UrbanSim jobs by zone and POLARIS employment category.

    Parameters
    ----------
    jobs_df : pd.DataFrame
        UrbanSim jobs table containing ``sector_id`` and either ``zone_id``
        or ``block_id``.
    block_to_zone : Mapping[str, int] | None, optional
        Mapping from census ``block_id`` to POLARIS ``zone_id`` when ``jobs_df``
        does not already include a ``zone_id`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``zone`` with columns listed in
        ``POLARIS_EMPLOYMENT_COLUMNS``.
    """
    if "sector_id" not in jobs_df.columns:
        raise ValueError("jobs_df must contain a 'sector_id' column.")

    working = jobs_df.copy()
    if "zone_id" in working.columns:
        working["zone"] = working["zone_id"].astype(int)
    elif "block_id" in working.columns and block_to_zone is not None:
        block_series = working["block_id"].astype(str)
        working["zone"] = block_series.map(block_to_zone)
        working = working.dropna(subset=["zone"]).copy()
        working["zone"] = working["zone"].astype(int)
    else:
        raise ValueError(
            "jobs_df must include 'zone_id' or 'block_id' with block_to_zone mapping."
        )

    sector_cols = list(POLARIS_EMPLOYMENT_COLUMNS[:-1])
    working["employment_column"] = (
        working["sector_id"]
        .astype(str)
        .map(SECTOR_TO_POLARIS_EMPLOYMENT_COLUMN)
        .fillna("employment_other")
    )

    grouped = (
        working.groupby(["zone", "employment_column"])
        .size()
        .unstack(fill_value=0)
    )
    for col in sector_cols:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped = grouped[sector_cols].astype(int)
    grouped["employment_total"] = grouped[sector_cols].sum(axis=1).astype(int)
    grouped.index.name = "zone"
    return grouped


def update_supply_zone_employment(
    db_supply_path: Path | str,
    jobs_df: pd.DataFrame,
    block_to_zone: Mapping[str, int] | None = None,
) -> int:
    """Update the ``zone`` table employment columns in ``{db_name}-Supply.sqlite``.

    Zeros out all 7 ``employment_*`` columns on ``zone`` first so zones with no
    jobs in ``jobs_df`` do not retain stale employment counts, then updates
    ``employment_total``, ``employment_retail``, ``employment_government``,
    ``employment_manufacturing``, ``employment_services``,
    ``employment_industrial``, and ``employment_other`` for each zone present in
    ``jobs_df`` (fixing the legacy ``Zone.Update_Zone_in_DB`` copy-paste bug that
    wrote ``employment_retail = ?`` six times).

    Parameters
    ----------
    db_supply_path : Path | str
        Path to the POLARIS ``{db_name}-Supply.sqlite`` database.
    jobs_df : pd.DataFrame
        UrbanSim jobs table containing ``sector_id`` and zone/block identifiers.
    block_to_zone : Mapping[str, int] | None, optional
        Mapping from census ``block_id`` to POLARIS ``zone_id``.

    Returns
    -------
    int
        Number of zone rows updated from ``jobs_df`` in ``{db_name}-Supply.sqlite``.
    """
    supply_path = Path(db_supply_path)
    if not supply_path.exists():
        raise FileNotFoundError(
            f"POLARIS Supply SQLite database not found: {supply_path}"
        )

    zonal_employment = aggregate_zonal_employment(jobs_df, block_to_zone=block_to_zone)
    reset_sql = (
        "UPDATE zone SET "
        "employment_total = 0, "
        "employment_retail = 0, "
        "employment_government = 0, "
        "employment_manufacturing = 0, "
        "employment_services = 0, "
        "employment_industrial = 0, "
        "employment_other = 0;"
    )
    update_sql = (
        "UPDATE zone SET "
        "employment_total = ?, "
        "employment_retail = ?, "
        "employment_government = ?, "
        "employment_manufacturing = ?, "
        "employment_services = ?, "
        "employment_industrial = ?, "
        "employment_other = ? "
        "WHERE zone = ?;"
    )

    records = [
        (
            int(row["employment_total"]),
            int(row["employment_retail"]),
            int(row["employment_government"]),
            int(row["employment_manufacturing"]),
            int(row["employment_services"]),
            int(row["employment_industrial"]),
            int(row["employment_other"]),
            int(zone_id),
        )
        for zone_id, row in zonal_employment.iterrows()
    ]

    with sqlite3.connect(supply_path) as conn:
        conn.execute(reset_sql)
        conn.executemany(update_sql, records)
        conn.commit()

    logger.info(
        "Updated employment totals for %d zones in %s",
        len(records),
        supply_path,
    )
    return len(records)
