"""Reusable POLARIS postprocessing helpers for UrbanSim feedback and SQLite summaries."""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)


def _load_location_to_zone_map(db_supply_path: Path) -> dict[int, int]:
    """Load the ``Location.location -> Location.zone`` mapping from ``Supply.sqlite``."""
    with sqlite3.connect(db_supply_path) as supply_conn:
        rows = supply_conn.execute("SELECT location, zone FROM Location;").fetchall()
    return {int(location): int(zone) for location, zone in rows}


def extract_work_and_school_locations(
    db_demand_path: Path | str,
    db_supply_path: Path | str | None = None,
    usim_datastore_path: Path | str | None = None,
    forecast_year: int | None = None,
    *,
    map_locations_to_zones: bool = False,
) -> pd.DataFrame:
    """Extract work and school locations from POLARIS SQLite DBs and update UrbanSim HDF5.

    Reads ``Person`` (and ``Household``) records from ``{db_name}-Demand.sqlite``,
    converts 0-indexed person IDs to 1-indexed ``member_id`` values (with composite
    key ``{household_id}_{person_index + 1}``), optionally maps ``work_location_id``
    and ``school_location_id`` to TAZ ``zone_id`` via ``Location`` in
    ``{db_name}-Supply.sqlite``, and updates ``work_zone_id`` and ``school_zone_id``
    on the ``persons`` table in the UrbanSim HDF5 datastore without clobbering
    unsampled persons with ``NaN``.

    Parameters
    ----------
    db_demand_path : Path | str
        Path to ``{db_name}-Demand.sqlite``.
    db_supply_path : Path | str | None, optional
        Path to ``{db_name}-Supply.sqlite``. When provided, populates mapped
        ``work_taz_zone_id`` and ``school_taz_zone_id`` columns (and uses them for
        ``work_zone_id`` / ``school_zone_id`` when ``map_locations_to_zones=True``).
    usim_datastore_path : Path | str | None, optional
        Path to the UrbanSim HDF5 datastore to update in place. If ``None``,
        only returns the extracted person location DataFrame.
    forecast_year : int | None, optional
        Forecast year key prefix in the HDF5 datastore (e.g. ``2020`` for
        ``"2020/persons"``; ``None`` uses ``"persons"``).
    map_locations_to_zones : bool, optional
        If ``True``, requires ``db_supply_path`` and writes TAZ ``zone`` IDs into
        ``work_zone_id`` / ``school_zone_id``. Defaults to ``False`` to preserve the
        legacy ``origin/main`` convention where UrbanSim's ``work_zone_id`` and
        ``school_zone_id`` store POLARIS ``work_location_id`` and
        ``school_location_id`` directly.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``household_id``, ``member_id``, ``person_key``
        (``{household_id}_{member_id}``), ``work_location_id``,
        ``school_location_id``, ``work_zone_id``, and ``school_zone_id``.
    """
    demand_path = Path(db_demand_path)
    if not demand_path.exists():
        raise FileNotFoundError(
            f"POLARIS Demand SQLite database not found: {demand_path}"
        )
    if map_locations_to_zones and db_supply_path is None:
        raise ValueError(
            "db_supply_path is required when map_locations_to_zones=True."
        )

    loc_to_zone: dict[int, int] | None = None
    if db_supply_path is not None:
        supply_path = Path(db_supply_path)
        if not supply_path.exists():
            raise FileNotFoundError(
                f"POLARIS Supply SQLite database not found: {supply_path}"
            )
        loc_to_zone = _load_location_to_zone_map(supply_path)

    query = (
        "SELECT p.person, p.household AS household_id, p.id AS person_index, "
        "p.work_location_id, p.school_location_id "
        "FROM Person p "
        "INNER JOIN Household h ON p.household = h.household;"
    )
    with sqlite3.connect(demand_path) as demand_conn:
        per_df = pd.read_sql_query(query, demand_conn)

    per_df["household_id"] = per_df["household_id"].astype(int)
    per_df["member_id"] = per_df["person_index"].astype(int) + 1
    per_df["person_key"] = (
        per_df["household_id"].astype(str) + "_" + per_df["member_id"].astype(str)
    )
    per_df["work_location_id"] = per_df["work_location_id"].fillna(-1).astype(int)
    per_df["school_location_id"] = per_df["school_location_id"].fillna(-1).astype(int)

    if loc_to_zone is not None:
        per_df["work_taz_zone_id"] = (
            per_df["work_location_id"].map(loc_to_zone).fillna(-1).astype(int)
        )
        per_df["school_taz_zone_id"] = (
            per_df["school_location_id"].map(loc_to_zone).fillna(-1).astype(int)
        )

    if map_locations_to_zones and loc_to_zone is not None:
        per_df["work_zone_id"] = per_df["work_taz_zone_id"]
        per_df["school_zone_id"] = per_df["school_taz_zone_id"]
    else:
        per_df["work_zone_id"] = per_df["work_location_id"]
        per_df["school_zone_id"] = per_df["school_location_id"]

    if usim_datastore_path is not None:
        h5_path = Path(usim_datastore_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"UrbanSim HDF5 datastore not found: {h5_path}")

        persons_key = (
            f"{forecast_year}/persons" if forecast_year is not None else "persons"
        )
        lookup = per_df.set_index(["household_id", "member_id"])

        with pd.HDFStore(str(h5_path), mode="a") as store:
            persons_df = store[persons_key]
            idx_name = persons_df.index.name or "person_id"
            persons_df.index.name = idx_name

            indexed_persons = persons_df.reset_index().set_index(
                ["household_id", "member_id"], drop=False
            )
            for col in ("work_zone_id", "school_zone_id"):
                existing = (
                    indexed_persons[col]
                    if col in indexed_persons.columns
                    else pd.Series(-1, index=indexed_persons.index, dtype=int)
                )
                updated_col = lookup[col].reindex(indexed_persons.index).combine_first(
                    existing
                )
                indexed_persons[col] = updated_col.fillna(-1).astype(int)

            updated_persons = indexed_persons.set_index(idx_name)
            store[persons_key] = updated_persons

        logger.info("Updated %s in UrbanSim datastore %s", persons_key, h5_path)

    return per_df


def summarize_polaris(
    db_demand_path: Path | str,
    db_supply_path: Path | str,
    db_result_path: Path | str,
    analysis_sql_path: Path | str | None = None,
) -> pd.DataFrame:
    """Run multi-SQLite ``ATTACH`` summary queries across Demand, Supply, and Result DBs.

    Attaches ``{db_name}-Supply.sqlite`` as schema ``a`` and
    ``{db_name}-Result.sqlite`` as schema ``b`` on a connection to
    ``{db_name}-Demand.sqlite``, optionally executes an analysis SQL script
    (e.g. ``wtf_baseline_analysis.sql``), and returns the zonal average wait
    times from ``b.ZoneWaitTimes``.

    Parameters
    ----------
    db_demand_path : Path | str
        Path to ``{db_name}-Demand.sqlite``.
    db_supply_path : Path | str
        Path to ``{db_name}-Supply.sqlite``.
    db_result_path : Path | str
        Path to ``{db_name}-Result.sqlite``.
    analysis_sql_path : Path | str | None, optional
        Optional path to a SQL analysis script to execute while ``Supply`` (``a``)
        and ``Result`` (``b``) databases are attached.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["zone", "avg_wait_minutes"]`` aggregated from
        ``b.ZoneWaitTimes``.
    """
    demand_path = Path(db_demand_path)
    supply_path = Path(db_supply_path)
    result_path = Path(db_result_path)

    for label, path in (
        ("Demand", demand_path),
        ("Supply", supply_path),
        ("Result", result_path),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"POLARIS {label} SQLite database not found: {path}"
            )

    with sqlite3.connect(demand_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("ATTACH DATABASE ? AS a;", (str(supply_path),))
        conn.execute("ATTACH DATABASE ? AS b;", (str(result_path),))

        if analysis_sql_path is not None:
            sql_path = Path(analysis_sql_path)
            if not sql_path.exists():
                raise FileNotFoundError(
                    f"POLARIS analysis SQL script not found: {sql_path}"
                )
            conn.executescript(sql_path.read_text(encoding="utf-8"))
            conn.commit()

        zone_wait_df = pd.read_sql_query(
            "SELECT zone, AVG(avg_wait_minutes) AS avg_wait_minutes "
            "FROM b.ZoneWaitTimes "
            "GROUP BY zone ORDER BY zone;",
            conn,
        )

    return zone_wait_df
