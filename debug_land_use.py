#!/usr/bin/env python
"""
Debug script to identify the land_use table join issue.
This script will help diagnose why demographic data isn't being joined correctly.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_land_use_join(zones, persons, households, jobs, asim_zone_id_col="TAZ"):
    """
    Debug helper to identify join mismatches in _create_land_use_table.

    Args:
        zones: zones GeoDataFrame
        persons: persons DataFrame
        households: households DataFrame
        jobs: jobs DataFrame
        asim_zone_id_col: The TAZ column name (default "TAZ")
    """

    logger.info("=" * 80)
    logger.info("DEBUGGING LAND USE TABLE CREATION")
    logger.info("=" * 80)

    # 1. Check zones index
    logger.info("\n1. ZONES DataFrame:")
    logger.info(f"   Index name: {zones.index.name}")
    logger.info(f"   Index dtype: {zones.index.dtype}")
    logger.info(f"   Number of zones: {len(zones)}")
    logger.info(f"   First 5 zone IDs: {zones.index[:5].tolist()}")
    logger.info(f"   Sample zone IDs: {sorted(zones.index.unique())[:10]}")

    # 2. Check if TAZ column exists in input tables
    logger.info(f"\n2. Checking '{asim_zone_id_col}' column existence:")
    for name, df in [("persons", persons), ("households", households), ("jobs", jobs)]:
        has_col = asim_zone_id_col in df.columns
        logger.info(f"   {name}: TAZ column exists = {has_col}")
        if has_col:
            logger.info(f"      TAZ dtype: {df[asim_zone_id_col].dtype}")
            logger.info(f"      Unique TAZ values: {df[asim_zone_id_col].nunique()}")
            logger.info(
                f"      First 5 TAZ values: {df[asim_zone_id_col].head().tolist()}"
            )
            logger.info(
                f"      Sample TAZ values: {sorted(df[asim_zone_id_col].unique())[:10]}"
            )

    # 3. Simulate the aggregation to see what index the aggregated data will have
    logger.info("\n3. Simulating aggregations:")

    if asim_zone_id_col in persons.columns:
        persons_test = persons.groupby(asim_zone_id_col).size()
        logger.info(f"   persons_agg index dtype: {persons_test.index.dtype}")
        logger.info(
            f"   persons_agg index values (first 5): {persons_test.index[:5].tolist()}"
        )
        logger.info(f"   persons_agg has {len(persons_test)} unique zones")
    else:
        logger.error(f"   persons does NOT have '{asim_zone_id_col}' column!")

    if asim_zone_id_col in households.columns:
        households_test = households.groupby(asim_zone_id_col).size()
        logger.info(f"   households_agg index dtype: {households_test.index.dtype}")
        logger.info(
            f"   households_agg index values (first 5): {households_test.index[:5].tolist()}"
        )
        logger.info(f"   households_agg has {len(households_test)} unique zones")
    else:
        logger.error(f"   households does NOT have '{asim_zone_id_col}' column!")

    if asim_zone_id_col in jobs.columns:
        jobs_test = jobs.groupby(asim_zone_id_col).size()
        logger.info(f"   jobs_agg index dtype: {jobs_test.index.dtype}")
        logger.info(
            f"   jobs_agg index values (first 5): {jobs_test.index[:5].tolist()}"
        )
        logger.info(f"   jobs_agg has {len(jobs_test)} unique zones")
    else:
        logger.error(f"   jobs does NOT have '{asim_zone_id_col}' column!")

    # 4. Check for join compatibility
    logger.info("\n4. Join compatibility check:")
    if asim_zone_id_col in persons.columns and asim_zone_id_col in households.columns:
        zones_ids = set(zones.index.astype(str))
        persons_ids = set(persons[asim_zone_id_col].astype(str).unique())
        households_ids = set(households[asim_zone_id_col].astype(str).unique())

        zones_not_in_data = zones_ids - persons_ids - households_ids
        data_not_in_zones = (persons_ids | households_ids) - zones_ids

        logger.info(f"   Zone IDs in zones but not in data: {len(zones_not_in_data)}")
        if zones_not_in_data and len(zones_not_in_data) <= 10:
            logger.info(f"      Examples: {list(zones_not_in_data)[:10]}")

        logger.info(f"   Zone IDs in data but not in zones: {len(data_not_in_zones)}")
        if data_not_in_zones and len(data_not_in_zones) <= 10:
            logger.info(f"      Examples: {list(data_not_in_zones)[:10]}")

        # Check dtype compatibility
        logger.info("\n5. Data type compatibility:")
        logger.info(f"   zones.index dtype: {zones.index.dtype}")
        if asim_zone_id_col in persons.columns:
            logger.info(f"   persons[TAZ] dtype: {persons[asim_zone_id_col].dtype}")
        if asim_zone_id_col in households.columns:
            logger.info(
                f"   households[TAZ] dtype: {households[asim_zone_id_col].dtype}"
            )
        if asim_zone_id_col in jobs.columns:
            logger.info(f"   jobs[TAZ] dtype: {jobs[asim_zone_id_col].dtype}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    # This is meant to be imported and used in the preprocessor
    print("This module provides debug_land_use_join() function.")
    print("Import it in your preprocessor code and call it before the join.")
