"""
Geospatial utility functions for PILATES.

This module provides functions for fetching and manipulating geographical data,
primarily focusing on Census TIGERweb API interactions and spatial assignments
for various geographical entities like blocks and zones.
"""

from __future__ import annotations

import time

import geopandas as gpd
import pandas as pd
import logging
import requests
from shapely.geometry import Polygon
from tqdm import tqdm
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from pilates.workspace import Workspace

from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


def _ensure_geoid_column(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize block GEOID column naming.

    Older cached shapefiles or regional sources may use alternate names like
    'geoid', 'GEOID10', or similar. This helper ensures a canonical 'GEOID'
    column exists and is string-typed.
    """
    if "GEOID" in blocks_gdf.columns:
        blocks_gdf["GEOID"] = blocks_gdf["GEOID"].astype(str)
        return blocks_gdf

    # Heuristic search for common variants
    lowered = {c.lower(): c for c in blocks_gdf.columns}
    candidates = [
        "geoid",
        "geoid10",
        "geoid20",
        "block_geoid",
        "blockid",
        "block_id",
    ]
    for cand in candidates:
        if cand in lowered:
            src = lowered[cand]
            blocks_gdf = blocks_gdf.rename(columns={src: "GEOID"})
            blocks_gdf["GEOID"] = blocks_gdf["GEOID"].astype(str)
            logger.info(f"Normalized block GEOID column from '{src}' to 'GEOID'.")
            return blocks_gdf

    raise KeyError(
        "Block geometries are missing a GEOID column. "
        f"Available columns: {blocks_gdf.columns.tolist()}"
    )


def get_county_block_geoms(
    state_fips: str,
    county_fips: str,
    zone_type: str = "block",
    result_size: int = 10000,
) -> gpd.GeoDataFrame:
    """
    Fetches block or block group geometries from the Census TIGERweb API for a given state and county.

    Args:
        state_fips (str): FIPS code of the state.
        county_fips (str): FIPS code of the county.
        zone_type (str, optional): Type of geographical zone to retrieve.
                                   Can be "block", "taz" (which triggers block geometries),
                                   or "block_group". Defaults to "block".
        result_size (int, optional): Number of records to request per API call. Defaults to 10000.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the requested geographical features
                          with 'GEOID', 'STATE', 'COUNTY', 'TRACT', 'BLKGRP', 'BLOCK',
                          'CENTLAT', 'CENTLON' attributes, and 'geometry'.
    """
    if (zone_type == "block") or (zone_type == "taz"):  # to map blocks to taz.
        base_url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
            #             'Tracts_Blocks/MapServer/12/query?where=STATE%3D{0}+and+COUNTY%3D{1}' #2020 census
            "tigerWMS_Census2010/MapServer/18/query?where=STATE%3D'{0}'+and+COUNTY%3D'{1}'"  # 2010 census
            "&resultRecordCount={2}&resultOffset={3}&orderBy=GEOID"
            "&outFields=GEOID%2CSTATE%2CCOUNTY%2CTRACT%2CBLKGRP%2CBLOCK%2CCENTLAT"
            '%2CCENTLON&outSR=%7B"wkid"+%3A+4326%7D&f=json'
        )

    elif zone_type == "block_group":
        base_url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
            #             'Tracts_Blocks/MapServer/11/query?where=STATE%3D{0}+and+COUNTY%3D{1}' #2020 census
            "tigerWMS_Census2010/MapServer/16/query?where=STATE%3D'{0}'+and+COUNTY%3D'{1}'"  # 2010 census
            "&resultRecordCount={2}&resultOffset={3}&orderBy=GEOID"
            "&outFields=GEOID%2CSTATE%2CCOUNTY%2CTRACT%2CBLKGRP%2CCENTLAT"
            '%2CCENTLON&outSR=%7B"wkid"+%3A+4326%7D&f=json'
        )

    blocks_remaining = True
    all_features = []
    page = 0
    max_pages = 100
    while blocks_remaining and page < max_pages:
        offset = page * result_size
        url = base_url.format(state_fips, county_fips, result_size, offset)
        result = requests.get(url)
        try:
            features = result.json()["features"]
        except Exception as e:
            logger.error(f"Error parsing features: {e}")
            break
        all_features += features
        if "exceededTransferLimit" in result.json().keys():
            if result.json()["exceededTransferLimit"]:
                page += 1
            else:
                blocks_remaining = False
        else:
            if len(features) == 0:
                blocks_remaining = False
            else:
                page += 1
        time.sleep(0.2)  # be nice to the API

    if page == max_pages:
        logger.warning(
            "Reached max_pages limit in get_county_block_geoms, possible infinite loop avoided."
        )

    df = pd.DataFrame()
    for feature in all_features:
        tmp = pd.DataFrame([feature["attributes"]])
        try:
            tmp["geometry"] = Polygon(
                feature["geometry"]["rings"][0], feature["geometry"]["rings"][1:]
            )
            df = pd.concat((df, tmp))
        except Exception as e:
            logger.error(
                f"Error parsing features: {e}. Geometry: {feature['geometry']}"
            )
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    return _ensure_geoid_column(gdf)


def get_block_geoms(
    settings: Dict[str, Any],
    workspace: "Workspace",
    data_dir: str = "./tmp/",
    year: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Retrieves block geometries for the specified region, either from a cached shapefile
    or by downloading them from the Census TIGERweb API.

    The function first attempts to load existing block geometries from `data_dir`.
    If not found, it downloads the geometries for all counties specified in the
    PILATES settings and saves them to `data_dir` for future use.

    Args:
        settings (Dict[str, Any]): The PILATES settings dictionary, used to extract
                                   FIPS codes (state and counties) and zone type.
        workspace (Workspace): The workspace object, though not directly used for paths here,
                               it's part of the standard signature for model inputs.
        data_dir (str, optional): Directory to store/load cached block geometries.
                                  Defaults to "./tmp/".
        year (Optional[int], optional): Simulation year. Not directly used by this function
                                        but part of the standard signature. Defaults to None.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing block geometries for the configured region.
    """
    region = get_setting(settings, "run.region") or "beam"

    # Handle both flat and nested FIPS config structures
    FIPS = get_setting(settings, "shared.geography.FIPS")
    if isinstance(FIPS, dict) and "state" not in FIPS:
        # Region-nested structure: drill down using the current region
        region_fips = FIPS.get(region, {})
        state_fips = region_fips["state"]
        county_codes = region_fips["counties"]
    else:
        # Flat structure
        state_fips = FIPS["state"]
        county_codes = FIPS["counties"]

    zone_type = get_setting(settings, "shared.geography.zones.zone_type")
    if zone_type == "taz":
        zone_type_v1 = "block"  # triger block geometries
    else:
        zone_type_v1 = zone_type

    all_block_geoms = []
    file_name = "{0}_{1}.shp".format("block", region)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(os.path.join(data_dir, file_name)):
        logger.info("Loading block geoms from disk!")
        blocks_gdf = gpd.read_file(os.path.join(data_dir, file_name))
        blocks_gdf = _ensure_geoid_column(blocks_gdf)

    else:
        logger.info(
            "No block geoms found at {0}".format(os.path.join(data_dir, file_name))
        )
        logger.info("Downloading {} geoms from Census TIGERweb API!".format(zone_type))

        # get block geoms from census tigerweb API
        for county in tqdm(
            county_codes,
            total=len(county_codes),
            desc="Getting block geoms for {0} counties".format(len(county_codes)),
        ):
            county_gdf = get_county_block_geoms(state_fips, county, "block")
            all_block_geoms.append(county_gdf)

        blocks_gdf = gpd.GeoDataFrame(
            pd.concat(all_block_geoms, ignore_index=True), crs="EPSG:4326"
        )

        blocks_gdf = _ensure_geoid_column(blocks_gdf)

        # # make sure geometries match with geometries in blocks table

        # save to disk
        logger.info(
            "Got {0} block geometries. Saving to disk.".format(len(all_block_geoms))
        )
        blocks_gdf.to_file(os.path.join(data_dir, file_name))

    return blocks_gdf


def get_taz_from_block_geoms(
    blocks_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    local_crs: str,
    zone_col_name: str,
) -> pd.Series:
    """
    Assigns Traffic Analysis Zones (TAZs) to block geometries based on spatial intersection
    and proximity.

    Blocks are assigned to TAZs first by finding the TAZ with the maximum
    intersection area. Unassigned blocks are then assigned to the nearest TAZ
    based on centroid distance.

    Args:
        blocks_gdf (gpd.GeoDataFrame): GeoDataFrame of block geometries (e.g., from `get_block_geoms`).
        zones_gdf (gpd.GeoDataFrame): GeoDataFrame of TAZ geometries, with a column
                                      identified by `zone_col_name` representing the TAZ ID.
        local_crs (str): The local Coordinate Reference System (e.g., "EPSG:26910")
                         to use for area and distance calculations.
        zone_col_name (str): The name of the column in `zones_gdf` that contains
                             the unique zone identifiers (e.g., 'TAZ').

    Returns:
        pd.Series: A pandas Series where the index is 'GEOID' from `blocks_gdf`
                   and values are the assigned TAZ IDs.
    """
    logger.info("Assigning blocks to TAZs!")

    # df to store GEOID to TAZ results
    block_to_taz_results = pd.DataFrame()

    # ignore empty geoms
    zones_gdf = zones_gdf[~zones_gdf["geometry"].is_empty]

    # convert to meter-based proj
    zones_gdf = zones_gdf.to_crs(local_crs)
    blocks_gdf = blocks_gdf.to_crs(local_crs)

    zones_gdf["zone_area"] = zones_gdf.geometry.area

    # assign zone ID's to blocks based on max area of intersection
    intx = gpd.overlay(blocks_gdf, zones_gdf.reset_index(), how="intersection")
    intx["intx_area"] = intx["geometry"].area
    intx = intx.sort_values(["GEOID", "intx_area"], ascending=False)
    intx = intx.drop_duplicates("GEOID", keep="first")

    # add to results df
    block_to_taz_results = pd.concat(
        (block_to_taz_results, intx[["GEOID", zone_col_name]])
    )

    # assign zone ID's to remaining blocks based on shortest
    # distance between block and zone centroids
    unassigned_mask = ~blocks_gdf["GEOID"].isin(block_to_taz_results["GEOID"])

    if any(unassigned_mask):
        blocks_gdf["geometry"] = blocks_gdf["geometry"].centroid
        zones_gdf["geometry"] = zones_gdf["geometry"].centroid

        all_dists = blocks_gdf.loc[unassigned_mask, "geometry"].apply(
            lambda x: zones_gdf["geometry"].distance(x)
        )

        nearest = all_dists.idxmin(axis=1).reset_index()
        nearest.columns = ["blocks_idx", zone_col_name]
        nearest.set_index("blocks_idx", inplace=True)
        nearest["GEOID"] = blocks_gdf.reindex(nearest.index)["GEOID"]

        block_to_taz_results = pd.concat(
            (block_to_taz_results, nearest[["GEOID", zone_col_name]])
        )

    return block_to_taz_results.set_index("GEOID")[zone_col_name]


def get_zone_from_points(
    df: pd.DataFrame, zones_gdf: gpd.GeoDataFrame, local_crs: str
) -> pd.Series:
    """
    Assigns zone IDs to point features based on their spatial location.

    This function performs a spatial join to determine which zone each point
    (defined by 'x' and 'y' coordinates in the input DataFrame) falls within.

    Args:
        df (pd.DataFrame): DataFrame containing point features.
                           Must have 'x' and 'y' columns representing coordinates.
                           The DataFrame's index will be used as the ID of the point feature.
        zones_gdf (gpd.GeoDataFrame): GeoDataFrame of zone geometries.
                                      Its index should represent the unique zone IDs (e.g., 'zone_id').
        local_crs (str): The local Coordinate Reference System (e.g., "EPSG:26910")
                         to use for spatial operations.

    Returns:
        pd.Series: A pandas Series where the index matches `df.index`
                   and values are the corresponding zone IDs from `zones_gdf`.
    """
    logger.info("Assigning zone IDs to {0}".format(df.index.name))
    zone_id_col = zones_gdf.index.name

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326")

    zones_gdf.geometry.crs = "EPSG:4326"

    # convert to meters-based local crs
    gdf = gdf.to_crs(local_crs)
    zones_gdf = zones_gdf.to_crs(local_crs)

    # Spatial join
    intx = gpd.sjoin(gdf, zones_gdf.reset_index(), how="left", predicate="intersects")

    if len(intx) != len(gdf):
        raise ValueError(
            "Spatial join changed row count while assigning zone IDs: "
            f"{len(gdf)} input rows, {len(intx)} joined rows."
        )

    return intx[zone_id_col]


def geoid_to_zone_map(settings, year=None):
    """
    DEPRECATED. This function is ambiguous and has been replaced by more specific
    functions in `pilates.utils.zone_utils`.

    - To get the canonical zone definition, use `get_canonical_zones`.
    - To get a block-to-zone mapping, use `get_block_to_zone_mapping`.
    """
    raise DeprecationWarning(
        "geoid_to_zone_map is deprecated. Use functions from pilates.utils.zone_utils instead."
    )
