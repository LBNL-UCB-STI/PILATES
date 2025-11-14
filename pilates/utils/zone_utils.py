import logging
import os
import pandas as pd
import geopandas as gpd

from pilates.config import PilatesConfig
from pilates.workspace import Workspace

# Lazy imports of geog functions are performed inside the functions that need them to avoid circular imports.

logger = logging.getLogger(__name__)


def load_canonical_zones(settings: PilatesConfig, workspace: Workspace) -> gpd.GeoDataFrame:
    """
    Loads the canonical zone geometries from the mutable ActivitySim data directory,
    validates them, and returns a sorted GeoDataFrame.

    This function is the new single source of truth for zone definitions.

    Args:
        settings (PilatesConfig): The simulation configuration.
        workspace (Workspace): The workspace object.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame of zone geometries, indexed and sorted
                          by the canonical zone ID.
    """
    logger.info("--- Loading Canonical Zone Geometries ---")
    zone_config = settings.shared.geography.zones

    source_file_basename = os.path.basename(zone_config.source_file)
    source_file = os.path.join(
        workspace.get_asim_mutable_data_dir(),
        source_file_basename
    )
    id_col = zone_config.canonical_id_col

    if not os.path.exists(source_file):
        raise FileNotFoundError(
            f"Canonical zone source file not found: {source_file}. It should have been copied by the ActivitySim preprocessor."
        )

    logger.info(f"Reading canonical zones from '{source_file}'")
    gdf = gpd.read_file(source_file)

    # Validate that the canonical ID column exists
    if id_col not in gdf.columns:
        raise KeyError(
            f"Canonical ID column '{id_col}' not found in source file "
            f"'{source_file}'. Available columns: {gdf.columns.tolist()}"
        )

    # Validate that the canonical IDs are unique
    if not gdf[id_col].is_unique:
        raise ValueError(f"Canonical ID column '{id_col}' contains duplicate values.")

    # Set the index to the canonical ID
    gdf = gdf.set_index(id_col)

    # Rename the index to the ActivitySim expected index name
    asim_index_col = zone_config.activitysim_index_col
    if gdf.index.name != asim_index_col:
        gdf.index.name = asim_index_col

    # Sort the GeoDataFrame by the index (the canonical ID)
    # This is critical for ensuring consistent order for all downstream models
    gdf = gdf.sort_index()
    gdf.index = gdf.index.astype(str)

    logger.info(f"Successfully loaded and sorted {len(gdf)} canonical zones.")
    return gdf


def get_canonical_zones(settings: PilatesConfig, workspace: Workspace) -> pd.DataFrame:
    """
    Reads the canonical zone definitions from the mutable ActivitySim data directory.

    Args:
        settings (PilatesConfig): The simulation configuration.
        workspace (Workspace): The workspace object.

    Returns:
        pd.DataFrame: DataFrame with 'zone_key' and 'asim_id' columns.
    """
    zone_config = settings.shared.geography.zones
    source_file_basename = os.path.basename(zone_config.source_file)

    path = os.path.join(
        workspace.get_asim_mutable_data_dir(),
        source_file_basename
    )

    if not os.path.exists(path):
        zones = load_canonical_zones(settings, workspace)
    else:
        zones = gpd.read_file(path)
    return zones


def get_block_to_zone_mapping(settings, year, workspace):
    """
    Generates a mapping from census block GEOID to the canonical zone ID.

    This function performs a geometric mapping by finding the zone with the
    largest area of intersection for each block.

    Args:
        settings (PilatesConfig): The simulation configuration.
        year (int): The simulation year (used by get_block_geoms).
        workspace (Workspace): The workspace object.

    Returns:
        dict: A dictionary mapping block GEOID to canonical zone ID.
    """
    logger.info("--- Generating Block to Zone Mapping via Geometric Intersection ---")

    # 1. Load the authoritative, sorted canonical zone geometries
    canonical_zones_gdf = load_canonical_zones(settings, workspace)
    zone_id_col = canonical_zones_gdf.index.name

    # 2. Get the census block geometries
    # This function handles its own caching and downloading from TIGERweb API
    from pilates.utils.geog import get_block_geoms, get_taz_from_block_geoms
    blocks_gdf = get_block_geoms(settings, year=year, workspace=workspace)

    # 3. Perform the geometric mapping
    # This function finds the zone with max intersection area for each block
    local_crs = settings.shared.geography.local_crs
    block_to_zone_series = get_taz_from_block_geoms(
        blocks_gdf, canonical_zones_gdf, local_crs, zone_id_col
    )

    # 4. Convert the resulting Series to a dictionary
    mapping = block_to_zone_series.to_dict()
    logger.info(f"Successfully created block to zone mapping for {len(mapping)} blocks.")

    return mapping
