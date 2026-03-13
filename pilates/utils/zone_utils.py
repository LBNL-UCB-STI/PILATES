"""
Utility functions for managing geographical zones and their mappings.

This module provides functionalities for loading canonical zone definitions,
generating block-to-zone mappings, and ensuring consistent indexing and metadata
for zonal data, particularly important for ActivitySim and other spatially-aware
models within the PILATES framework.
"""

import logging
import os
import inspect
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import shutil
import zarr

from pilates.config import PilatesConfig
from pilates.utils.path_utils import find_project_root

# Lazy imports of geog functions are performed inside the functions that need them to avoid circular imports.

logger = logging.getLogger(__name__)


def _log_zarr_runtime_context() -> None:
    logger.info(
        "Zarr rewrite host Python stack: xarray=%s (%s) zarr=%s (%s)",
        xr.__version__,
        getattr(xr, "__file__", "unknown"),
        getattr(zarr, "__version__", "unknown"),
        getattr(zarr, "__file__", "unknown"),
    )
    logger.info(
        "Zarr rewrite host to_zarr signature: %s",
        inspect.signature(xr.Dataset.to_zarr),
    )


def _log_zarr_store_state(label: str, skim_path: str, skims_ds: xr.Dataset) -> None:
    store_entries = []
    if os.path.isdir(skim_path):
        store_entries = sorted(os.listdir(skim_path))[:10]

    coord_summaries = []
    for coord_name in ("otaz", "dtaz"):
        if coord_name not in skims_ds.coords:
            continue
        coord = skims_ds.coords[coord_name]
        first = coord.values[0] if len(coord) else None
        last = coord.values[-1] if len(coord) else None
        coord_summaries.append(
            {
                "name": coord_name,
                "dtype": str(coord.dtype),
                "size": int(coord.size),
                "first": first.item() if hasattr(first, "item") else first,
                "last": last.item() if hasattr(last, "item") else last,
                "attrs": dict(coord.attrs),
            }
        )

    logger.info(
        "%s skims store state: path=%s entries=%s dims=%s coord_summaries=%s attrs=%s",
        label,
        skim_path,
        store_entries,
        {name: int(size) for name, size in skims_ds.sizes.items()},
        coord_summaries,
        dict(skims_ds.attrs),
    )


def load_canonical_zones(
    settings: PilatesConfig, workspace: "Workspace"
) -> gpd.GeoDataFrame:
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

    id_col = zone_config.canonical_id_col
    source_file = None

    if settings.activitysim is not None:
        source_file_basename = os.path.basename(zone_config.source_file)
        candidate = os.path.join(
            workspace.get_asim_mutable_data_dir(), source_file_basename
        )
        if os.path.exists(candidate):
            source_file = candidate

    if source_file is None:
        source_file = zone_config.source_file
        if not os.path.isabs(source_file):
            project_root = find_project_root(start_path=os.path.dirname(__file__))
            if project_root:
                source_file = os.path.join(project_root, source_file)

    if not os.path.exists(source_file):
        raise FileNotFoundError(
            f"Canonical zone source file not found: {source_file}"
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


def get_canonical_zones(
    settings: PilatesConfig, workspace: "Workspace"
) -> pd.DataFrame:
    """
    DEPRECATED: Use load_canonical_zones instead.

    This function now delegates to `load_canonical_zones` to ensure a
    sorted, canonical GeoDataFrame is always returned.

    Args:
        settings (PilatesConfig): The simulation configuration.
        workspace (Workspace): The workspace object.

    Returns:
        pd.DataFrame: A sorted GeoDataFrame of canonical zones.
    """
    logger.warning(
        "`get_canonical_zones` is deprecated and will be removed. Use `load_canonical_zones`."
    )
    return load_canonical_zones(settings, workspace)


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
    logger.info(
        f"Successfully created block to zone mapping for {len(mapping)} blocks."
    )

    return mapping


def ensure_0_based_and_flag_zarr_skims(skim_path: str, settings, workspace):
    """
    Ensures a Zarr skim file is 0-based indexed and has the 'preprocessed' flag.
    If the file is found to be 1-based, it re-indexes it to 0-based and
    adds the 'preprocessed' flag, then overwrites the original file.

    Parameters:
    -----------
    skim_path : str
        Path to the Zarr skim store.
    settings : PilatesConfig
        The full settings object, needed for loading canonical zones.
    workspace : Workspace
        The workspace object, needed for loading canonical zones.
    """
    if not os.path.exists(skim_path):
        logger.warning(
            f"Skim path not found, cannot ensure 0-based and flag: {skim_path}"
        )
        return

    logger.info(f"Ensuring 0-based indexing and 'preprocessed' flag for {skim_path}")
    try:
        _log_zarr_runtime_context()
        with xr.open_zarr(skim_path) as skims_ds:
            _log_zarr_store_state("Pre-correction", skim_path, skims_ds)
            needs_correction = False
            if (
                len(skims_ds.coords["otaz"]) > 0
                and skims_ds.coords["otaz"].values[0] == 1
            ):
                logger.warning("Skims appear to be 1-based. Re-indexing to 0-based.")
                canonical_zones_df = load_canonical_zones(settings, workspace)
                new_coords = np.arange(len(canonical_zones_df))
                skims_ds = skims_ds.assign_coords(otaz=new_coords, dtaz=new_coords)
                logger.info(
                    f"Corrected skims otaz coords: {skims_ds.coords['otaz'].values[:5]}...{skims_ds.coords['otaz'].values[-5:]}"
                )
                needs_correction = True
            else:
                logger.info("Skims already appear to be 0-based.")

            # Ensure 'preprocessed' flag is present
            if "preprocessed" not in skims_ds["otaz"].attrs:
                logger.info("Adding 'preprocessed' flag.")
                skims_ds["otaz"].attrs["preprocessed"] = "zero-based-contiguous"
                skims_ds["dtaz"].attrs["preprocessed"] = "zero-based-contiguous"
                needs_correction = True
            else:
                logger.info("'preprocessed' flag already present.")

            if needs_correction:
                # Overwrite the Zarr store with the corrected version
                # Use a temporary path for atomic write
                temp_path = f"{skim_path}_temp_corrected"
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)  # Clean up previous temp if any

                for name in skims_ds.variables:
                    skims_ds[name].encoding = {}
                try:
                    skims_ds.to_zarr(
                        temp_path, mode="w", consolidated=True, zarr_format=2
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to write skims in Zarr v2 format. ActivitySim "
                        "requires Zarr v2; if that requirement changes, update "
                        "the skims writer to allow newer formats."
                    ) from e

                # Atomically replace the original
                if os.path.exists(skim_path):
                    shutil.rmtree(skim_path)
                os.rename(temp_path, skim_path)
                with xr.open_zarr(skim_path) as corrected_ds:
                    _log_zarr_store_state("Post-correction", skim_path, corrected_ds)
                logger.info(
                    f"Successfully corrected and flagged Zarr skims at {skim_path}"
                )
            else:
                _log_zarr_store_state("No-op", skim_path, skims_ds)
                logger.info(f"No correction needed for Zarr skims at {skim_path}")

    except Exception as e:
        logger.error(
            f"Failed to ensure 0-based and flag Zarr skims: {e}", exc_info=True
        )
        raise  # Re-raise to ensure pipeline fails if this critical step fails
