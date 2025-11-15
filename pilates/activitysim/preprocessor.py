import glob
import logging
import os
import shutil
import time
from multiprocessing import Pool, cpu_count
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import openmatrix as omx
import pandas as pd
import requests
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
import matplotlib.pyplot as plt

from pilates.config import PilatesConfig
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils.duckdb_manager import DuckDBManager
from pilates.utils.geog import get_zone_from_points, get_block_geoms
from pilates.utils.zone_utils import (
    get_block_to_zone_mapping,
    get_canonical_zones,
    load_canonical_zones,
)
from pilates.utils.io import get_merged_usim_input_datastore_path, read_datastore
from pilates.utils.provenance import FileProvenanceTracker, find_project_root
from pilates.utils.database_upload import create_database_manager
from pilates.utils.settings_helper import get as get_setting
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

# Define expected data types for BEAM raw skims CSV files.
# This helps in efficient memory usage and correct data interpretation when reading large CSVs.
beam_skims_types = {
    "timePeriod": str,
    "pathType": str,
    "origin": str,
    "destination": str,
    "TIME_minutes": np.float32,
    "TOTIVT_IVT_minutes": np.float32,
    "VTOLL_FAR": np.float32,
    "DIST_meters": np.float32,
    "WACC_minutes": np.float32,
    "WAUX_minutes": np.float32,
    "WEGR_minutes": np.float32,
    "DTIM_minutes": np.float32,
    "DDIST_meters": np.float32,
    "KEYIVT_minutes": np.float32,
    "FERRYIVT_minutes": np.float32,
    "BOARDS": np.float32,
    "DEBUG_TEXT": str,
}

# Define expected data types for BEAM raw origin skims CSV files.
# These skims typically contain information related to ride-hail services.
beam_origin_skims_types = {
    "origin": str,
    "timePeriod": str,
    "reservationType": str,
    "waitTimeInMinutes": np.float32,
    "costPerMile": np.float32,
    "unmatchedRequestPortion": np.float32,
    "observations": int,
}

# Default average speeds in miles per hour for different transit modes.
# Used for imputing missing travel times in skims.
default_speed_mph = {
    "COM": 25.0,  # Commuter rail
    "HVY": 20.0,  # Heavy rail
    "LRF": 12.5,  # Light rail/Ferry
    "LOC": 12.5,  # Local bus
    "EXP": 17.0,  # Express bus
    "TRN": 15.0,  # Transit (general)
}
# Default fares in dollars for different transit modes.
# Used for imputing missing fare values in skims.
default_fare_dollars = {
    "COM": 10.0,
    "HVY": 4.0,
    "LRF": 2.5,
    "LOC": 2.5,
    "EXP": 4.0,
    "TRN": 15.0,
}


#########################
#### Common functions ###
#########################
def zone_order(settings: PilatesConfig, workspace: "Workspace") -> np.ndarray:
    """
    Retrieves the ordered list of zone keys from the canonical zones file.

    This function reads the authoritative `canonical_zones.csv` file to establish
    a consistent order for zones, which is crucial for creating ordered skim matrices.

    Args:
        settings (PilatesConfig): The current PILATES settings object,
            containing configuration for shared resources and geography.
        workspace (Workspace): The current workspace object, providing access to
            various data directories and state.

    Returns:
        np.ndarray: A one-dimensional NumPy array containing the ordered zone keys.
    """
    canonical_zones_df = get_canonical_zones(settings, workspace)
    order = canonical_zones_df["zone_key"].values
    return order


def read_skims(
    settings: PilatesConfig, mode: str = "a", data_dir: Optional[str] = None, file_name: str = "skims.omx"
) -> omx.File:
    """
    Opens an OpenMatrix (OMX) file for skims.

    This function provides a convenient way to open OMX skim files, handling
    default paths and different file access modes.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        mode (str): The file access mode.
            'r' for read-only;
            'w' to write (erases existing file);
            'a' to read/write an existing file (will create it if it doesn't exist).
        data_dir (Optional[str]): The directory where the OMX file is located.
            If None, it defaults to `settings.activitysim.local_mutable_data_folder`.
        file_name (str): The name of the OMX file. Defaults to "skims.omx".

    Returns:
        omx.File: An opened OpenMatrix file object.
    """
    if data_dir is None:
        data_dir = settings.activitysim.local_mutable_data_folder
    path = os.path.join(data_dir, file_name)
    skims = omx.open_file(path, mode=mode)
    return skims


# Mapping of PILATES setting parameter names to ActivitySim config parameter names.
# This allows dynamic updates to ActivitySim's settings.yaml.
asim_param_map = {"random_seed": "rng_base_seed"}
# Mapping of PILATES setting parameter names to their Pydantic paths within the PILATES config.
# Used to retrieve the current value of a parameter from the PILATES settings.
ASIM_PYDANTIC_PATH_MAP = {"random_seed": "activitysim.random_seed"}


def update_asim_config(
    settings: PilatesConfig, full_path: str, param: str, valueOverride: Optional[str] = None
) -> None:
    """
    Updates a specific parameter in the ActivitySim configuration file (`settings.yaml`).

    This function reads the ActivitySim `settings.yaml` file, locates the specified
    parameter, and updates its value. If the parameter is not found, it will be
    added to the end of the file.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        full_path (str): The full path to the ActivitySim configuration directory.
        param (str): The name of the parameter to update (e.g., "random_seed").
        valueOverride (Optional[str]): An optional value to override the setting with.
            If None, the value will be retrieved from the PILATES settings using
            `ASIM_PYDANTIC_PATH_MAP`.
    """
    config_header = asim_param_map[param]
    if valueOverride is None:
        # Get the Pydantic path for the parameter from the mapping
        pydantic_path = ASIM_PYDANTIC_PATH_MAP.get(param)
        if not pydantic_path:
            logger.warning(
                f"Parameter '{param}' has no defined Pydantic path. Cannot update asim config."
            )
            return
        # Retrieve the configuration value from PILATES settings
        config_value = get_setting(settings, pydantic_path)
    else:
        config_value = valueOverride

    # Do not update if the value from settings is None
    if config_value is None:
        logger.debug(
            f"Skipping asim config update for '{param}' because value is None."
        )
        return

    # Construct the full path to the ActivitySim settings.yaml file
    path_list = [
        full_path,
        get_setting(settings, "activitysim.local_mutable_configs_folder"),
        get_setting(settings, "activitysim.main_configs_dir", "configs"),
        "settings.yaml",
    ]

    asim_config_path = os.path.join(*path_list)
    modified = False
    # Read the existing config file
    with open(asim_config_path, "r") as file:
        data = file.readlines()
    # Write back the modified content
    with open(asim_config_path, "w") as file:
        for line in data:
            if config_header in line:
                if not modified:
                    # Preserve indentation
                    indent = line.split(config_header)[0]
                    file.writelines(
                        indent + config_header + ": " + str(config_value) + "\n"
                    )
                modified = True
            else:
                file.writelines(line)
        # If the parameter was not found, add it to the end of the file
        if not modified:
            file.writelines("\n")
            file.writelines(config_header + ": " + str(config_value) + "\n")


####################################
#### RAW BEAM SKIMS TO SKIMS.OMX ###
####################################
def read_skim(filename: str) -> pd.DataFrame:
    """
    Reads a single raw BEAM skim CSV file into a pandas DataFrame.

    Args:
        filename (str): The full path to the BEAM skim CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the raw BEAM skim data, with
            data types enforced by `beam_skims_types`.
    """
    logger.info("Loading raw beam skims from disk: {}".format(filename))
    df = pd.read_csv(filename, index_col=None, header=0, dtype=beam_skims_types)
    return df


def _load_raw_beam_skims(
    settings: PilatesConfig, convertFromCsv: bool = True, blank: bool = False
) -> Optional[pd.DataFrame]:
    """
    Reads raw BEAM skims from local storage.

    This function handles reading BEAM skims which can be in CSV format (single file
    or multiple files in a directory) or already in OMX format. It supports parallel
    reading for multiple CSV files.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        convertFromCsv (bool): If True, attempts to read from CSV files.
            If False, and the file is not OMX, it will return None.
        blank (bool): If True, immediately returns None, effectively skipping
            the loading process.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame containing the concatenated
            BEAM skims, or None if `blank` is True, or if the file is not found
            and `convertFromCsv` is False.
    """
    zone_type = settings.shared.geography.zones.zone_type
    skims_fname = settings.shared.skims.fname
    path_to_beam_skims = os.path.join(settings.beam.local_output_folder, skims_fname)

    # If the skims file does not exist and we are not converting from CSV, return None
    if (not os.path.exists(path_to_beam_skims)) and (not convertFromCsv):
        return None
    # If blank is True, skip loading and return None
    if blank:
        return None

    try:
        if ".csv" in path_to_beam_skims:
            # Read a single CSV file
            skims = read_skim(path_to_beam_skims)
        elif path_to_beam_skims.endswith(".omx"):
            # If it's already an OMX file, open it directly
            skims = omx.open_file(path_to_beam_skims, mode="a")
        else:  # Assume path is a folder with multiple CSV files
            all_files = glob.glob(path_to_beam_skims + "/*")
            # Use multiprocessing to read multiple CSVs in parallel
            agents = len(all_files)
            pool = Pool(processes=agents)
            result = pool.map(read_skim, all_files)
            # Concatenate all read DataFrames
            skims = pd.concat(result, axis=0, ignore_index=True)
    except KeyError:
        raise KeyError("Couldn't find input skims at {0}".format(path_to_beam_skims))
    return skims


def _load_raw_beam_origin_skims(settings: PilatesConfig) -> pd.DataFrame:
    """
    Reads raw BEAM origin skims (CSV format) from local storage.

    Args:
        settings (PilatesConfig): The current PILATES settings object.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the raw BEAM origin skim data,
            with data types enforced by `beam_origin_skims_types`.
    """

    origin_skims_fname = settings.shared.skims.origin_fname
    path_to_beam_skims = os.path.join(
        settings.beam.local_output_folder, origin_skims_fname
    )
    skims = pd.read_csv(path_to_beam_skims, dtype=beam_origin_skims_types)
    return skims


def _create_skim_object(
    settings: PilatesConfig, overwrite: bool = True, output_dir: Optional[str] = None
) -> Tuple[bool, bool, bool]:
    """
    Creates or manages the mutable OMX file for storing skim matrices.

    This function determines whether a new `skims.omx` file needs to be created
    or if an existing one should be used/overwritten. It's a preparatory step
    before populating the OMX file with skim data.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        overwrite (bool): If True, an existing `skims.omx` file will be removed
            and a new one will be prepared. Defaults to True.
        output_dir (Optional[str]): The directory where the `skims.omx` file
            should be located. If None, it defaults to
            `settings.activitysim.local_mutable_data_folder`.

    Returns:
        Tuple[bool, bool, bool]: A tuple containing three boolean values:
            - `new_file_created`: True if a new `skims.omx` file was created or
              an existing one was prepared for overwrite.
            - `should_use_csv_input_skims`: True if the system should proceed
              with processing CSV input skims.
            - `omx_file_was_created`: True if an empty OMX file was explicitly
              created in this call.
    """
    if output_dir is None:
        output_dir = settings.activitysim.local_mutable_data_folder
    skims_path = os.path.join(output_dir, "skims.omx")
    final_skims_exist = os.path.exists(skims_path)

    skims_fname = settings.shared.skims.fname
    omx_skim_output = skims_fname.endswith(".omx")
    # beam_output_dir = settings.beam.local_output_folder # This variable is not used
    mutable_skims_location = os.path.join(output_dir, "skims.omx")
    mutable_skims_exist = os.path.exists(mutable_skims_location)
    should_use_csv_input_skims = mutable_skims_exist and (not omx_skim_output)

    if final_skims_exist:
        if overwrite:
            logger.info("Found existing skims, removing.")
            os.remove(skims_path)
            # A new file will be created, so return True for new_file_created
            return True, should_use_csv_input_skims, False
        else:
            # If not overwriting, and mutable skims don't exist, copy the existing one
            if not mutable_skims_exist:
                shutil.copyfile(skims_path, mutable_skims_location)
            logger.info("Found existing skims, no need to re-create.")
            # No new file created, and no need to process CSVs if OMX already exists
            return False, False, False

    # If final skims don't exist or we are overwriting, and the output is expected to be OMX
    if ((not final_skims_exist) or overwrite) and omx_skim_output:
        if mutable_skims_exist:
            # Mutable skims exist, so we can use them, no new file created
            return True, should_use_csv_input_skims, False
        else:
            logger.info("Creating skims.omx")
            # Create an empty OMX file
            skims = omx.open_file(mutable_skims_location, "w")
            skims.close()
            # A new empty OMX file was created
            return True, should_use_csv_input_skims, True

    logger.exception("We should not be here: Unexpected state in _create_skim_object")
    return False, False, False


def _raw_beam_skims_preprocess(
    settings: PilatesConfig, year: int, skims_df: pd.DataFrame, workspace: "Workspace"
) -> pd.DataFrame:
    """
    Validates and preprocesses raw BEAM skims.

    This function performs several validation checks on the raw BEAM skims,
    such as ensuring all origin and destination zones are present in the
    canonical zone order. It then preprocesses the skims by setting a
    multi-index, converting distances to miles, and handling infinite or
    zero values.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        year (int): The simulation year.
        skims_df (pd.DataFrame): A DataFrame containing the raw BEAM skims.
        workspace (Workspace): The current workspace object.

    Returns:
        pd.DataFrame: The preprocessed and validated skims DataFrame.

    Raises:
        AssertionError: If there are missing origin or destination zone IDs
            in the BEAM skims compared to the canonical zones.
        ValueError: If origin-destination distances contain infinite or zero values
            after preprocessing.
    """
    # Validations: Ensure all origins and destinations in the skims are part of the canonical zones.
    origin_taz = skims_df.origin.unique()
    destination_taz = skims_df.destination.unique()
    assert len(origin_taz) == len(destination_taz), "Number of unique origins and destinations must be equal."

    order = zone_order(settings, workspace)

    test_1 = set(origin_taz).issubset(set(order))
    test_2 = set(destination_taz).issubset(set(order))
    test_3 = len(set(order) - set(origin_taz))
    test_4 = len(set(order) - set(destination_taz))
    assert test_1, f"There are {test_3} missing origin zone ids in BEAM skims"
    assert test_2, f"There are {test_4} missing destination zone ids in BEAM skims"

    # Preprocess skims:
    # Set a multi-index for easier slicing and aggregation.
    skims_df.set_index(
        ["timePeriod", "pathType", "origin", "destination"], inplace=True
    )
    # Convert distances from meters to miles.
    skims_df.loc[:, "DIST_miles"] = skims_df["DIST_meters"] * (0.621371 / 1000)
    skims_df.loc[:, "DDIST_miles"] = skims_df["DDIST_meters"] * (0.621371 / 1000)
    # Replace infinite values and zeros with NaN for consistent handling.
    skims_df.replace({np.inf: np.nan, 0: np.nan}, inplace=True)  # TEMPORARY FIX

    # Check for remaining infinite or zero values in DDIST_miles after replacement.
    inf = np.isinf(skims_df["DDIST_miles"]).values.sum() > 0
    zeros = (skims_df["DDIST_miles"] == 0).sum() > 0
    if inf or zeros:
        raise ValueError("Origin-Destination distances contains inf or zero values.")

    return skims_df


def _raw_beam_origin_skims_preprocess(
    settings: PilatesConfig, year: int, origin_skims_df: pd.DataFrame, workspace: "Workspace"
) -> pd.DataFrame:
    """
    Validates and preprocesses raw BEAM origin skims.

    This function checks if all origin zone IDs in the raw BEAM origin skims
    are present in the canonical zone order. It then filters the DataFrame
    to include only relevant origins and sets a multi-index.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        year (int): The simulation year.
        origin_skims_df (pd.DataFrame): A DataFrame containing the raw BEAM origin skims.
        workspace (Workspace): The current workspace object.

    Returns:
        pd.DataFrame: The preprocessed and validated origin skims DataFrame.

    Raises:
        AssertionError: If there are missing origin zone IDs in the BEAM skims
            compared to the canonical zones.
    """
    # Validations: Ensure all origins in the skims are part of the canonical zones.
    origin_taz = origin_skims_df.origin.unique()

    order = zone_order(settings, workspace)

    # Check for missing origin zone IDs
    test_3 = len(set(order) - set(origin_taz))
    assert test_3 == 0, f"There are {test_3} missing origin zone ids in BEAM skims"

    # Filter the DataFrame to include only origins present in the canonical order
    # and set a multi-index for consistency.
    return origin_skims_df.loc[origin_skims_df["origin"].isin(order)].set_index(
        ["timePeriod", "reservationType", "origin"]
    )


def _create_skims_by_mode(
    settings: PilatesConfig, skims_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the preprocessed BEAM skims DataFrame into separate DataFrames for
    auto and transit modes.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        skims_df (pd.DataFrame): A DataFrame containing the preprocessed BEAM skims,
            expected to have a multi-index including 'pathType'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames:
            - `auto_df`: Skims filtered for highway (auto) path types.
            - `transit_df`: Skims filtered for transit path types.

    Raises:
        AssertionError: If no auto skims or no transit skims are found after splitting.
    """
    logger.info("Splitting BEAM skims by mode.")

    # Retrieve highway and transit path types from settings
    hwy_paths = settings.beam.simulated_hwy_paths
    transit_paths = settings.shared.skims.transit_paths.keys()

    logger.info("Splitting out auto skims.")
    # Use pd.IndexSlice for efficient multi-index slicing
    auto_df = skims_df.loc[pd.IndexSlice[:, hwy_paths, :, :], :]
    assert len(auto_df) > 0, "No auto skims found after splitting by highway paths."

    logger.info("Splitting out transit skims.")
    transit_df = skims_df.loc[pd.IndexSlice[:, transit_paths, :, :], :]
    assert len(transit_df) > 0, "No transit skims found after splitting by transit paths."

    # Delete the original skims_df to free up memory
    del skims_df
    return auto_df, transit_df


def _build_square_matrix(
    series: pd.Series, num_taz: int, source: str = "origin", fill_na: float = 0
) -> np.ndarray:
    """
    Builds a square (num_taz x num_taz) NumPy matrix from a pandas Series.

    This function is used to expand a 1-dimensional series (e.g., representing
    origin-based or destination-based values) into a full origin-destination
    matrix.

    Args:
        series (pd.Series): The pandas Series containing the values to be
            transformed into a square matrix.
        num_taz (int): The total number of Transportation Analysis Zones (TAZ).
            This determines the dimensions of the output square matrix.
        source (str): Specifies whether the series represents 'origin' or
            'destination' based values. This affects how the series is tiled.
            Defaults to "origin".
        fill_na (float): The value to use for filling NaN (Not a Number) values
            in the input series before tiling. Defaults to 0.

    Returns:
        np.ndarray: A square NumPy array (num_taz x num_taz) representing the
            origin-destination matrix.

    Raises:
        logger.error: If an unrecognized `source` value is provided.
    """
    # Tile the series values to create a num_taz x num_taz matrix.
    # fillna(fill_na) ensures that any NaN values in the series are replaced before tiling.
    out = np.tile(series.fillna(fill_na).values, (num_taz, 1))
    if source == "origin":
        # If the series represents origin-based values, transpose to get O-D format.
        return out.transpose()
    elif source == "destination":
        # If the series represents destination-based values, no transpose needed.
        return out
    else:
        # Log an error for invalid source types.
        logger.error(
            "1-d skims must be associated with either 'origin' or 'destination'"
        )
        # Return an empty matrix or raise an error, depending on desired behavior.
        # For now, returning an empty matrix to avoid crashing.
        return np.zeros((num_taz, num_taz), dtype=np.float32)


def _build_od_matrix(
    df: pd.DataFrame, metric: str, order: np.ndarray, fill_na: float = 0.0
) -> Tuple[np.ndarray, bool]:
    """
    Transforms skims from a pandas DataFrame into a NumPy square matrix (O-D matrix format).

    This function takes a DataFrame of skims, pivots it based on origin and
    destination, and converts it into a dense square matrix according to a
    specified zone order. It also handles missing values and infinite values.

    Args:
        df (pd.DataFrame): A DataFrame containing clean skims, expected to have
            'origin' and 'destination' in its index or columns.
        metric (str): The name of the metric column in the DataFrame to use
            for generating the skim matrix (e.g., "SOV_TIME").
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones for both rows and columns of the O-D matrix. The values in
            `order` should match the zone identifiers in the DataFrame.
        fill_na (float): The value to fill missing (NaN) entries in the
            resulting O-D matrix. Defaults to 0.0.

    Returns:
        Tuple[np.ndarray, bool]: A tuple containing:
            - np.ndarray: A square NumPy array representing the O-D matrix.
            - bool: `useDefaults`, which is True if the `metric` column was not
              found in the input DataFrame, indicating that default values might
              have been used.

    Raises:
        AssertionError: If the resulting matrix's index or columns do not
            match the specified `order`, or if the matrix is not square.
        ValueError: If origin-destination distances contain inf or zero values.
    """
    # Initialize an empty DataFrame with the correct order for index (origin) and columns (destination).
    out = pd.DataFrame(
        np.nan, index=order, columns=order, dtype=np.float32
    ).rename_axis(index="origin", columns="destination")

    useDefaults = True # Assume defaults are used unless metric is found and processed

    if metric in df.columns:
        # Pivot the DataFrame to get a matrix-like structure with origin as index and destination as columns.
        pivot = df[metric].unstack()
        # Fill the pre-initialized 'out' DataFrame with values from the pivoted DataFrame.
        # This handles cases where not all OD pairs are present in the input 'df'.
        out.loc[pivot.index, pivot.columns] = pivot.fillna(np.nan)
        useDefaults = False
        # Check for and handle infinite values, replacing them with fill_na.
        infs = np.isinf(out)
        if np.any(infs):
            logger.warning(
                f"Replacing {infs.sum().sum()} infs in skim {metric}"
            )
            out[infs] = fill_na
    else:
        # If the metric is not found in the DataFrame, useDefaults remains True.
        logger.warning(f"Metric '{metric}' not found in DataFrame columns. Using default values.")

    num_zones = len(order)

    # Assertions to ensure the resulting matrix is consistent with the expected zone order and is square.
    assert out.index.isin(order).all(), "There are missing origins in the final OD matrix."
    assert out.columns.isin(order).all(), "There are missing destinations in the final OD matrix."
    assert (
        num_zones,
        num_zones,
    ) == out.shape, "Origin-Destination matrix is not square."

    # Fill any remaining NaN values with the specified fill_na value and return the NumPy array.
    return out.fillna(fill_na).values, useDefaults


def impute_distances(
    zones: Union[pd.DataFrame, gpd.GeoDataFrame],
    origin: Optional[Union[List[int], np.ndarray]] = None,
    destination: Optional[Union[List[int], np.ndarray]] = None,
) -> np.ndarray:
    """
    Imputes distances in miles for missing Origin-Destination (OD) pairs by
    calculating the Cartesian distance between zone centroids.

    If `origin` and `destination` are not provided, it calculates a full
    OD matrix of distances between all zone centroids. If `origin` and
    `destination` lists are provided, it calculates distances only for
    those specific OD pairs.

    Args:
        zones (Union[pd.DataFrame, gpd.GeoDataFrame]): A DataFrame containing
            zone information. If a pandas DataFrame, it must have a 'geometry'
            column where `wkt.loads` can be applied. If a GeoDataFrame, it
            should already have valid geometries.
        origin (Optional[Union[List[int], np.ndarray]]): An optional list or
            array-like of origin zone identifiers. If provided, `destination`
            must also be provided and have the same length.
        destination (Optional[Union[List[int], np.ndarray]]): An optional list
            or array-like of destination zone identifiers. If provided, `origin`
            must also be provided and have the same length.

    Returns:
        np.ndarray: A NumPy array of imputed distances in miles.
            - If `origin` and `destination` are None, returns a square matrix
              of shape (num_zones, num_zones).
            - If `origin` and `destination` are provided, returns a 1D array
              of shape (len(origin),).

    Raises:
        AssertionError: If `zones` is not a GeoPandas DataFrame after conversion,
            or if `origin` and `destination` have different lengths when both are provided.
    """
    # Convert pandas DataFrame to GeoDataFrame if necessary, ensuring geometry column is properly parsed.
    if isinstance(zones, pd.core.frame.DataFrame):
        zones.geometry = zones.geometry.astype(str).apply(wkt.loads)
        zones = gpd.GeoDataFrame(zones, geometry="geometry", crs="EPSG:4326")

    assert isinstance(
        zones, gpd.geodataframe.GeoDataFrame
    ), "Input 'zones' must be a GeoPandas dataframe or convertible to one."

    gdf = zones.copy()

    # Transform GeoDataFrame to a CRS that uses meters for accurate distance calculation.
    gdf = gdf.to_crs("EPSG:3857")

    if (origin is None) and (destination is None):
        # Calculate full OD matrix of distances between all zone centroids.
        gdf = gdf.reset_index(drop=True).geometry.centroid
        x, y = gdf.geometry.x.values, gdf.geometry.y.values
        # Calculate Euclidean distance between all pairs of centroids.
        distInMeters = np.linalg.norm(
            [x[:, None] - x[None, :], y[:, None] - y[None, :]], axis=0
        )
        # Fill diagonal (self-distances) with a small non-zero value (e.g., 400 meters).
        np.fill_diagonal(distInMeters, 400)
        # Convert meters to miles.
        return distInMeters * (0.621371 / 1000)

    else:
        # Calculate distances for specific OD pairs.
        assert len(origin) == len(
            destination
        ), 'Parameters "origin" and "destination" must have the same length.'
        # Adjust origin/destination IDs (assuming 0-indexed input, 1-indexed zones).
        origin_idx = (origin + 1).astype(str)
        destination_idx = (destination + 1).astype(str)

        # Select centroids for the specified origin and destination pairs.
        orig_centroids = gdf.loc[origin_idx].reset_index(drop=True).geometry.centroid
        dest_centroids = gdf.loc[destination_idx].reset_index(drop=True).geometry.centroid

        # Calculate distances between selected origin and destination centroids.
        # Replace any zero distances with a small value (e.g., 100 meters) to avoid issues.
        # Convert meters to miles.
        return orig_centroids.distance(dest_centroids).replace({0: 100}).values * (0.621371 / 1000)


def _distance_skims(
    settings: PilatesConfig,
    year: int,
    input_skims: Optional[Union[pd.DataFrame, omx.File]],
    order: np.ndarray,
    data_dir: str,
    workspace: "Workspace",
) -> None:
    """
    Generates and imputes distance matrices for drive, walk, and bike modes
    and stores them in the OMX skims file.

    This function is responsible for populating the OMX skims file with
    distance-related measures. It can either use existing distance data
    from `input_skims` or impute missing distances based on zone centroids.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        year (int): The simulation year.
        input_skims (Optional[Union[pd.DataFrame, omx.File]]): An optional
            DataFrame or OMX file containing initial skim data. If a DataFrame,
            it's expected to contain a 'DIST' column. If an OMX file, it's
            checked for a 'DIST' matrix.
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones for the skim matrices.
        data_dir (str): The directory where the OMX skims file is located.
        workspace (Workspace): The current workspace object.
    """
    logger.info("Creating distance skims.")

    skims_fname = "skims.omx"
    mutable_skims_location = os.path.join(data_dir, skims_fname)
    needToClose = True # Flag to determine if the OMX file needs to be closed at the end

    # If input_skims is already an open OMX file, use it directly; otherwise, open a new one.
    if input_skims is not None:
        output_skims = input_skims
        needToClose = False
    else:
        output_skims = omx.open_file(mutable_skims_location, mode="a")

    # Determine the distance column from settings.
    dist_column = settings.beam.asim_hwy_measure_map["DIST"]
    mx_dist = None # Initialize mx_dist

    if isinstance(input_skims, pd.DataFrame):
        # If input is a DataFrame, extract distance and build OD matrix.
        dist_df = input_skims[[dist_column]].groupby(level=[2, 3]).agg("first")
        mx_dist, useDefaults = _build_od_matrix(
            dist_df, dist_column, order, fill_na=np.nan
        )
        if useDefaults:
            logger.warning(
                f"Filling in default skim values for measure {dist_column} because they're not in BEAM outputs"
            )
    elif isinstance(input_skims, omx.File):
        # If input is an OMX file, try to read 'DIST' matrix.
        if "DIST" in input_skims.list_matrices():
            mx_dist = np.array(input_skims["DIST"], dtype=np.float32)
        else:
            # If 'DIST' not found, initialize with NaNs.
            mx_dist = np.full((len(order), len(order)), np.nan, dtype=np.float32)
    else:
        # If no input skims or unrecognized type, initialize with NaNs.
        mx_dist = np.full((len(order), len(order)), np.nan, dtype=np.float32)

    # Impute missing distances
    missing = np.isnan(mx_dist)
    if missing.all():
        # If all distances are missing, impute all.
        logger.info("Imputing all missing distance skims.")
        zones = load_canonical_zones(settings, workspace)
        mx_dist = impute_distances(zones)
        output_skims["DIST"] = mx_dist
    elif missing.any():
        # If some distances are missing, impute only those.
        orig, dest = np.where(missing == True)
        logger.info(f"Imputing {len(orig)} missing distance skims.")
        zones = load_canonical_zones(settings, workspace)
        imputed_dist = impute_distances(zones, orig, dest)
        mx_dist[orig, dest] = imputed_dist
        output_skims["DIST"] = mx_dist
    else:
        # No missing distances, use existing.
        logger.info("No need to impute missing distance skims.")
        # Ensure 'DIST' is written to output_skims if it wasn't already.
        if "DIST" not in output_skims.list_matrices():
            output_skims["DIST"] = mx_dist

    # Create distance matrices for bike and walk, currently mirroring drive distances.
    # Also initialize bike and walk trip matrices to zeros.
    if "DISTBIKE" not in output_skims.list_matrices():
        output_skims["DISTBIKE"] = mx_dist
    if "DISTWALK" not in output_skims.list_matrices():
        output_skims["DISTWALK"] = mx_dist
    if "BIKE_TRIPS" not in output_skims.list_matrices():
        output_skims["BIKE_TRIPS"] = np.zeros_like(mx_dist)
    if "WALK_TRIPS" not in output_skims.list_matrices():
        output_skims["WALK_TRIPS"] = np.zeros_like(mx_dist)

    # Close the OMX file if it was opened within this function.
    if needToClose:
        output_skims.close()


def _build_od_matrix_parallel(
    tup: Tuple[pd.DataFrame, Dict[str, str], int, np.ndarray, float]
) -> Dict[str, np.ndarray]:
    """
    Helper function to build Origin-Destination (OD) matrices in parallel.

    This function is designed to be used with `multiprocessing.Pool.map` to
    efficiently create multiple OD matrices for different measures.

    Args:
        tup (Tuple[pd.DataFrame, Dict[str, str], int, np.ndarray, float]): A tuple
            containing the arguments for `_build_od_matrix`:
            - `df` (pd.DataFrame): The input DataFrame of skims for a specific group.
            - `measure_map` (Dict[str, str]): A dictionary mapping generic measure
              names to their corresponding column names in the DataFrame.
            - `num_taz` (int): The total number of TAZs.
            - `order` (np.ndarray): The desired order of zones.
            - `fill_na` (float): The value to fill NaN entries with.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are measure names and
            values are the corresponding NumPy OD matrices.
    """
    df, measure_map, num_taz, order, fill_na = tup
    out = dict()
    for measure in measure_map.keys():
        # If the DataFrame for this group is empty, create a zero matrix.
        if len(df.index) == 0:
            mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
            # Mark as using defaults since no data was available.
            useDefaults = True
        # Handle specific measures like 'FAR' (fare) or 'BOARDS' (boardings)
        elif (measure == "FAR") or (measure == "BOARDS"):
            mtx, useDefaults = _build_od_matrix(
                df, measure_map[measure], order, fill_na
            )
        # If the measure's column exists in the DataFrame, build the OD matrix.
        elif measure_map[measure] in df.columns:
            # Note: ActivitySim models often expect transit skim values to be scaled (e.g., x100).
            # This comment indicates that if skims are not from Cube, a scaling factor might be needed.
            # The actual scaling (multiplication by 100) happens in _transit_skims.
            mtx, useDefaults = _build_od_matrix(
                df, measure_map[measure], order, fill_na
            )
        else:
            # If the measure is not found, create a zero matrix and mark as using defaults.
            mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
            useDefaults = True # Explicitly set to True here for clarity
        out[measure] = mtx
    return out


def _transit_skims(
    settings: PilatesConfig,
    transit_df: pd.DataFrame,
    order: np.ndarray,
    data_dir: Optional[str] = None,
) -> None:
    """
    Generates and populates transit OMX skims from a preprocessed transit DataFrame.

    This function takes a DataFrame containing transit skim data, groups it by
    time period and path type, and then uses parallel processing to build
    Origin-Destination (OD) matrices for various transit measures. These matrices
    are then written to the OMX skims file.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        transit_df (pd.DataFrame): A DataFrame containing preprocessed transit skims,
            expected to have a multi-index including 'timePeriod' and 'pathType'.
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones for the skim matrices.
        data_dir (Optional[str]): The directory where the OMX skims file is located.
            If None, it defaults to `settings.activitysim.local_mutable_data_folder`.
    """

    logger.info("Creating transit skims.")
    transit_paths = settings.shared.skims.transit_paths
    periods = settings.shared.skims.periods
    measure_map = settings.beam.asim_transit_measure_map
    skims = read_skims(settings, mode="a", data_dir=data_dir)
    num_taz = len(order)
    fill_na = 0.0

    # Group the transit DataFrame by time period and path type for parallel processing.
    groupBy = transit_df.groupby(level=[0, 1])

    # Use a multiprocessing Pool to build OD matrices in parallel.
    with Pool(cpu_count() - 1) as p:
        ret_list = p.map(
            _build_od_matrix_parallel,
            [
                (group.loc[name], measure_map, num_taz, order, fill_na)
                for name, group in groupBy
            ],
        )

    resultsDict = dict()
    # Consolidate results from parallel processing into a single dictionary.
    for (period, path), processedDict in zip(groupBy.groups.keys(), ret_list):
        for measure, mtx in processedDict.items():
            name = f"{path}_{measure}__{period}"
            resultsDict[name] = mtx

    # Iterate through all expected transit paths, periods, and measures to populate the OMX file.
    for path, measures in transit_paths.items():
        for period in periods:
            for measure in measures:
                name = f"{path}_{measure}__{period}"
                if name in resultsDict:
                    mtx = resultsDict[name]
                else:
                    # If a measure is missing, log a warning and create a zero matrix.
                    logger.warning(
                        f"Filling in default skim values for measure {name} because they're not in BEAM outputs"
                    )
                    mtx = np.zeros((num_taz, num_taz), dtype=np.float32)

                # Handle infinite values by replacing them with NaN.
                if np.any(np.isinf(mtx)):
                    logger.warning(
                        f"Replacing {np.isinf(mtx).sum().sum()} infs in skim {name}"
                    )
                    mtx[np.isinf(mtx)] = np.nan

                # Apply scaling for travel times (multiply by 100) as ActivitySim expects.
                if (measure == "FAR") or (measure == "BOARDS"):
                    skims[name] = mtx
                else:
                    skims[name] = mtx * 100
    skims.close()


def _ridehail_skims(
    settings: PilatesConfig,
    ridehail_df: pd.DataFrame,
    order: np.ndarray,
    data_dir: Optional[str] = None,
) -> None:
    """
    Generates and populates ridehail OMX skims from a preprocessed ridehail DataFrame.

    This function processes ridehail skim data, iterating through defined
    ridehail paths and periods to create OD matrices for various measures
    like rejection probability and wait times. These matrices are then
    written to the OMX skims file.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        ridehail_df (pd.DataFrame): A DataFrame containing preprocessed ridehail skims.
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones for the skim matrices.
        data_dir (Optional[str]): The directory where the OMX skims file is located.
            If None, it defaults to `settings.activitysim.local_mutable_data_folder`.
    """
    ridehail_path_map = settings.beam.ridehail_path_map
    periods = settings.shared.skims.periods
    measure_map = settings.beam.asim_ridehail_measure_map
    skims = read_skims(settings, mode="a", data_dir=data_dir)
    num_taz = len(order)
    df = ridehail_df.copy() # Create a copy to avoid modifying the original DataFrame

    # Iterate through each defined ridehail path and time period.
    for path, skimPath in ridehail_path_map.items():
        for period in periods:
            # Select data for the current period and path, reindexing by zone order.
            df_ = df.loc[(period, skimPath), :].loc[order, :]
            for measure, skimMeasure in measure_map.items():
                name = f"{path}_{measure}__{period}"
                if measure == "REJECTIONPROB":
                    # For rejection probability, build a square matrix from the series.
                    mtx = _build_square_matrix(df_[skimMeasure], num_taz, "origin", 0.0)
                elif measure_map[measure] in df_.columns:
                    # For other measures present in the DataFrame, build a square matrix.
                    # The comment about scaling for transit skims from Cube is noted,
                    # but explicitly states it might not apply to wait time for ridehail.
                    mtx = _build_square_matrix(df_[skimMeasure], num_taz, "origin", 0.0)
                else:
                    # If the measure is not found, create a zero matrix.
                    mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
                # Store the resulting matrix in the OMX skims file.
                skims[name] = mtx
    skims.close()
    # Delete DataFrames to free up memory.
    del df, df_


def _get_field_or_else_empty(
    skims: Optional[omx.File], field: str, num_taz: int
) -> Tuple[np.ndarray, bool]:
    """
    Retrieves a specific matrix from an OMX skims file or returns an empty
    (NaN-filled) matrix if the field does not exist or the skims object is None.

    Args:
        skims (Optional[omx.File]): An opened OpenMatrix file object, or None.
        field (str): The name of the matrix (field) to retrieve from the OMX file.
        num_taz (int): The total number of TAZs, used to create an empty matrix
            if the field is not found.

    Returns:
        Tuple[np.ndarray, bool]: A tuple containing:
            - np.ndarray: The retrieved matrix (as float32) or an NaN-filled
              matrix of shape (num_taz, num_taz).
            - bool: True if the field was not found and an empty matrix was created,
              False otherwise.
    """
    if skims is None:
        return np.full((num_taz, num_taz), np.nan, dtype=np.float32), True
    if field in skims.list_matrices():
        return np.array(skims[field], dtype=np.float32), False
    else:
        return np.full((num_taz, num_taz), np.nan, dtype=np.float32), True


def _fill_ridehail_skims(
    settings: PilatesConfig,
    input_skims: Optional[omx.File],
    order: np.ndarray,
    data_dir: str,
) -> None:
    """
    Fills in missing ridehail skim matrices in the OMX file, potentially using
    default values for missing data.

    This function iterates through defined ridehail paths, periods, and measures.
    If a skim matrix is missing in the output OMX file, it attempts to retrieve
    it from `input_skims` and fills any remaining NaN values with predefined
    defaults (e.g., for wait time or rejection probability).

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        input_skims (Optional[omx.File]): An optional OMX file containing
            ridehail skim data to be merged.
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones for the skim matrices.
        data_dir (str): The directory where the OMX skims file is located.
    """
    logger.info("Merging ridehail omx skims.")

    ridehail_path_map = settings.beam.ridehail_path_map
    periods = settings.shared.skims.periods
    measure_map = settings.beam.asim_ridehail_measure_map

    num_taz = len(order)

    skims_fname = "skims.omx"
    # The beam_output_dir variable is commented out as it's not directly used here.
    # beam_output_dir = get_setting(settings, "beam.local_output_folder")
    mutable_skims_location = os.path.join(data_dir, skims_fname)
    needToClose = True # Flag to determine if the OMX file needs to be closed at the end

    # If input_skims is already an open OMX file, use it directly; otherwise, open a new one.
    if input_skims is not None:
        output_skims = input_skims
        needToClose = False
    else:
        output_skims = omx.open_file(mutable_skims_location, mode="a")

    output_skim_tables = output_skims.list_matrices()

    # NOTE: time is in units of minutes for ActivitySim skims.
    for path, skimPath in ridehail_path_map.items():
        logger.info(f"Writing tables for path type {path}")
        for period in periods:
            completed_measure = f"{path}_TRIPS__{period}"
            failed_measure = f"{path}_FAILURES__{period}"

            # Get existing or empty matrices for TRIPS and FAILURES.
            temp_completed, createdTemp = _get_field_or_else_empty(
                output_skims, completed_measure, num_taz
            )
            temp_failed, createdTemp = _get_field_or_else_empty(
                output_skims, failed_measure, num_taz
            )

            if createdTemp:
                # If created, initialize with nan_to_num and set attributes.
                output_skims[completed_measure] = np.nan_to_num(
                    np.array(temp_completed)
                )
                output_skims[failed_measure] = np.nan_to_num(np.array(temp_failed))
                output_skims[completed_measure].attrs.mode = path
                output_skims[failed_measure].attrs.mode = path
                output_skims[completed_measure].attrs.measure = "TRIPS"
                output_skims[failed_measure].attrs.measure = "FAILURES"
                output_skims[completed_measure].attrs.timePeriod = period
                output_skims[failed_measure].attrs.timePeriod = period

            for measure in measure_map.keys():
                name = f"{path}_{measure}__{period}"
                inOutputSkims = name in output_skim_tables
                if not inOutputSkims:
                    # If the skim is not already in the output, try to get it from input_skims.
                    temp, createdTemp = _get_field_or_else_empty(
                        input_skims, name, num_taz
                    )
                    missing_values = np.isnan(temp)
                    if measure == "WAIT":
                        # Default wait time for missing values.
                        temp[missing_values] = 6.0
                    elif measure == "REJECTIONPROB":
                        # Default rejection probability for missing values.
                        temp[missing_values] = 0.2

                    if not inOutputSkims:
                        # Add the new skim to the output OMX file and set attributes.
                        output_skims[name] = temp
                        output_skims[name].attrs.mode = path
                        output_skims[name].attrs.measure = measure
                        output_skims[name].attrs.timePeriod = period
                    else:
                        # If it exists but had missing values, update them.
                        if np.any(missing_values):
                            output_skims[name][:] = temp
    # Close the OMX file if it was opened within this function.
    if needToClose:
        output_skims.close()


def _fill_transit_skims(
    settings: PilatesConfig,
    input_skims: Optional[omx.File],
    order: np.ndarray,
    data_dir: str,
) -> None:
    """
    Fills in missing transit skim matrices in the OMX file, potentially using
    default values for missing data.

    This function iterates through defined transit paths, periods, and measures.
    If a skim matrix is missing in the output OMX file, it attempts to retrieve
    it from `input_skims` and fills any remaining NaN values with predefined
    defaults based on the measure type (e.g., travel times, fares, boards).

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        input_skims (Optional[omx.File]): An optional OMX file containing
            transit skim data to be merged.
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones for the skim matrices.
        data_dir (str): The directory where the OMX skims file is located.
    """
    logger.info("Merging transit omx skims.")

    transit_paths = settings.shared.skims.transit_paths
    periods = settings.shared.skims.periods
    measure_map = settings.beam.asim_transit_measure_map

    num_taz = len(order)

    skims_fname = "skims.omx"
    # The beam_output_dir variable is commented out as it's not directly used here.
    # beam_output_dir = get_setting(settings, "beam.local_output_folder")
    mutable_skims_location = os.path.join(data_dir, skims_fname)
    needToClose = True # Flag to determine if the OMX file needs to be closed at the end

    # If input_skims is already an open OMX file, use it directly; otherwise, open a new one.
    if input_skims is not None:
        output_skims = input_skims
        needToClose = False
    else:
        output_skims = omx.open_file(mutable_skims_location, mode="a")

    # Get the distance matrix, which might be used for imputing missing transit times.
    distance_miles = np.array(output_skims["DIST"], dtype=np.float32)
    output_skim_tables = output_skims.list_matrices()

    # NOTE: time is in units of minutes for ActivitySim skims.

    for path, measures in transit_paths.items():
        logger.info(f"Writing tables for path type {path}")
        for period in periods:
            completed_measure = f"{path}_TRIPS__{period}"
            failed_measure = f"{path}_FAILURES__{period}"

            # Get existing or empty matrices for TRIPS and FAILURES.
            temp_completed, createdTemp = _get_field_or_else_empty(
                output_skims, completed_measure, num_taz
            )
            temp_failed, createdTemp = _get_field_or_else_empty(
                output_skims, failed_measure, num_taz
            )

            if createdTemp:
                # If created, initialize with nan_to_num and set attributes.
                output_skims[completed_measure] = np.nan_to_num(
                    np.array(temp_completed)
                )
                output_skims[failed_measure] = np.nan_to_num(np.array(temp_failed))
                output_skims[completed_measure].attrs.mode = path
                output_skims[failed_measure].attrs.mode = path
                output_skims[completed_measure].attrs.measure = "TRIPS"
                output_skims[failed_measure].attrs.measure = "FAILURES"
                output_skims[completed_measure].attrs.timePeriod = period
                output_skims[failed_measure].attrs.timePeriod = period

            # Iterate through each measure for the current path and period.
            for measure in measures:
                name = f"{path}_{measure}__{period}"
                inOutputSkims = name in output_skim_tables
                if not inOutputSkims:
                    # If the skim is not already in the output, try to get it from input_skims.
                    temp, createdTemp = _get_field_or_else_empty(
                        input_skims, name, num_taz
                    )
                    missing_values = np.isnan(temp)
                    if np.any(missing_values):
                        # Impute missing values based on the measure type.
                        if (
                            (measure == "IVT")
                            or (measure == "TOTIVT")
                            or (measure == "KEYIVT")
                        ):
                            # Impute in-vehicle time based on distance and default speed, scaled by 100.
                            temp[missing_values] = (
                                distance_miles[missing_values]
                                / default_speed_mph.get(path.split("_")[1], 10.0)
                                * 60
                                * 100
                            )
                        elif measure == "DIST":
                            # Impute distance with the general distance matrix.
                            temp[missing_values] = distance_miles[missing_values]
                        elif (
                            (measure == "WAIT")
                            or (measure == "WACC")
                            or (measure == "WEGR")
                            or (measure == "IWAIT")
                        ):
                            # Impute various wait times with a default high value.
                            temp[missing_values] = 1000.0
                        elif measure == "FAR":
                            # Impute fare based on default fare dollars for the mode.
                            temp[missing_values] = default_fare_dollars.get(
                                path.split("_")[1], 4.0
                            )
                        elif measure == "BOARDS":
                            # Impute boardings with a default value.
                            temp[missing_values] = 2.0
                        elif (measure == "DDIST") and (
                            path.endswith("DRV") or path.startswith("DRV")
                        ):
                            # Impute drive distance with a default value.
                            temp[missing_values] = 2.0
                        elif (measure == "DTIME") and (
                            path.endswith("DRV") or path.startswith("DRV")
                        ):
                            # Impute drive time with a default value.
                            temp[missing_values] = 1500.0
                        else:
                            # Default for any other missing measure.
                            temp[missing_values] = 0.0

                    # Add the new skim to the output OMX file and set attributes.
                    output_skims[name] = temp
                    output_skims[name].attrs.mode = path
                    output_skims[name].attrs.measure = measure
                    output_skims[name].attrs.timePeriod = period

    # Close the OMX file if it was opened within this function.
    if needToClose:
        output_skims.close()


def _fill_auto_skims(
    settings: PilatesConfig,
    input_skims: Optional[omx.File],
    order: np.ndarray,
    data_dir: Optional[str] = None,
) -> None:
    """
    Fills in missing auto (drive) skim matrices in the OMX file, potentially using
    default values for missing data.

    This function iterates through defined highway paths, periods, and measures.
    If a skim matrix is missing in the output OMX file, it attempts to retrieve
    it from `input_skims` and fills any remaining NaN values with predefined
    defaults based on the measure type (e.g., travel times, distances, tolls).

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        input_skims (Optional[omx.File]): An optional OMX file containing
            auto skim data to be merged.
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones for the skim matrices.
        data_dir (Optional[str]): The directory where the OMX skims file is located.
            If None, it defaults to `settings.activitysim.local_mutable_data_folder`.
    """
    logger.info("Merging drive omx skims.")

    # Retrieve periods, highway paths, and measure map from settings.
    periods = settings.shared.skims.periods
    paths = settings.shared.skims.hwy_paths
    measure_map = settings.beam.asim_hwy_measure_map

    num_taz = len(order)

    skims_fname = "skims.omx"
    # The beam_output_dir variable is commented out as it's not directly used here.
    # beam_output_dir = get_setting(settings, "beam.local_output_folder")
    mutable_skims_location = os.path.join(data_dir, skims_fname)
    needToClose = True # Flag to determine if the OMX file needs to be closed at the end

    # If input_skims is already an open OMX file, use it directly; otherwise, open a new one.
    if input_skims is not None:
        output_skims = input_skims
        needToClose = False
    else:
        output_skims = omx.open_file(mutable_skims_location, mode="a")

    # Get the distance matrix, which might be used for imputing missing auto times.
    distance_miles = np.array(output_skims["DIST"])
    output_skim_tables = output_skims.list_matrices()
    nSkimsCreated = 0 # Counter for newly created skims.

    # NOTE: time is in units of minutes for ActivitySim skims.

    for path in paths:
        logger.info(f"Writing tables for path type {path}")
        for period in periods:
            completed_measure = f"{path}_TRIPS__{period}"
            failed_measure = f"{path}_FAILURES__{period}"

            # Get existing or empty matrices for TRIPS and FAILURES.
            temp_completed, createdTemp = _get_field_or_else_empty(
                output_skims, completed_measure, num_taz
            )
            temp_failed, createdTemp = _get_field_or_else_empty(
                output_skims, failed_measure, num_taz
            )

            if createdTemp:
                nSkimsCreated += 1
                # If created, initialize with nan_to_num and set attributes.
                output_skims[completed_measure] = np.nan_to_num(
                    np.array(temp_completed)
                )
                output_skims[failed_measure] = np.nan_to_num(np.array(temp_failed))
                output_skims[completed_measure].attrs.mode = path
                output_skims[failed_measure].attrs.mode = path
                output_skims[completed_measure].attrs.measure = "TRIPS"
                output_skims[failed_measure].attrs.measure = "FAILURES"
                output_skims[completed_measure].attrs.timePeriod = period
                output_skims[failed_measure].attrs.timePeriod = period
            
            # Iterate through each measure for the current path and period.
            for measure in measure_map.keys():
                name = f"{path}_{measure}__{period}"
                inOutputSkims = name in output_skim_tables
                if not inOutputSkims:
                    nSkimsCreated += 1
                    # If the skim is not already in the output, try to get it from input_skims.
                    temp, createdTemp = _get_field_or_else_empty(
                        input_skims, name, num_taz
                    )
                    missing_values = np.isnan(temp)
                    if np.any(missing_values):
                        # Impute missing values based on the measure type.
                        if measure == "TIME":
                            # Impute time based on distance and assumed average speed.
                            temp[missing_values] = (
                                distance_miles[missing_values] / 35 * 60
                            )  # Assume 35 mph average
                        elif measure == "DIST":
                            # Impute distance with the general distance matrix.
                            temp[missing_values] = distance_miles[missing_values]
                        elif measure == "BTOLL":
                            # Impute bridge toll with 0.0.
                            temp[missing_values] = 0.0
                        elif measure == "VTOLL":
                            # Impute vehicle toll based on path type.
                            if path.endswith("TOLL"):
                                toll = 5.0
                            else:
                                toll = 0.0
                            temp[missing_values] = toll
                        else:
                            logger.error(
                                f"Trying to read unknown measure {measure} from BEAM skims"
                            )

                    # Add the new skim to the output OMX file and set attributes.
                    output_skims[name] = temp
                    output_skims[name].attrs.mode = path
                    output_skims[name].attrs.measure = measure
                    output_skims[name].attrs.timePeriod = period
    logger.info(f"Created {nSkimsCreated} new skims in the omx object")
    # Close the OMX file if it was opened within this function.
    if needToClose:
        output_skims.close()


def _auto_skims(
    settings: PilatesConfig,
    auto_df: pd.DataFrame,
    order: np.ndarray,
    data_dir: Optional[str] = None,
) -> None:
    """
    Generates and populates auto (drive) OMX skims from a preprocessed auto DataFrame.

    This function processes auto skim data, grouping it by time period and path type.
    It then uses parallel processing to build Origin-Destination (OD) matrices
    for various auto measures. These matrices are then written to the OMX skims file.
    Missing values are imputed based on distances and assumed average speeds.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        auto_df (pd.DataFrame): A DataFrame containing preprocessed auto skims,
            expected to have a multi-index including 'timePeriod' and 'pathType'.
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones for the skim matrices.
        data_dir (Optional[str]): The directory where the OMX skims file is located.
            If None, it defaults to `settings.activitysim.local_mutable_data_folder`.
    """
    logger.info("Creating drive skims.")

    # Retrieve periods, highway paths, and measure map from settings.
    periods = settings.shared.skims.periods
    paths = settings.shared.skims.hwy_paths
    measure_map = settings.beam.asim_hwy_measure_map
    skims = read_skims(settings, mode="a", data_dir=data_dir)
    num_taz = len(order)
    # beam_hwy_paths is defined but not used in this function.
    # beam_hwy_paths = settings.beam.simulated_hwy_paths
    fill_na = np.nan

    # Group the auto DataFrame by time period and path type for parallel processing.
    groupBy = auto_df.groupby(level=[0, 1])

    # Use a multiprocessing Pool to build OD matrices in parallel.
    with Pool(cpu_count() - 1) as p:
        ret_list = p.map(
            _build_od_matrix_parallel,
            [
                (group.loc[name], measure_map, num_taz, order, fill_na)
                for name, group in groupBy
            ],
        )

    resultsDict = dict()
    # Consolidate results from parallel processing into a single dictionary.
    for (period, path), processedDict in zip(groupBy.groups.keys(), ret_list):
        # Special handling for "SOV" (Single Occupancy Vehicle) path type.
        # It seems to apply SOV measures to all defined highway paths.
        if path == "SOV":
            for measure, mtx in processedDict.items():
                for path_ in paths:
                    name = f"{path_}_{measure}__{period}"
                    resultsDict[name] = mtx

    # Iterate through all expected highway paths, periods, and measures to populate the OMX file.
    for period in periods:
        for path in paths:
            for measure in measure_map.keys():
                name = f"{path}_{measure}__{period}"
                if name in resultsDict:
                    mtx = resultsDict[name]
                    missing = np.isnan(mtx)

                    if missing.any():
                        # If there are missing values, attempt to impute them.
                        distances = np.array(skims["DIST"])
                        orig, dest = np.where(missing == True)
                        missing_measure = distances[orig, dest]

                        if measure == "DIST":
                            mtx[orig, dest] = missing_measure
                        elif measure == "TIME":
                            # Impute time based on distance and assumed average speed.
                            mtx[orig, dest] = missing_measure * (
                                60 / 40
                            )  # Assumes average speed of 40 miles/hour
                        else:
                            # Default to 0 for other measures (e.g., tolls) if missing.
                            mtx[orig, dest] = 0  # Assumes no toll or payment
                else:
                    # If a measure is entirely missing, create a zero matrix and log a warning.
                    mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
                    logger.warning(
                        f"Filling in default skim values for measure {name} because they're not in BEAM outputs"
                    )
                # Handle infinite values by replacing them with NaN.
                if np.any(np.isinf(mtx)):
                    logger.warning(
                        f"Replacing {np.isinf(mtx).sum().sum()} infs in skim {name}"
                    )
                    mtx[np.isinf(mtx)] = np.nan
                # Store the resulting matrix in the OMX skims file.
                skims[name] = mtx
    skims.close()


def _create_offset(
    settings: PilatesConfig, order: np.ndarray, data_dir: Optional[str] = None
) -> None:
    """
    Creates a 'zone_id' mapping (offset) in the OMX skims file.

    This mapping is essential for ActivitySim to correctly interpret the
    zone IDs within the skim matrices. It assigns a sequential integer
    ID to each zone based on the provided `order`.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        order (np.ndarray): A 1D NumPy array specifying the desired order of
            zones. The length of this array determines the range of zone IDs.
        data_dir (Optional[str]): The directory where the OMX skims file is located.
            If None, it defaults to `settings.activitysim.local_mutable_data_folder`.
    """
    logger.info("Creating skims offset keys")

    # Open skims object
    skims = read_skims(settings, mode="a", data_dir=data_dir)
    zone_id = np.arange(1, len(order) + 1)

    # Generint offset
    skims.create_mapping("zone_id", zone_id, overwrite=True)
    skims.close()


def create_skims_from_beam(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    Orchestrates the creation of ActivitySim-compatible skims from BEAM outputs.

    This is the main function for converting raw BEAM skims into the OMX format
    required by ActivitySim. It handles loading, preprocessing, and populating
    various skim matrices (distance, auto, transit, ridehail) into an OMX file.
    It also includes an option for skim validation.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        state (WorkflowState): The current workflow state, including the
            simulation year and other relevant information.
        workspace (Workspace): The current workspace object, providing access
            to data directories.
        output_dir (Optional[str]): The directory where the output `skims.omx`
            file should be saved. If None, it defaults to a path derived from
            `state.full_path` and `settings.activitysim.local_mutable_data_folder`.
        overwrite (bool): If True, forces the recreation of the `skims.omx` file,
            even if it already exists. Defaults to False.

    Returns:
        str: The full path to the generated `skims.omx` file.
    """
    if not output_dir:
        output_dir = os.path.join(
            state.full_path, settings.activitysim.local_mutable_data_folder
        )

    # If running in static skims mode and ActivitySim skims already exist
    # there is no point in recreating them.
    static_skims = settings.static_skims
    if static_skims:
        overwrite = False

    new, convertFromCsv, blankSkims = _create_skim_object(
        settings, overwrite, output_dir=output_dir
    )
    validation = settings.asim_validation

    order = zone_order(settings, workspace)

    if new:
        tempSkims = _load_raw_beam_skims(settings, convertFromCsv, blankSkims)
        if isinstance(tempSkims, pd.DataFrame):
            skims = tempSkims.loc[
                tempSkims.origin.isin(order) & tempSkims.destination.isin(order), :
            ]
            skims = _raw_beam_skims_preprocess(settings, state.year, skims, workspace)
            auto_df, transit_df = _create_skims_by_mode(settings, skims)
            ridehail_df = _load_raw_beam_origin_skims(settings)
            ridehail_df = _raw_beam_origin_skims_preprocess(
                settings, state.year, ridehail_df, workspace
            )

            # Create skims
            _distance_skims(
                settings,
                state.year,
                auto_df,
                order,
                data_dir=output_dir,
                workspace=workspace,
            )
            _auto_skims(settings, auto_df, order, data_dir=output_dir)
            _transit_skims(settings, transit_df, order, data_dir=output_dir)
            _ridehail_skims(settings, ridehail_df, order, data_dir=output_dir)

            del auto_df, transit_df
        else:
            # beam_output_dir = get_setting(settings, "beam.local_output_folder")
            _distance_skims(settings, state.year, tempSkims, order, data_dir=output_dir)
            _fill_auto_skims(settings, tempSkims, order, data_dir=output_dir)
            _fill_transit_skims(settings, tempSkims, order, data_dir=output_dir)
            _fill_ridehail_skims(settings, tempSkims, order, data_dir=output_dir)
            if isinstance(tempSkims, omx.File):
                tempSkims.close()

    _create_offset(settings, order, data_dir=output_dir)

    if validation:
        order = zone_order(settings, state.year)
        skim_validations(settings, state.year, order, data_dir=output_dir)

    return os.path.join(output_dir, "skims.omx")


def plot_skims(
    settings: PilatesConfig,
    zones: gpd.GeoDataFrame,
    skims: np.ndarray,
    order: np.ndarray,
    random_sample: int = 6,
    cols: int = 3,
    name: str = "DIST",
    units: str = "in miles",
) -> None:
    """
    Plots a map of skims for a random set of origin zones to all other zones.
    This function is primarily used for validation and debugging purposes,
    visualizing how a specific skim measure varies across the study area
    from selected origin zones.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        zones (gpd.GeoDataFrame): A GeoDataFrame containing zone geometries
            and attributes.
        skims (np.ndarray): A 2D NumPy array representing a skim measure,
            typically of shape (num_zones, num_zones).
        order (np.ndarray): A 1D NumPy array specifying the ordered zone IDs.
        random_sample (int): The number of random zones to select for plotting.
            Defaults to 6.
        cols (int): The number of columns in the resulting subplot grid.
            Defaults to 3.
        name (str): The name of the skim measure being plotted (e.g., "DIST", "SOV_TIME").
            Used in plot titles and filenames. Defaults to "DIST".
        units (str): The unit of analysis for the skim measure (e.g., "in miles",
            "in minutes"). Used in plot titles. Defaults to "in miles".
    """
    random_sample = random_sample
    cols = cols
    rows = int(random_sample / cols)
    zone_ids = list(zones.sample(random_sample).index.astype(int))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 20))

    counter = 0
    for row in range(rows):
        for col in range(cols):

            zone_id = int(zone_ids[counter])
            name_ = name + "_zone_id_" + str(zone_id)
            zone_measure = skims[zone_id - 1, :]
            empty = zone_measure.sum() == 0
            while empty:
                zone_id = int(list(zones.sample(1).index)[0])
                name_ = name + "_zone_id_" + str(zone_id)
                zone_measure = skims[zone_id - 1, :]
                empty = zone_measure.sum() == 0
            zones.loc[:, name_] = zone_measure
            zones.loc[:, name_] = zones.loc[:, name_].replace({999: np.nan, 0: np.nan})
            counter += 1
            bg_id = order[zone_id - 1]

            zones.plot(column=name_, legend=True, ax=axs[row][col])
            axs[row][col].set_title(
                "{0} ({1}) from zone {2} \n block_group {3} ".format(
                    name, units, zone_id, bg_id
                )
            )

    # Saving plots to files.
    asim_validation = settings.activitysim.validation_folder
    if not os.path.isdir(asim_validation):
        os.mkdir(asim_validation)

    save_path = os.path.join(asim_validation, "skims_validation_" + name + ".pdf")
    fig.savefig(save_path)


def skim_validations(
    settings: PilatesConfig,
    year: int,
    order: np.ndarray,
    workspace: "Workspace",
    data_dir: Optional[str] = None,
) -> None:
    """
    Generates and saves validation plots for various skim measures.

    This function reads the generated OMX skims, extracts key measures like
    distances, SOV travel times, and public transit travel times, and then
    uses `plot_skims` to visualize them for a random sample of zones.
    The plots are saved as PDF files for review.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        year (int): The simulation year (currently unused but kept for consistency).
        order (np.ndarray): A 1D NumPy array specifying the ordered zone IDs.
        workspace (Workspace): The current workspace object.
        data_dir (Optional[str]): The directory where the OMX skims file is located.
            If None, it defaults to `settings.activitysim.local_mutable_data_folder`.
    """
    logger.info("Generating skims validation plots.")
    skims = read_skims(settings, mode="r", data_dir=data_dir)
    zone = load_canonical_zones(settings, workspace)

    # Skims matrices
    num_zones = len(order)
    distances = np.array(skims["DIST"])
    sov_time = np.array(skims["SOV_TIME__AM"])
    loc_time_list = [
        "WLK_LOC_WLK_TOTIVT__AM",
        "WLK_LOC_WLK_XWAIT__AM",
        "WLK_LOC_WLK_IWAIT__AM",
        "WLK_LOC_WLK_WAUX__AM",
    ]
    PuT_time = np.zeros((num_zones, num_zones), dtype=np.float32)
    for measure in loc_time_list:
        time = np.array(skims[measure]) / 100
        PuT_time = PuT_time + time
    PuT_time[np.array(skims["WLK_LOC_WLK_TRIPS__AM"]) == 0] = np.nan

    # Plots
    plot_skims(settings, zone, distances, order, 6, 3, "DIST", "in miles")
    plot_skims(settings, zone, sov_time, order, 6, 3, "SOV_TIME", "in minutes")
    plot_skims(settings, zone, PuT_time, order, 6, 3, "WLK_LOC_WLK_TIME", "in minutes")


class ActivitysimPreprocessor(GenericPreprocessor):
    """
    ActivitySim-specific preprocessor that consolidates all preprocessing steps.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)
        self.required_input_data = ["usim_data", "beam_geoms", "asim_configs"]

    def _should_use_database_input(self, settings: PilatesConfig) -> bool:
        """
        Determine whether to use database input mode for ActivitySim.

        Database input mode is enabled when:
        1. UrbanSim is turned off (land_use_model != 'urbansim')
        2. Database is configured and enabled
        3. ActivitySim database configuration is present

        Args:
            settings: PILATES settings dictionary

        Returns:
            bool: True if database input should be used
        """
        # Check if UrbanSim is turned off
        land_use_model = settings.run.models.land_use or ""
        urbansim_disabled = land_use_model != "urbansim"

        # Check if database is configured
        database_enabled = settings.shared.database.enabled

        # Check if ActivitySim database configuration exists
        asim_db_enabled = settings.activitysim.database.enabled

        # Database input mode requires all conditions
        use_database = urbansim_disabled and database_enabled and asim_db_enabled

        if use_database:
            logger.info(
                "Database input mode enabled: UrbanSim is off and database is configured"
            )
        elif not urbansim_disabled:
            logger.info("Database input mode disabled: UrbanSim is enabled")
        elif not database_enabled:
            logger.info("Database input mode disabled: Database not configured")
        elif not asim_db_enabled:
            logger.info(
                "Database input mode disabled: ActivitySim database configuration not enabled"
            )

        return use_database

    def _clean_activitysim_data(
        self, df: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        """
        Clean and format data for ActivitySim compatibility.

        Args:
            df: Raw DataFrame from database
            table_name: Name of the ActivitySim table

        Returns:
            Cleaned DataFrame ready for ActivitySim
        """
        clean_df = df.copy()

        # Remove database metadata columns that ActivitySim doesn't need
        metadata_cols = [
            "id",
            "file_record_id",
            "run_id",
            "openlineage_id",
            "created_at",
        ]
        for col in metadata_cols:
            if col in clean_df.columns:
                clean_df = clean_df.drop(columns=[col])

        # Handle specific table cleaning
        if table_name == "households":
            # Ensure household_id is the index for ActivitySim
            if "household_id" in clean_df.columns:
                clean_df = clean_df.set_index("household_id")
                clean_df.index.name = "household_id"

            # Ensure required columns exist with defaults
            required_cols = {
                "TAZ": "1",
                "persons": 1,
                "income": 50000,
                "cars": 1,
                "HHT": 1,
                "workers": 1,
            }
            for col, default_val in required_cols.items():
                if col not in clean_df.columns:
                    clean_df[col] = default_val

        elif table_name == "persons":
            # Ensure person_id is the index for ActivitySim
            if "person_id" in clean_df.columns:
                clean_df = clean_df.set_index("person_id")
                clean_df.index.name = "person_id"

            # Ensure required columns exist with defaults
            required_cols = {
                "household_id": 1,
                "TAZ": "1",
                "age": 35,
                "worker": 0,
                "student": 0,
                "ptype": 1,
                "pemploy": 1,
                "pstudent": 3,
                "member_id": 1,
                "workplace_taz": -1,
                "school_taz": -1,
                "home_x": 0.0,
                "home_y": 0.0,
            }
            for col, default_val in required_cols.items():
                if col not in clean_df.columns:
                    clean_df[col] = default_val

        elif table_name == "land_use":
            # Ensure TAZ is the index for ActivitySim
            if "TAZ" in clean_df.columns:
                clean_df = clean_df.set_index("TAZ")
                clean_df.index.name = "TAZ"

            # Ensure required columns exist with defaults
            required_cols = {
                "TOTPOP": 100,
                "TOTHH": 50,
                "TOTEMP": 75,
                "TOTACRE": 10.0,
                "area_type": 3,
                "employment_density": 7.5,
                "pop_density": 10.0,
                "hh_density": 5.0,
            }
            for col, default_val in required_cols.items():
                if col not in clean_df.columns:
                    clean_df[col] = default_val

        # Convert data types appropriately
        for col in clean_df.columns:
            # Convert numeric columns that might be stored as objects
            if clean_df[col].dtype == "object":
                try:
                    # Try to convert to numeric
                    clean_df[col] = pd.to_numeric(clean_df[col], errors="ignore")
                except:
                    pass

        # Replace any NaN values with appropriate defaults
        clean_df = clean_df.fillna(0)

        logger.debug(
            f"Cleaned {table_name} data: shape {clean_df.shape}, columns {list(clean_df.columns)}"
        )
        return clean_df

    def copy_data_to_mutable_location(
        self,
        settings: PilatesConfig,
        output_dir: str,
    ) -> Tuple[RecordStore, RecordStore]:
        # Delegate to module-level function

        return _copy_data_to_mutable_location(
            settings, output_dir, self.provenance_tracker
        )

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: RecordStore = RecordStore(),
    ) -> RecordStore:
        """
        Run all preprocessing steps for ActivitySim in order.
        """
        settings = getattr(self.state, "full_settings", None)
        if settings is None:
            raise ValueError("Workflow state must have a 'full_settings' attribute.")

        logger.info("Starting ActivitysimPreprocessor.preprocess()")

        # Record inputs to preprocessor
        # Raw BEAM skims are an input.
        skims_fname = settings.shared.skims.fname
        path_to_beam_skims_in_current_run_workspace = os.path.join(
            workspace.get_beam_mutable_data_dir(),
            settings.run.region,
            skims_fname,
        )

        # Ensure BEAM input data is present, even if Initialization was skipped or incomplete
        if not os.path.exists(path_to_beam_skims_in_current_run_workspace):
            logger.info(
                "[ActivitysimPreprocessor] BEAM skims not found in current workspace. Copying from production."
            )
            beam_production_path = os.path.abspath(
                os.path.join(
                    find_project_root(),  # Assuming find_project_root is available
                    "pilates",
                    "beam",
                    "production",
                    settings.run.region,
                )
            )
            dest_dir = os.path.dirname(path_to_beam_skims_in_current_run_workspace)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copyfile(
                os.path.join(beam_production_path, skims_fname),
                path_to_beam_skims_in_current_run_workspace,
            )
            # Also record provenance for this copy
            self.provenance_tracker.record_input_file(
                "activitysim_preprocessor",
                os.path.join(beam_production_path, skims_fname),
                description="BEAM skims from production (copied on demand)",
                short_name="omx_skims_production",
            )
            self.provenance_tracker.record_output_file(
                "activitysim_preprocessor",
                path_to_beam_skims_in_current_run_workspace,
                description="BEAM skims copied to current workspace",
                short_name="omx_skims_current_workspace",
            )

        input_records = workspace.output_data.get("activitysim", RecordStore())
        input_records_filtered = RecordStore(
            recordList=[
                rec
                for rec in input_records.all_records()
                if rec.short_name in self.required_input_data
            ]
        )
        output_records = RecordStore()

        pre_run_hash = self.provenance_tracker.start_model_run(
            "activitysim_preprocessor",
            year=self.state.current_year,
            iteration=self.state.current_inner_iter,
            description="Preprocessing for ActivitySim warm start",
            inputs=input_records_filtered,
        )

        if os.path.exists(
            path_to_beam_skims_in_current_run_workspace
        ):  # <--- This condition should now be true
            input_skims_record = self.provenance_tracker.record_input_file(
                "activitysim_preprocessor",
                path_to_beam_skims_in_current_run_workspace,
                short_name="omx_skims",
                model_run_id=pre_run_hash,
                description="Raw BEAM OD skims",
            )
            skims_loc = os.path.join(workspace.get_asim_mutable_data_dir(), "skims.omx")
            os.makedirs(os.path.dirname(skims_loc), exist_ok=True)
            shutil.copyfile(path_to_beam_skims_in_current_run_workspace, skims_loc)
            input_records.add_record(input_skims_record)
        else:
            os.makedirs(workspace.get_asim_mutable_data_dir(), exist_ok=True)
            skims_loc = create_skims_from_beam(
                settings,
                self.state,
                workspace,
                output_dir=workspace.get_asim_mutable_data_dir(),
            )

        # Record the skims file as an OUTPUT of the preprocessor
        skim_record = self.provenance_tracker.record_output_file(
            "activitysim_preprocessor",
            skims_loc,
            model_run_id=pre_run_hash,
            short_name="omx_skims",
            description="OD Skims copied over to ASim data directory",
        )

        if self.state.current_inner_iter > 0:
            # 2: Re-use existing persons/households/skim cache
            last_asim_hash = self.provenance_tracker.run_info.get_latest_model_run(
                "activitysim"
            )
            record_hashes = self.provenance_tracker.run_info.model_runs[
                last_asim_hash
            ].input_record_hashes
            data_from_usim = [
                self.provenance_tracker.run_info.file_records.get(h)
                for h in record_hashes
                if h in self.provenance_tracker.run_info.file_records
            ]
            data_from_usim = [
                r
                for r in data_from_usim
                if r.short_name in ["households_asim_in", "persons_asim_in"]
            ]
            logger.info(
                f"Retrieved {len(data_from_usim)} records from previous ActivitySim run."
            )
        else:
            # 2. Create ActivitySim input data from database or UrbanSim H5
            # Check if database input mode is configured
            if self._should_use_database_input(settings):
                logger.info("Using database input mode for ActivitySim")
                data_from_usim = create_asim_data_from_database(
                    settings,
                    self.state,
                    workspace,
                    self.provenance_tracker,
                    model_run_hash=pre_run_hash,
                )
            else:
                logger.info("Using H5 input mode for ActivitySim")
                data_from_usim = create_asim_data_from_h5(
                    settings,
                    self.state,
                    workspace,
                    self.provenance_tracker,
                    model_run_hash=pre_run_hash,
                )

        logger.info("ActivitysimPreprocessor.preprocess() completed.")

        all_outputs = (
            ([skim_record] if skim_record else [])
            + data_from_usim
            + list(output_records.records.values())
        )

        self.provenance_tracker.complete_model_run(
            run_hash=pre_run_hash, output_records=all_outputs
        )
        return RecordStore(recordList=all_outputs)


#######################################
#### UrbanSim to ActivitySim tables ###
#######################################
def _get_full_time_enrollment(state_fips: str, year: str) -> pd.Series:
    """
    Downloads full-time college enrollment data from educationdata.urban.org
    for a given state and year.

    Args:
        state_fips (str): The FIPS code of the state to query.
        year (str): The academic year for which to retrieve enrollment data.

    Returns:
        pd.Series: A pandas Series with 'unitid' as the index and
            'full_time' enrollment as values.
    """
    base_url = (
        "https://educationdata.urban.org/api/v1/"
        "{t}/{so}/{e}/{y}/{l}/?{f}&{s}&{r}&{cl}&{ds}&{fips}"
    )
    levels = ["undergraduate", "graduate"]
    enroll_list = []
    for level in levels:
        level_url = base_url.format(
            t="college-university",
            so="ipeds",
            e="fall-enrollment",
            y=year,
            l=level,
            f="ftpt=1",
            s="sex=99",
            r="race=99",
            cl="class_level=99",
            ds="degree_seeking=99",
            fips="fips={0}".format(state_fips),
        )
        enroll_result = requests.get(level_url)
        enroll = pd.DataFrame(enroll_result.json()["results"])
        enroll = enroll[["unitid", "enrollment_fall"]].rename(
            columns={"enrollment_fall": level}
        )
        enroll[level].clip(0, inplace=True)
        enroll.set_index("unitid", inplace=True)
        enroll_list.append(enroll)
    full_time = pd.concat(enroll_list, axis=1).fillna(0)
    full_time["full_time"] = full_time["undergraduate"] + full_time["graduate"]
    s = full_time.full_time
    assert s.index.name == "unitid"
    return s


def _get_part_time_enrollment(state_fips: str) -> pd.Series:
    """
    Downloads part-time college enrollment data from educationdata.urban.org
    for a given state. The year is currently hardcoded to "2015".

    Args:
        state_fips (str): The FIPS code of the state to query.

    Returns:
        pd.Series: A pandas Series with 'unitid' as the index and
            'part_time' enrollment as values.
    """
    base_url = (
        "https://educationdata.urban.org/api/v1/"
        "{t}/{so}/{e}/{y}/{l}/?{f}&{s}&{r}&{cl}&{ds}&{fips}"
    )
    levels = ["undergraduate", "graduate"]
    enroll_list = []
    for level in levels:
        level_url = base_url.format(
            t="college-university",
            so="ipeds",
            e="fall-enrollment",
            y="2015",
            l=level,
            f="ftpt=2",
            s="sex=99",
            r="race=99",
            cl="class_level=99",
            ds="degree_seeking=99",
            fips="fips={0}".format(state_fips),
        )

        enroll_result = requests.get(level_url)
        enroll = pd.DataFrame(enroll_result.json()["results"])
        enroll = enroll[["unitid", "enrollment_fall"]].rename(
            columns={"enrollment_fall": level}
        )
        enroll[level].clip(0, inplace=True)
        enroll.set_index("unitid", inplace=True)
        enroll_list.append(enroll)

    part_time = pd.concat(enroll_list, axis=1).fillna(0)
    part_time["part_time"] = part_time["undergraduate"] + part_time["graduate"]
    s = part_time.part_time
    assert s.index.name == "unitid"
    return s


def _get_school_enrollment(state_fips: str, county_codes: List[str]) -> pd.DataFrame:
    """
    Downloads K-12 school enrollment data from educationdata.urban.org
    for a given state and a list of county codes.

    Args:
        state_fips (str): The FIPS code of the state to query.
        county_codes (List[str]): A list of 3-digit county FIPS codes
            within the state to filter the data.

    Returns:
        pd.DataFrame: A DataFrame containing school enrollment data,
            indexed by 'ncessch' (NCES school ID), with columns for
            county code, latitude, longitude (renamed to 'x' and 'y'),
            and enrollment.
    """
    logger.info("Downloading school enrollment data from educationdata.urban.org!")
    base_url = (
        "https://educationdata.urban.org/api/v1/"
        + "{topic}/{source}/{endpoint}/{year}/?{filters}"
    )

    # at the moment you can't seem to filter results by county
    enroll_filters = "fips={0}".format(state_fips)
    enroll_url = base_url.format(
        topic="schools",
        source="ccd",
        endpoint="directory",
        year="2015",
        filters=enroll_filters,
    )

    school_tables = []
    while True:
        response = requests.get(enroll_url).json()
        count = response["count"]
        next_page = response["next"]
        data = response["results"]
        enroll = pd.DataFrame(data)
        school_tables.append(enroll)
        if next_page is not None:
            enroll_url = next_page
            time.sleep(2)
        else:
            break

    enrollment = pd.concat(school_tables, axis=0)
    assert len(enrollment) == count
    enrollment = enrollment[
        ["ncessch", "county_code", "latitude", "longitude", "enrollment"]
    ].set_index("ncessch")
    enrollment["county_code"] = enrollment["county_code"].str[-3:]
    enrollment = enrollment[enrollment["county_code"].isin(county_codes)]
    enrollment.rename(columns={"longitude": "x", "latitude": "y"}, inplace=True)
    enrollment["enrollment"].clip(0, inplace=True)
    enrollment = enrollment[~enrollment.enrollment.isna()]

    return enrollment


def _get_college_enrollment(state_fips: str, county_codes: List[str]) -> pd.DataFrame:
    """
    Downloads college directory and enrollment data (full-time and part-time)
    from educationdata.urban.org for a given state and list of county codes.

    Args:
        state_fips (str): The FIPS code of the state to query.
        county_codes (List[str]): A list of 3-digit county FIPS codes
            within the state to filter the data.

    Returns:
        pd.DataFrame: A DataFrame containing college information, including
            institution name, coordinates (renamed to 'x' and 'y'),
            and calculated full-time and part-time enrollment figures.
            Indexed by 'unitid'.
    """
    year = "2015"
    logger.info("Downloading college data from educationdata.urban.org!")
    base_url = (
        "https://educationdata.urban.org/api/v1/"
        + "{topic}/{source}/{endpoint}/{year}/?{filters}"
    )

    colleges_list = []
    total_count = 0
    for county in county_codes:
        county_fips = str(state_fips) + str(county)
        college_filters = "county_fips={0}".format(county_fips)
        college_url = base_url.format(
            topic="college-university",
            source="ipeds",
            endpoint="directory",
            year=year,
            filters=college_filters,
        )
        response = requests.get(college_url).json()
        count = response["count"]
        total_count += count
        college = pd.DataFrame(response["results"])
        colleges_list.append(college)
        time.sleep(2)

    colleges = pd.concat(colleges_list)
    assert len(colleges) == total_count
    colleges = colleges[["unitid", "inst_name", "longitude", "latitude"]].set_index(
        "unitid"
    )
    colleges.rename(columns={"longitude": "x", "latitude": "y"}, inplace=True)

    logger.info(
        "Downloading college full-time enrollment data from " "educationdata.urban.org!"
    )
    fte = _get_full_time_enrollment(state_fips, year)
    colleges["full_time_enrollment"] = fte.reindex(colleges.index)

    logger.info(
        "Downloading college part-time enrollment data from " "educationdata.urban.org!"
    )
    pte = _get_part_time_enrollment(state_fips)
    colleges["part_time_enrollment"] = pte.reindex(colleges.index)
    return colleges


def _get_park_cost(
    zones: pd.DataFrame,
    weights: List[float],
    index_cols: List[str],
    output_cols: List[str],
) -> pd.Series:
    """
    Calculates a 'parking cost' metric based on weighted sums of various
    zone-level attributes.

    Args:
        zones (pd.DataFrame): A DataFrame containing zone data with the
            `output_cols` present.
        weights (List[float]): A list of numerical weights to apply to the
            corresponding `index_cols`.
        index_cols (List[str]): A list of column names from `zones` that
            correspond to the `weights`.
        output_cols (List[str]): A list of column names from `zones` that
            will be used in the matrix multiplication with `weights`.

    Returns:
        pd.Series: A pandas Series representing the calculated parking cost
            for each zone, with values clipped at 0 (no negative costs).
    """
    params = pd.Series(weights, index=index_cols)
    cols = zones[output_cols]
    s = cols @ params
    return s.where(s > 0, 0)


def _compute_area_type_metric(zones: pd.DataFrame) -> pd.Series:
    """
    Computes a metric used to classify urban area types based on population,
    employment, and acreage.

    This metric is sensitive to the region and TAZ geometries it was designed
    for (SF Bay Area). Its accuracy should be visually assessed for new regions.

    Args:
        zones (pd.DataFrame): A DataFrame containing zone data with 'TOTPOP',
            'TOTEMP', and 'TOTACRE' columns.

    Returns:
        pd.Series: A pandas Series containing the computed area type metric
            for each zone, with NaN values filled with 0.
    """
    """
    Because of the modifiable areal unit problem, it is probably a good
    idea to visually assess the accuracy of this metric when implementing
    in a new region. The metric was designed using SF Bay Area data and TAZ
    geometries. So what is considered "suburban" by SFMTC standard might be
    "urban" or "urban fringe" in less densesly developed regions, which
    can impact the results of the auto ownership and mode choice models.

    This issue should eventually resolve itself once we are able to re-
    estimate these two models for every new region/implementation. In the
    meantime, we expect that for regions less dense than the SF Bay Area,
    the area types classifications will be overly conservative. If anything,
    this bias results towards higher auto-ownership and larger auto-oriented
    mode shares. However, we haven't found this to be the case.
    """
    zones_df = zones[["TOTPOP", "TOTEMP", "TOTACRE"]].copy()

    metric_vals = ((1 * zones_df["TOTPOP"]) + (2.5 * zones_df["TOTEMP"])) / zones_df[
        "TOTACRE"
    ]

    return metric_vals.fillna(0)


def _compute_area_type(zones: pd.DataFrame) -> pd.Series:
    """
    Classifies zones into urban area types based on the `area_type_metric`.

    The classification uses predefined bins to assign an integer area type:
    0 = regional core, 1 = central business district, 2 = urban business,
    3 = urban, 4 = suburban, 5 = rural.

    Args:
        zones (pd.DataFrame): A DataFrame containing zone data, which must
            include an 'area_type_metric' column.

    Returns:
        pd.Series: A pandas Series (of string type) indicating the classified
            area type for each zone.
    """
    # Integer, 0=regional core, 1=central business district,
    # 2=urban business, 3=urban, 4=suburban, 5=rural
    area_types = pd.cut(
        zones["area_type_metric"],
        [0, 6, 30, 55, 100, 300, float("inf")],
        labels=["5", "4", "3", "2", "1", "0"],
        include_lowest=True,
    ).astype(str)
    return area_types


def _clean_activitysim_data_for_csv(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Clean and format data for ActivitySim CSV compatibility.

    This is a standalone version for use in module-level functions.

    Args:
        df (pd.DataFrame): Raw DataFrame from database.
        table_name (str): Name of the ActivitySim table (e.g., "households", "persons", "land_use").

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for ActivitySim, with metadata
            columns removed, required columns ensured with defaults, and
            appropriate data types.
    """
    clean_df = df.copy()

    # Remove database metadata columns that ActivitySim doesn't need
    metadata_cols = ["id", "file_record_id", "run_id", "openlineage_id", "created_at"]
    for col in metadata_cols:
        if col in clean_df.columns:
            clean_df = clean_df.drop(columns=[col])

    # Handle specific table cleaning
    if table_name == "households":
        # Ensure household_id is the index for ActivitySim
        if "household_id" in clean_df.columns:
            clean_df = clean_df.set_index("household_id")
            clean_df.index.name = "household_id"

        # Ensure required columns exist with defaults
        required_cols = {
            "TAZ": "1",
            "persons": 1,
            "income": 50000,
            "cars": 1,
            "HHT": 1,
            "workers": 1,
        }
        for col, default_val in required_cols.items():
            if col not in clean_df.columns:
                clean_df[col] = default_val

    elif table_name == "persons":
        # Ensure person_id is the index for ActivitySim
        if "person_id" in clean_df.columns:
            clean_df = clean_df.set_index("person_id")
            clean_df.index.name = "person_id"

        # Ensure required columns exist with defaults
        required_cols = {
            "household_id": 1,
            "TAZ": "1",
            "age": 35,
            "worker": 0,
            "student": 0,
            "ptype": 1,
            "pemploy": 1,
            "pstudent": 3,
            "member_id": 1,
            "workplace_taz": -1,
            "school_taz": -1,
            "home_x": 0.0,
            "home_y": 0.0,
        }
        for col, default_val in required_cols.items():
            if col not in clean_df.columns:
                clean_df[col] = default_val

    elif table_name == "land_use":
        # Ensure TAZ is the index for ActivitySim
        if "TAZ" in clean_df.columns:
            clean_df = clean_df.set_index("TAZ")
            clean_df.index.name = "TAZ"

        # Ensure required columns exist with defaults
        required_cols = {
            "TOTPOP": 100,
            "TOTHH": 50,
            "TOTEMP": 75,
            "TOTACRE": 10.0,
            "area_type": 3,
            "employment_density": 7.5,
            "pop_density": 10.0,
            "hh_density": 5.0,
        }
        for col, default_val in required_cols.items():
            if col not in clean_df.columns:
                clean_df[col] = default_val

    # Convert data types appropriately
    for col in clean_df.columns:
        # Convert numeric columns that might be stored as objects
        if clean_df[col].dtype == "object":
            try:
                # Try to convert to numeric
                clean_df[col] = pd.to_numeric(clean_df[col], errors="ignore")
            except:
                pass

    # Replace any NaN values with appropriate defaults
    clean_df = clean_df.fillna(0)

    logger.debug(
        f"Cleaned {table_name} data: shape {clean_df.shape}, columns {list(clean_df.columns)}"
    )
    return clean_df


def enrollment_tables(
    settings: PilatesConfig,
    zones: gpd.GeoDataFrame,
    enrollment_type: str = "schools",
    asim_zone_id_col: str = "TAZ",
) -> pd.DataFrame:
    """
    Retrieves and processes school or college enrollment data, assigning
    enrollment figures to ActivitySim zones.

    This function first attempts to load enrollment data from a local CSV file.
    If the file does not exist, it downloads the data from educationdata.urban.org.
    It then maps the enrollment locations to the appropriate ActivitySim zones
    (TAZ, block group, etc.) and saves the processed data locally for future use.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        zones (gpd.GeoDataFrame): A GeoDataFrame containing zone geometries,
            used for spatial mapping of enrollment points to zones.
        enrollment_type (str): The type of enrollment data to retrieve.
            Must be either "schools" (K-12) or "colleges". Defaults to "schools".
        asim_zone_id_col (str): The name of the column in the ActivitySim
            input tables that represents the zone ID (e.g., "TAZ", "BLKGRP").
            Defaults to "TAZ".

    Returns:
        pd.DataFrame: A DataFrame containing the enrollment data with an
            assigned ActivitySim zone ID column.

    Raises:
        KeyError: If an invalid `enrollment_type` is provided.
    """
    region = settings.run.region
    FIPS = settings.shared.geography.FIPS
    state_fips = FIPS["state"]
    county_codes = FIPS["counties"]
    local_crs = settings.shared.geography.local_crs

    zone_type = settings.shared.geography.zones.zone_type
    path_to_schools_data = "pilates/utils/data/{0}/{1}_{2}.csv".format(
        region, zone_type, enrollment_type
    )
    assert enrollment_type in [
        "schools",
        "colleges",
    ], "enrollemnt type one of ['schools', 'colleges']"

    if not os.path.exists(path_to_schools_data):
        if enrollment_type == "schools":
            enrollment = _get_school_enrollment(state_fips, county_codes)
        elif enrollment_type == "colleges":
            enrollment = _get_college_enrollment(state_fips, county_codes)
        else:
            raise KeyError("enrollemnt type one of ['schools', 'colleges']")
    else:
        logger.info("Reading school enrollment data from disk!")
        enrollment = pd.read_csv(path_to_schools_data, dtype={asim_zone_id_col: str})

    if asim_zone_id_col not in enrollment.columns:
        enrollment_df = enrollment[["x", "y"]].copy()
        enrollment_df.index.name = "school_id"
        enrollment[asim_zone_id_col] = get_zone_from_points(
            enrollment_df, zones, local_crs
        )

        enrollment = enrollment.dropna(subset=[asim_zone_id_col])
        enrollment[asim_zone_id_col] = enrollment[asim_zone_id_col].astype(str)
        del enrollment_df
        logger.info("Saving {} enrollment data to disk!".format(enrollment_type))
        enrollment.to_csv(path_to_schools_data)

    return enrollment


def _copy_data_to_mutable_location(
    settings: PilatesConfig,
    folder_path: str,
    provenance_tracker: FileProvenanceTracker,
) -> Tuple[RecordStore, RecordStore]:
    """
    Copies ActivitySim source data (canonical zones, clipped geometries, configs)
    to a mutable location within the current run's workspace.

    This function ensures that ActivitySim has access to necessary input files
    by copying them from their original source locations to a designated
    mutable directory. It also records the provenance of these copied files.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        folder_path (str): The base path for the mutable ActivitySim data directory.
        provenance_tracker (FileProvenanceTracker): The provenance tracker
            instance to record file operations.

    Returns:
        Tuple[RecordStore, RecordStore]: A tuple containing two RecordStore objects:
            - `input_records`: Records of the original source files.
            - `output_records`: Records of the copied files in the mutable location.
    """
    logger.info("Copying ActivitySim source data to mutable location.")
    input_dir = folder_path
    os.makedirs(input_dir, exist_ok=True)
    region = get_setting(settings, "run.region")
    input_records = RecordStore()
    output_records = RecordStore()
    project_root = find_project_root()

    # --- 1. Handle Canonical Zone Geometries ---
    zone_source_path = settings.shared.geography.zones.source_file
    if not os.path.isabs(zone_source_path):
        zone_source_path = os.path.join(project_root, zone_source_path)

    if os.path.exists(zone_source_path):
        zone_fname = os.path.basename(zone_source_path)
        asim_zones_path = os.path.join(input_dir, zone_fname)
        logger.info(
            f"Copying canonical zones from {zone_source_path} to {asim_zones_path}"
        )
        shutil.copy(zone_source_path, asim_zones_path)

        input_rec = provenance_tracker.record_input_file(
            "activitysim_preprocessor",
            zone_source_path,
            short_name="canonical_zones_source",
        )
        output_rec = provenance_tracker.record_output_file_with_inputs(
            "activitysim_preprocessor",
            asim_zones_path,
            [input_rec],
            short_name="canonical_zones_mutable",
        )
        input_records.add_record(input_rec)
        output_records.add_record(output_rec)
    else:
        logger.warning(
            f"Canonical zone source file not found at {zone_source_path}, skipping copy."
        )

    # --- 2. Handle Clipped Geometries (from BEAM) ---
    clipped_geoms_rel_path = get_setting(settings, "activitysim.clipped_geoms_path")
    if clipped_geoms_rel_path:
        # This path is relative to the BEAM router directory
        beam_prod_dir = os.path.join(project_root, settings.beam.local_input_folder)
        router_dir = os.path.join(beam_prod_dir, region, settings.beam.router_directory)
        clipped_geoms_source_path = os.path.join(router_dir, clipped_geoms_rel_path)

        if os.path.exists(clipped_geoms_source_path):
            clipped_fname = os.path.basename(clipped_geoms_source_path)
            asim_clipped_path = os.path.join(input_dir, clipped_fname)
            logger.info(
                f"Copying clipped geoms from {clipped_geoms_source_path} to {asim_clipped_path}"
            )
            shutil.copy(clipped_geoms_source_path, asim_clipped_path)

            input_rec = provenance_tracker.record_input_file(
                "activitysim_preprocessor",
                clipped_geoms_source_path,
                short_name="clipped_geoms_source",
            )
            output_rec = provenance_tracker.record_output_file_with_inputs(
                "activitysim_preprocessor",
                asim_clipped_path,
                [input_rec],
                short_name="clipped_geoms_mutable",
            )
            input_records.add_record(input_rec)
            output_records.add_record(output_rec)
        else:
            logger.warning(
                f"Clipped geoms file not found at {clipped_geoms_source_path}, skipping copy."
            )

    # --- 3. Handle ActivitySim Configs ---
    configs_source_dir = os.path.join(
        get_setting(settings, "activitysim.local_configs_folder"),
        get_setting(settings, "run.region"),
    )
    configs_dest_dir = os.path.abspath(
        os.path.join(
            folder_path,
            "..",
            "..",
            get_setting(settings, "activitysim.local_mutable_configs_folder"),
        )
    )
    logger.info(
        "Moving asim configs from {0} to {1}".format(
            configs_source_dir, configs_dest_dir
        )
    )
    if os.path.exists(configs_dest_dir):
        shutil.rmtree(configs_dest_dir)
    shutil.copytree(configs_source_dir, configs_dest_dir)

    git_hash = provenance_tracker.get_git_hash(configs_source_dir)
    input_records.add_record(
        provenance_tracker.record_repo_input(
            "activitysim_preprocessor",
            repo_path=configs_source_dir,
            short_name="asim_configs",
            description="ActivitySim configs repo",
            git_hash=git_hash,
        )
    )
    output_records.add_record(
        provenance_tracker.record_repo_input(
            "activitysim_preprocessor",
            repo_path=configs_dest_dir,
            short_name="asim_configs",
            description="ActivitySim configs repo",
        )
    )
    return input_records, output_records


def copy_beam_geoms(
    settings: PilatesConfig,
    beam_geoms_location: str,
    asim_geoms_location: str,
    beam_shape_location: Optional[str],
    provenance_tracker: FileProvenanceTracker,
    workspace: "Workspace",
) -> Tuple[RecordStore, RecordStore]:
    """
    Copies and processes BEAM geometry files for ActivitySim.

    This function reads BEAM geometry data, maps zone IDs according to the
    configured zone type, and saves the processed geometries for ActivitySim.
    It also handles an optional BEAM shapefile, updating its zone IDs if present.
    All file operations are tracked for provenance.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        beam_geoms_location (str): The file path to the input BEAM geometries CSV.
        asim_geoms_location (str): The destination file path for the processed
            ActivitySim geometries CSV.
        beam_shape_location (Optional[str]): The optional file path to the BEAM
            zones shapefile. If provided, its zone IDs will be updated.
        provenance_tracker (FileProvenanceTracker): The provenance tracker
            instance to record file operations.
        workspace (Workspace): The current workspace object.

    Returns:
        Tuple[RecordStore, RecordStore]: A tuple containing two RecordStore objects:
            - `input_records`: Records of the original BEAM geometry files.
            - `output_records`: Records of the processed ActivitySim geometry files.
    """
    zone_type_column = {"block_group": "BLKGRP", "taz": "TAZ", "block": "BLK"}
    beam_geoms_file = pd.read_csv(beam_geoms_location, dtype={"GEOID": str})
    beam_geoms_in = provenance_tracker.record_input_file(
        model="activitysim_preprocessor",
        file_path=beam_geoms_location,
        short_name="beam_geoms_reference",
        description="BEAM geometry file",
    )
    zone_type = get_setting(settings, "shared.geography.zones.zone_type").lower()
    zone_id_col = zone_type_column[zone_type]
    mapping = get_block_to_zone_mapping(
        settings, get_setting(settings, "run.start_year"), workspace
    )

    if zone_id_col not in beam_geoms_file.columns:

        if zone_type == "block":
            logger.info("Mapping block IDs")
            beam_geoms_file["TAZ"] = (
                beam_geoms_file["GEOID"].astype(str).replace(mapping)
            )

        elif zone_type == "block_group":
            logger.info("Mapping block group IDs to TAZ ids")
            beam_geoms_file["TAZ"] = (
                beam_geoms_file["GEOID"].astype(str).replace(mapping)
            )

        elif zone_type == "taz":
            from_col = get_setting(
                settings,
                "shared.skims.geoms_index_col",
                default=get_setting(settings, "geoms_index_col", "zone_id"),
            )
            to_col = "TAZ"
            logger.info("Renaming TAZ column from {0} to {1}".format(from_col, to_col))
            beam_geoms_file.rename(columns={from_col: to_col}, inplace=True)
        else:
            logger.error(f"Unrecognized zone type {zone_type}, ASim may fail")

    beam_geoms_file.to_csv(asim_geoms_location)
    asim_geoms_out = provenance_tracker.record_output_file(
        model="activitysim_preprocessor",
        file_path=asim_geoms_location,
        short_name="asim_geoms",
        description="ASIM geometry file",
    )

    inputs = [beam_geoms_in]
    outputs = [asim_geoms_out]

    if beam_shape_location is not None:
        logger.info(
            "Mapping BEAM geometry geoid column {0} to zone_id column {1}".format(
                get_setting(settings, "beam.skim_zone_geoid_col"),
                get_setting(settings, "beam.skim_zone_source_id_col"),
            )
        )
        beam_shape_in = provenance_tracker.record_input_file(
            model="activitysim_preprocessor",
            file_path=beam_shape_location,
            short_name="beam_shape",
            description="BEAM zones shapefile",
        )
        inputs.append(beam_shape_in)
        zones = gpd.read_file(beam_shape_location)
        zones[get_setting(settings, "beam.skim_zone_source_id_col")] = (
            zones[get_setting(settings, "beam.skim_zone_geoid_col")]
            .astype(str)
            .map(mapping)
        )
        logger.info(
            "Re-saving BEAM geometry shapefile with updated zone_id to {0}".format(
                beam_shape_location
            )
        )
        zones.to_file(beam_shape_location)
        beam_shape_out = provenance_tracker.record_output_file(
            model="activitysim_preprocessor",
            file_path=beam_shape_location,
            short_name="beam_shape",
            description="BEAM zones shapefile",
        )
        outputs.append(beam_shape_out)
    return RecordStore(recordList=inputs), RecordStore(recordList=outputs)


def _update_persons_table(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    unassigned_households: pd.Series,
    blocks: pd.DataFrame,
    asim_zone_id_col: str = "TAZ",
) -> pd.DataFrame:
    """
    Updates person attributes and assigns zones for ActivitySim processing.

    This function cleans and enriches the persons DataFrame by:
    - Removing persons belonging to unassigned households.
    - Assigning ActivitySim zone IDs based on household block IDs.
    - Calculating person types (`ptype`), employment status (`pemploy`),
      and student status (`pstudent`) based on age, worker, and student flags.
    - Adding home coordinates (x, y) from block data.
    - Converting and cleaning workplace and school TAZs.
    - Filtering out invalid records (e.g., null zone IDs, invalid ages).
    - Recalculating `member_id` within each household.
    - Clearing school/work locations for specific person types (e.g., workers
      shouldn't have school locations, non-workers/students shouldn't have
      work/school locations).

    Args:
        persons (pd.DataFrame): DataFrame containing person records.
        households (pd.DataFrame): DataFrame containing household records,
            including 'block_id'.
        unassigned_households (pd.Series): A Series of household IDs that
            could not be assigned to a location. Persons from these households
            will be dropped.
        blocks (pd.DataFrame): DataFrame containing block/zone information,
            including 'x', 'y' coordinates and the `asim_zone_id_col`.
        asim_zone_id_col (str): Column name for the ActivitySim zone ID
            (e.g., 'TAZ'). Defaults to 'TAZ'.

    Returns:
        pd.DataFrame: DataFrame with updated and cleaned person attributes,
            ready for ActivitySim.
    """
    # Convert index to int and filter out unassigned persons
    persons.index = persons.index.astype(int)
    unassigned_mask = persons.household_id.isin(unassigned_households)
    logger.info(
        f"Dropping {unassigned_mask.sum()} people from {unassigned_households.shape[0]} unassigned households"
    )

    persons = persons.loc[~unassigned_mask].copy()

    household_zone_map = households["block_id"].map(blocks[asim_zone_id_col])
    persons[asim_zone_id_col] = (
        persons["household_id"].map(household_zone_map).astype(str)
    )

    # Precalculate age ranges and worker/student status
    persons["age"] = persons["age"].fillna(-1)
    age_ranges = {
        "adult": persons.age >= 18,
        "working_age": persons.age.between(18, 64, inclusive="both"),
        "senior": persons.age >= 65,
        "teen": persons.age.between(16, 17, inclusive="both"),
        "child": persons.age.between(6, 15, inclusive="both"),
        "young_child": persons.age.between(0, 5, inclusive="both"),
    }

    # Calculate person types more efficiently
    conditions = [
        (
            age_ranges["adult"] & (persons.worker == 1) & (persons.student != 1)
        ),  # type 1
        (age_ranges["adult"] & (persons.student == 1)),  # type 3
        (
            age_ranges["working_age"] & (persons.worker != 1) & (persons.student != 1)
        ),  # type 4
        (
            age_ranges["senior"] & (persons.worker != 1) & (persons.student != 1)
        ),  # type 5
        age_ranges["teen"],  # type 6
        age_ranges["child"],  # type 7
        age_ranges["young_child"],  # type 8
    ]
    values = [1, 3, 4, 5, 6, 7, 8]
    persons["ptype"] = np.select(conditions, values, default=0)

    # Calculate employment status
    conditions = [
        ((persons.worker == 1) & (persons.age >= 16)),  # type 1
        ((persons.worker == 0) & (persons.age >= 16)),  # type 3
        (persons.age < 16),  # type 4
    ]
    values = [1, 3, 4]
    persons["pemploy"] = np.select(conditions, values, default=0)

    # Calculate student status
    conditions = [
        (persons.age <= 18),  # type 1
        ((persons.student == 1) & (persons.age > 18)),  # type 2
        (persons.student == 0),  # type 3
    ]
    values = [1, 2, 3]
    persons["pstudent"] = np.select(conditions, values, default=0)

    # Add home coordinates efficiently
    home_coords = (
        households[["block_id"]]
        .merge(blocks[["x", "y"]], left_on="block_id", right_index=True)
        .set_index(households.index)
    )
    persons["home_x"] = persons["household_id"].map(home_coords["x"])
    persons["home_y"] = persons["household_id"].map(home_coords["y"])

    # Convert location fields
    for field in ["workplace_taz", "school_taz"]:
        try:
            source_field = (
                "work_zone_id" if field == "workplace_taz" else "school_zone_id"
            )
            persons[field] = pd.to_numeric(
                persons[source_field], errors="coerce"
            ).fillna(-1)
        except KeyError:
            logger.info(f"Field `{field}` not present in input h5 file")

    # Clean numeric fields
    persons["worker"] = pd.to_numeric(persons["worker"], errors="coerce").fillna(0)
    persons["student"] = pd.to_numeric(persons["student"], errors="coerce").fillna(0)

    # Filter invalid records
    mask = ~persons[asim_zone_id_col].isnull() & (persons["age"] >= 1.0)

    if all(col in persons.columns for col in ["workplace_taz", "school_taz"]):
        mask &= ~((persons.worker == 1) & (persons.workplace_taz < 0))
        mask &= ~((persons.student == 1) & (persons.school_taz < 0))

    persons = persons.loc[mask].dropna().copy()

    # Reset member IDs
    persons["member_id"] = persons.groupby("household_id").cumcount() + 1

    # Clear school/work locations for specific person types
    workers_mask = persons["ptype"] == 1
    nonwork_mask = persons.ptype.isin([4, 5])

    # Clear school locations for workers
    if "school_taz" in persons.columns:
        before_count = (workers_mask & (persons.school_taz >= 0)).sum()
        persons.loc[workers_mask, ["school_taz", "school_zone_id"]] = -1
        after_count = (workers_mask & (persons.school_taz >= 0)).sum()
        logger.info(
            f"Workers with school location: {before_count} before, {after_count} after cleaning"
        )

    # Clear work/school locations for non-workers
    if all(col in persons.columns for col in ["workplace_taz", "school_taz"]):
        before_school = (nonwork_mask & (persons.school_taz > 0)).sum()
        before_work = (nonwork_mask & (persons.workplace_taz > 0)).sum()
        persons.loc[
            nonwork_mask,
            ["school_taz", "school_zone_id", "workplace_taz", "work_zone_id"],
        ] = -1
        after_school = (nonwork_mask & (persons.school_taz > 0)).sum()
        after_work = (nonwork_mask & (persons.workplace_taz > 0)).sum()

        logger.info(
            f"Non-workers/students with school location: {before_school} before, {after_school} after"
        )
        logger.info(
            f"Non-workers/students with work location: {before_work} before, {after_work} after"
        )

    return persons


def _update_households_table(
    households: pd.DataFrame, blocks: pd.DataFrame, asim_zone_id_col: str = "TAZ"
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Updates household attributes and assigns zones for ActivitySim processing.

    This function assigns ActivitySim zone IDs to households based on their
    `block_id` and the provided `blocks` DataFrame. It filters out households
    that cannot be assigned a valid zone, creates new column variables like
    `HHT` (household type), and ensures 'cars' is an integer type.

    Args:
        households (pd.DataFrame): DataFrame containing household records.
        blocks (pd.DataFrame): DataFrame containing block/zone information,
            including the `asim_zone_id_col`.
        asim_zone_id_col (str): Column name for the ActivitySim zone ID
            (e.g., 'TAZ'). Defaults to 'TAZ'.

    Returns:
        Tuple[pd.DataFrame, pd.Index]: A tuple containing:
            - pd.DataFrame: The updated and cleaned households DataFrame,
              with `household_id` set as the index.
            - pd.Index: An Index of household IDs that were dropped due to
              not having a valid TAZ.
    """
    # assign zones
    households.index = households.index.astype(int)
    households[asim_zone_id_col] = (
        blocks[asim_zone_id_col].reindex(households["block_id"]).values
    )
    hh_null_taz = (
        ~(households[asim_zone_id_col].astype(float).astype("Int64") > 0)
    ).fillna(True)

    households[asim_zone_id_col] = households[asim_zone_id_col].astype(str)
    logger.info("Dropping {0} households without TAZs".format(hh_null_taz.sum()))
    hh_null_taz_id = households.index[hh_null_taz]

    # CHANGE: Make explicit copy when filtering
    households = households[~hh_null_taz].copy()

    # create new column variables
    s = households.persons
    households.loc[:, "HHT"] = s.where(s == 1, 4)
    households["cars"] = households["cars"].astype(int)

    # clean up dataframe structure
    # TODO: move this to annotate_households.yaml in asim settings
    #     hh_names_dict = {
    #         'persons': 'PERSONS',
    #         'cars': 'VEHICL'}
    #     households = households.rename(columns=hh_names_dict)
    if "household_id" in households.columns:
        households.set_index("household_id", inplace=True)
    else:
        households.index.name = "household_id"

    return households, hh_null_taz_id


def _update_jobs_table(
    jobs: pd.DataFrame,
    blocks: pd.DataFrame,
    settings: PilatesConfig,
    local_crs: str,
    asim_zone_id_col: str = "TAZ",
    workspace: Optional["Workspace"] = None,
) -> Tuple[int, pd.DataFrame]:
    """
    Updates the jobs table by assigning ActivitySim zone IDs and reassigning
    jobs from blocks with no land area.

    This function ensures that all jobs are assigned to a valid ActivitySim
    zone. It also identifies jobs located in blocks with zero land area
    (which can cause issues with density calculations) and reassigns them
    to the nearest valid block with land area.

    Args:
        jobs (pd.DataFrame): DataFrame containing job records.
        blocks (pd.DataFrame): DataFrame containing block information,
            including 'block_id' and 'square_meters_land'.
        settings (PilatesConfig): The current PILATES settings object.
        local_crs (str): The local coordinate reference system string (e.g., "EPSG:3857").
        asim_zone_id_col (str): Column name for the ActivitySim zone ID
            (e.g., 'TAZ'). Defaults to 'TAZ'.
        workspace (Optional[Workspace]): The current workspace object.
            Required if `get_block_geoms` needs to be called.

    Returns:
        Tuple[int, pd.DataFrame]: A tuple containing:
            - int: The number of jobs that were reassigned.
            - pd.DataFrame: The updated jobs DataFrame.
    """
    # assign zones
    jobs[asim_zone_id_col] = blocks[asim_zone_id_col].reindex(jobs["block_id"]).values

    jobs[asim_zone_id_col] = jobs[asim_zone_id_col].astype(str)

    # make sure jobs are only assigned to blocks with land area > 0
    # so that employment density distributions don't contain Inf/NaN
    blocks = blocks[["square_meters_land"]]
    jobs["square_meters_land"] = blocks.reindex(jobs["block_id"])[
        "square_meters_land"
    ].values
    jobs_w_no_land = jobs[jobs["square_meters_land"] == 0]
    blocks_to_reassign = jobs_w_no_land["block_id"].unique()
    num_reassigned = len(blocks_to_reassign)

    if num_reassigned > 0:

        logger.info("Reassigning jobs out of blocks with no land area!")
        blocks_gdf = get_block_geoms(settings, workspace)
        blocks_gdf.set_index("GEOID", inplace=True)
        blocks_gdf["square_meters_land"] = blocks["square_meters_land"].reindex(
            blocks_gdf.index
        )
        blocks_gdf = blocks_gdf.to_crs(local_crs)

        for block_id in tqdm(
            blocks_to_reassign, desc="Redistributing jobs from blocks:"
        ):
            candidate_mask = (blocks_gdf.index.values != block_id) & (
                blocks_gdf["square_meters_land"] > 0
            )
            new_block_id = (
                blocks_gdf[candidate_mask]
                .distance(blocks_gdf.loc[block_id, "geometry"])
                .idxmin()
            )

            jobs.loc[jobs["block_id"] == block_id, "block_id"] = new_block_id

    else:
        logger.info("No block IDs to reassign in the jobs table!")

    return num_reassigned, jobs


def _update_blocks_table(
    settings: PilatesConfig,
    year: int,
    blocks: pd.DataFrame,
    households: pd.DataFrame,
    jobs: pd.DataFrame,
    zone_id_col: str,
    workspace: "Workspace",
) -> Tuple[bool, pd.DataFrame]:
    """
    Updates the blocks table with aggregated population, employment, and acreage
    information, and ensures correct zone ID mapping.

    This function calculates total employment (`TOTEMP`), total population (`TOTPOP`),
    and total acreage (`TOTACRE`) for each block based on the provided households
    and jobs data. It also ensures that the blocks table has the correct zone ID
    column (`zone_id_col`) mapped according to the configured zone type.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        year (int): The simulation year.
        blocks (pd.DataFrame): DataFrame containing block records.
        households (pd.DataFrame): DataFrame containing household records.
        jobs (pd.DataFrame): DataFrame containing job records.
        zone_id_col (str): The name of the column to use for zone IDs in the
            blocks table (e.g., 'TAZ_zone_id', 'block_group_zone_id').
        workspace (Workspace): The current workspace object.

    Returns:
        Tuple[bool, pd.DataFrame]: A tuple containing:
            - bool: `geoid_to_zone_mapping_updated`, True if the zone ID mapping
              was updated in this call, False otherwise.
            - pd.DataFrame: The updated blocks DataFrame.
    """
    blocks["TOTEMP"] = (
        jobs[["block_id", "sector_id"]]
        .groupby("block_id")["sector_id"]
        .count()
        .reindex(blocks.index)
        .fillna(0)
    )

    blocks["TOTPOP"] = (
        households[["block_id", "persons"]]
        .groupby("block_id")["persons"]
        .sum()
        .reindex(blocks.index)
        .fillna(0)
    )

    blocks["TOTACRE"] = blocks["square_meters_land"] / 4046.86

    # update blocks (should only have to be run if asim is loading
    # raw urbansim data that has yet to be touched by pilates)
    geoid_to_zone_mapping_updated = True

    zone_type = get_setting(settings, "shared.geography.zones.zone_type")
    zone_id_col = "{}_zone_id".format(zone_type)

    if zone_id_col not in blocks.columns:

        mapping = get_block_to_zone_mapping(settings, year, workspace)

        if zone_type == "block":
            logger.info("Mapping block IDs")
            blocks[zone_id_col] = blocks.index.astype(str).replace(mapping)

        elif zone_type == "block_group":
            logger.info("Mapping blocks to block group IDS")
            blocks[zone_id_col] = blocks.block_group_id.astype(str).replace(mapping)

        elif zone_type == "taz":
            logger.info("Mapping block IDs to TAZ")
            blocks[zone_id_col] = blocks.index.astype(str)
            blocks[zone_id_col] = blocks[zone_id_col].replace(mapping)

        geoid_to_zone_mapping_updated = True

    else:
        logger.info(
            "Blocks table already has zone IDs. Make sure skim zones "
            "haven't changed."
        )

    blocks[zone_id_col] = blocks[zone_id_col].astype(str)

    return geoid_to_zone_mapping_updated, blocks


def _create_land_use_table(
    settings: PilatesConfig,
    zones: pd.DataFrame,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    jobs: pd.DataFrame,
    blocks: pd.DataFrame,
    asim_zone_id_col: str = "TAZ",
) -> pd.DataFrame:
    """
    Creates the ActivitySim land use table by aggregating data from various
    sources (zones, households, persons, jobs, blocks, schools, colleges).

    This function consolidates demographic, employment, and enrollment data
    at the zone level, calculates various density metrics, and derives
    parking cost and area type classifications. The resulting DataFrame
    is suitable for use as the `land_use` input table in ActivitySim.

    Args:
        settings (PilatesConfig): The current PILATES settings object.
        zones (pd.DataFrame): A DataFrame containing base zone information.
            Expected to have 'TAZ' as its index.
        households (pd.DataFrame): A DataFrame containing household records.
        persons (pd.DataFrame): A DataFrame containing person records.
        jobs (pd.DataFrame): A DataFrame containing job records.
        blocks (pd.DataFrame): A DataFrame containing block-level information.
        asim_zone_id_col (str): The name of the column representing the
            ActivitySim zone ID (e.g., 'TAZ'). Defaults to 'TAZ'.

    Returns:
        pd.DataFrame: The comprehensive ActivitySim land use table.
    """
    logger.info("Creating land use table.")
    zone_type = get_setting(settings, "shared.geography.zones.zone_type")

    # DIAGNOSTIC LOGGING
    logger.info("=== DIAGNOSTIC INFO FOR LAND USE TABLE ===")
    logger.info(
        f"zones index: name={zones.index.name}, dtype={zones.index.dtype}, len={len(zones)}"
    )
    logger.info(f"zones index sample (first 10): {zones.index[:10].tolist()}")
    logger.info(
        f"persons[{asim_zone_id_col}]: dtype={persons[asim_zone_id_col].dtype if asim_zone_id_col in persons.columns else 'MISSING'}, nunique={persons[asim_zone_id_col].nunique() if asim_zone_id_col in persons.columns else 'N/A'}"
    )
    logger.info(
        f"persons[{asim_zone_id_col}] sample: {persons[asim_zone_id_col].unique()[:10].tolist() if asim_zone_id_col in persons.columns else 'N/A'}"
    )
    logger.info(
        f"households[{asim_zone_id_col}]: dtype={households[asim_zone_id_col].dtype if asim_zone_id_col in households.columns else 'MISSING'}, nunique={households[asim_zone_id_col].nunique() if asim_zone_id_col in households.columns else 'N/A'}"
    )
    logger.info(
        f"households[{asim_zone_id_col}] sample: {households[asim_zone_id_col].unique()[:10].tolist() if asim_zone_id_col in households.columns else 'N/A'}"
    )
    logger.info(
        f"jobs[{asim_zone_id_col}]: dtype={jobs[asim_zone_id_col].dtype if asim_zone_id_col in jobs.columns else 'MISSING'}, nunique={jobs[asim_zone_id_col].nunique() if asim_zone_id_col in jobs.columns else 'N/A'}"
    )
    logger.info(
        f"jobs[{asim_zone_id_col}] sample: {jobs[asim_zone_id_col].unique()[:10].tolist() if asim_zone_id_col in jobs.columns else 'N/A'}"
    )
    logger.info("==========================================")

    schools = enrollment_tables(
        settings, zones, enrollment_type="schools", asim_zone_id_col=asim_zone_id_col
    )
    colleges = enrollment_tables(
        settings, zones, enrollment_type="colleges", asim_zone_id_col=asim_zone_id_col
    )
    assert zones.index.name == "TAZ"
    assert zones.index.inferred_type == "string", "zone_id dtype should be str"
    for table in [households, persons, jobs, blocks, schools, colleges]:
        assert pd.api.types.is_string_dtype(
            table[asim_zone_id_col]
        ), "zone_id dtype in should be str"

    # create new column variables
    logger.info("Creating new columns in the land use table.")
    if zone_type != "taz":
        if "STATE" in zones.columns:
            zones.loc[:, "STATE"] = zones["STATE"].astype(str)
        else:
            zones.loc[:, "STATE"] = get_setting(settings, "shared.geography.FIPS")[
                "state"
            ]
        try:
            zones.loc[:, "COUNTY"] = zones["COUNTY"].astype(str)
        except:
            print("Skipping COUNTY")
        try:
            zones.loc[:, "TRACT"] = zones["TRACT"].astype(str)
        except:
            print("Skipping TRACT")
        try:
            zones.loc[:, "BLKGRP"] = zones["BLKGRP"].astype(str)
        except:
            print("Skipping BLKGRP")

    # --- Consolidated Persons Aggregation ---
    logger.info("Aggregating persons data.")
    persons_agg = (
        persons.assign(
            AGE0004=persons["age"].between(0, 4),
            AGE0519=persons["age"].between(5, 19),
            AGE2044=persons["age"].between(20, 44),
            AGE4564=persons["age"].between(45, 64),
            AGE64P=persons["age"] >= 65,
            AGE62P=persons["age"] >= 62,
        )
        .groupby(asim_zone_id_col)[
            ["AGE0004", "AGE0519", "AGE2044", "AGE4564", "AGE64P", "AGE62P"]
        ]
        .sum().fillna(0)
    )
    persons_agg["TOTPOP"] = persons.groupby(asim_zone_id_col).size()

    # --- Consolidated Households Aggregation ---
    logger.info("Aggregating households data.")
    households_agg = (
        households.assign(
            HHINCQ1=(households["income"] < 30000) | (households["income"].isna()),
            HHINCQ2=households["income"].between(30000, 59999),
            HHINCQ3=households["income"].between(60000, 99999),
            HHINCQ4=households["income"] >= 100000,
        )
        .groupby(asim_zone_id_col)[
            ["HHINCQ1", "HHINCQ2", "HHINCQ3", "HHINCQ4", "workers"]
        ]
        .sum().fillna(0)
    )
    households_agg["TOTHH"] = households.groupby(asim_zone_id_col).size()
    households_agg.rename(columns={"workers": "EMPRES"}, inplace=True)

    # --- Consolidated Jobs Aggregation ---
    logger.info("Aggregating jobs data.")
    jobs_agg = (
        jobs.assign(
            RETEMPN=jobs["sector_id"].isin(["44-45"]),
            FPSEMPN=jobs["sector_id"].isin(["52", "54"]),
            HEREMPN=jobs["sector_id"].isin(["61", "62", "71"]),
            AGREMPN=jobs["sector_id"].isin(["11"]),
            MWTEMPN=jobs["sector_id"].isin(["42", "31-33", "32", "48-49"]),
        )
        .groupby(asim_zone_id_col)[
            ["RETEMPN", "FPSEMPN", "HEREMPN", "AGREMPN", "MWTEMPN"]
        ]
        .sum().fillna(0)
    )
    jobs_agg["TOTEMP"] = jobs.groupby(asim_zone_id_col).size().fillna(0)

    # Calculate OTHEMPN from the aggregated sums
    sector_columns = ["RETEMPN", "FPSEMPN", "HEREMPN", "AGREMPN", "MWTEMPN"]
    jobs_agg["OTHEMPN"] = (jobs_agg["TOTEMP"] - jobs_agg[sector_columns].sum(axis=1)).fillna(0)

    # --- DIAGNOSTIC: Check aggregated data before join ---
    logger.info("=== PRE-JOIN DIAGNOSTICS ===")
    logger.info(
        f"persons_agg index: dtype={persons_agg.index.dtype}, len={len(persons_agg)}, sample={persons_agg.index[:5].tolist()}"
    )
    logger.info(
        f"households_agg index: dtype={households_agg.index.dtype}, len={len(households_agg)}, sample={households_agg.index[:5].tolist()}"
    )
    logger.info(
        f"jobs_agg index: dtype={jobs_agg.index.dtype}, len={len(jobs_agg)}, sample={jobs_agg.index[:5].tolist()}"
    )
    logger.info(
        f"zones index: dtype={zones.index.dtype}, len={len(zones)}, sample={zones.index[:5].tolist()}"
    )

    # Check overlap
    zones_ids = set(zones.index.astype(str))
    persons_ids = set(persons_agg.index.astype(str))
    households_ids = set(households_agg.index.astype(str))
    jobs_ids = set(jobs_agg.index.astype(str))

    logger.info(
        f"Zone IDs in zones but not in persons_agg: {len(zones_ids - persons_ids)}"
    )
    logger.info(
        f"Zone IDs in persons_agg but not in zones: {len(persons_ids - zones_ids)}"
    )
    if persons_ids - zones_ids:
        logger.info(f"  Examples: {list(persons_ids - zones_ids)[:10]}")
    logger.info("============================")

    # --- Join all aggregated data to the zones table ---
    logger.info("Joining aggregated data to zones.")
    zones = zones.join([persons_agg, households_agg, jobs_agg])

    # --- DIAGNOSTIC: Check result after join ---
    logger.info("=== POST-JOIN DIAGNOSTICS ===")
    logger.info(f"zones.TOTPOP.sum() = {zones.TOTPOP.sum()}")
    logger.info(f"zones.TOTHH.sum() = {zones.TOTHH.sum()}")
    logger.info(f"zones.TOTEMP.sum() = {zones.TOTEMP.sum()}")
    logger.info(f"Number of zones with TOTPOP > 0: {(zones.TOTPOP > 0).sum()}")
    logger.info("=============================")

    zones.loc[:, "SHPOP62P"] = (
        (zones.AGE62P / zones.TOTPOP).reindex(zones.index).fillna(0)
    )

    zones.loc[:, "TOTACRE"] = (
        blocks[["TOTACRE", asim_zone_id_col]]
        .groupby(asim_zone_id_col)["TOTACRE"]
        .sum()
        .reindex(zones.index)
        .fillna(0)
    )
    zones.loc[:, "HSENROLL"] = (
        schools[["enrollment", asim_zone_id_col]]
        .groupby(asim_zone_id_col)["enrollment"]
        .sum()
        .reindex(zones.index)
        .fillna(0)
    )
    zones.loc[:, "TOPOLOGY"] = 1  # FIXME
    zones.loc[:, "employment_density"] = (zones.TOTEMP / zones.TOTACRE).fillna(0.0)
    zones.loc[:, "pop_density"] = (zones.TOTPOP / zones.TOTACRE).fillna(0.0)
    zones.loc[:, "hh_density"] = (zones.TOTHH / zones.TOTACRE).fillna(0.0)
    zones.loc[:, "hq1_density"] = (zones.HHINCQ1 / zones.TOTACRE).fillna(0.0)
    zones.loc[:, "PRKCST"] = _get_park_cost(
        zones,
        [-1.92168743, 4.89511403, 4.2772001, 0.65784643],
        ["pop_density", "hh_density", "hq1_density", "employment_density"],
        ["employment_density", "pop_density", "hh_density", "hq1_density"],
    )
    zones.loc[:, "OPRKCST"] = _get_park_cost(
        zones,
        [-6.17833544, 17.55155703, 2.0786466],
        ["pop_density", "hh_density", "employment_density"],
        ["employment_density", "pop_density", "hh_density"],
    )
    zones.loc[:, "COLLFTE"] = (
        colleges[[asim_zone_id_col, "full_time_enrollment"]]
        .groupby(asim_zone_id_col)["full_time_enrollment"]
        .sum()
        .reindex(zones.index)
        .fillna(0)
    )
    zones.loc[:, "COLLPTE"] = (
        colleges[[asim_zone_id_col, "part_time_enrollment"]]
        .groupby(asim_zone_id_col)["part_time_enrollment"]
        .sum()
        .reindex(zones.index)
        .fillna(0)
    )
    zones.loc[:, "TERMINAL"] = 0
    zones.loc[:, "area_type_metric"] = _compute_area_type_metric(zones)
    zones.loc[:, "area_type"] = _compute_area_type(zones)
    zones.loc[:, "TERMINAL"] = 0  # FIXME
    zones.loc[:, "COUNTY"] = 1  # FIXME

    logger.info(zones.head())
    logger.info(zones.dtypes)

    for col in zones.columns:
        try:
            zones[col] = zones[col].fillna(0.0)
        except Exception as e:
            logger.info(f"Generated exception when trying to fillna in zones: {e}")

    return zones


def create_asim_data_from_database(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    provenance_tracker: "FileProvenanceTracker",
    model_run_hash: Optional[str] = None,
) -> List[FileRecord]:
    """
    Create ActivitySim input data from database records using dual storage.

    This function queries the database for pre-extracted ActivitySim input tables
    that were uploaded using the h5_to_database utility with dual storage support.
    It can use either processed ActivitySim data or reprocess from raw UrbanSim data.

    Args:
        settings (PilatesConfig): PILATES settings dictionary.
        state (WorkflowState): Workflow state.
        workspace (Workspace): Workspace instance.
        provenance_tracker (FileProvenanceTracker): For tracking provenance.
        model_run_hash (Optional[str]): Current model run hash.

    Returns:
        List[FileRecord]: List of FileRecord objects for the created CSV files.

    Raises:
        ValueError: If the database is not configured but database input mode is enabled.
        AssertionError: If the database manager is not a DuckDBManager (current limitation).
    """
    """
    Create ActivitySim input data from database records using dual storage.

    This function queries the database for pre-extracted ActivitySim input tables
    that were uploaded using the h5_to_database utility with dual storage support.
    It can use either processed ActivitySim data or reprocess from raw UrbanSim data.

    Args:
        settings: PILATES settings dictionary
        state: Workflow state
        workspace: Workspace instance
        provenance_tracker: For tracking provenance
        model_run_hash: Current model run hash

    Returns:
        List of FileRecord objects for the created CSV files
    """
    logger.info("🗄️ Creating ActivitySim input data from database (dual storage)")

    # Create database manager
    db_manager = create_database_manager(settings.shared.database)
    if not db_manager:
        raise ValueError(
            "Database is not configured but database input mode is enabled"
        )

    output_dir = workspace.get_asim_mutable_data_dir()
    output_records = []

    # Standard ActivitySim input tables we expect to find
    required_tables = ["households", "persons", "land_use"]
    optional_tables = ["accessibility", "zones", "maz", "taz"]

    # Database configuration for ActivitySim inputs
    db_config = settings.activitysim.database
    # config_hash = db_config.config_hash  # Optional: match specific config
    year = db_config.year  # Year to query for
    use_processed = db_config.use_processed_data

    logger.info("📊 Database query parameters:")
    # logger.info(f"   Config hash: {config_hash or 'any'}")
    config_hash = None
    logger.info(f"   Year: {year}")
    logger.info(
        f"   Data source: {'processed ActivitySim' if use_processed else 'raw UrbanSim (will reprocess)'}"
    )

    try:
        with db_manager:
            assert (
                isinstance(db_manager, DuckDBManager),
                "This doesn't work with databases beyond duckdb yet",
            )
            # Create ActivitySim input CSV files from database
            tables_created = 0

            # Process each required table
            for table_name in required_tables:
                logger.info(f"🔍 Processing {table_name} table...")

                if use_processed:
                    # Use pre-processed ActivitySim data (fast path)
                    data_df = db_manager.retrieve_activitysim_data(
                        table_name=table_name, config_hash=config_hash, year=year
                    )
                    data_source = "processed ActivitySim"
                else:
                    # Use raw UrbanSim data (reproducible path - would need reprocessing)
                    data_df = db_manager.retrieve_urbansim_raw_data(
                        table_name=table_name, config_hash=config_hash, year=year
                    )
                    data_source = "raw UrbanSim"

                    # TODO: Add reprocessing logic here when needed
                    if data_df is not None:
                        logger.warning(
                            f"⚠️  Raw data found but reprocessing not yet implemented for {table_name}"
                        )
                        logger.warning(
                            "   Consider using use_processed_data: true in activitysim_database config"
                        )
                        data_df = None

                if data_df is not None and not data_df.empty:
                    # Create CSV file for ActivitySim
                    output_csv_path = os.path.join(output_dir, f"{table_name}.csv")

                    # Clean and format the data for ActivitySim
                    clean_df = _clean_activitysim_data_for_csv(data_df, table_name)
                    clean_df.to_csv(
                        output_csv_path, index=True
                    )  # ActivitySim expects indexes

                    logger.info(
                        f"✅ Created {table_name}.csv with {len(clean_df)} records from {data_source}"
                    )

                    # Record as output
                    output_record = provenance_tracker.record_output_file(
                        "activitysim_preprocessor",
                        output_csv_path,
                        short_name=f"{table_name}_asim_in",
                        description=f"ActivitySim {table_name} input from database ({data_source})",
                        model_run_id=model_run_hash,
                    )

                    if output_record:
                        output_records.append(output_record)
                        tables_created += 1

                else:
                    logger.error(f"❌ No {table_name} data found in database")

            # Process optional tables
            for table_name in optional_tables:
                logger.info(f"🔍 Processing optional {table_name} table...")

                if use_processed:
                    data_df = db_manager.retrieve_activitysim_data(
                        table_name=table_name, config_hash=config_hash, year=year
                    )
                else:
                    # Skip optional tables for raw mode for now
                    data_df = None

                if data_df is not None and not data_df.empty:
                    output_csv_path = os.path.join(output_dir, f"{table_name}.csv")
                    clean_df = _clean_activitysim_data_for_csv(data_df, table_name)
                    clean_df.to_csv(output_csv_path, index=True)

                    logger.info(
                        f"✅ Created optional {table_name}.csv with {len(clean_df)} records"
                    )

                    output_record = provenance_tracker.record_output_file(
                        "activitysim_preprocessor",
                        output_csv_path,
                        short_name=f"{table_name}_asim_in",
                        description=f"ActivitySim {table_name} input from database",
                        model_run_id=model_run_hash,
                    )

                    if output_record:
                        output_records.append(output_record)
                        tables_created += 1
                else:
                    logger.info(
                        f"ℹ️  Optional table {table_name} not found in database (skipping)"
                    )

            logger.info(
                f"🎉 Successfully created {tables_created} ActivitySim input files from database"
            )

            if tables_created < len(required_tables):
                logger.warning(
                    f"⚠️  Only {tables_created}/{len(required_tables)} required tables were created successfully"
                )

            return output_records

    except Exception as e:
        logger.error(f"Failed to create ActivitySim data from database: {e}")
        raise


def create_asim_data_from_h5(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    provenance_tracker: "FileProvenanceTracker",
    model_run_hash: Optional[str] = None,
) -> List[FileRecord]:
    """
    Create ActivitySim input data from UrbanSim H5 outputs.

    This function reads household, person, job, and block data from UrbanSim's
    H5 output files, processes them to be compatible with ActivitySim's input
    requirements, and then generates the necessary CSV files (households.csv,
    persons.csv, land_use.csv). All generated files are tracked for provenance.

    Args:
        settings (PilatesConfig): PILATES settings dictionary.
        state (WorkflowState): Workflow state.
        workspace (Workspace): Workspace instance.
        provenance_tracker (FileProvenanceTracker): For tracking provenance.
        model_run_hash (Optional[str]): Current model run hash.

    Returns:
        List[FileRecord]: List of FileRecord objects for the created CSV files.
    """
    """
    Create ActivitySim input data from UrbanSim H5 outputs.

    Args:
        settings: PILATES settings dictionary
        state: Workflow state
        workspace: Workspace instance
        provenance_tracker: For tracking provenance
        model_run_hash: Current model run hash

    Returns:
        List of FileRecord objects for the created CSV files
    """
    logger.info("Creating ActivitySim input data from UrbanSim H5")

    output_dir = workspace.get_asim_mutable_data_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_records = []

    # Load canonical zones using the workspace-aware function
    zones = load_canonical_zones(settings, workspace)
    asim_zone_id_col = "TAZ"

    # Get block to zone mapping
    block_to_zone_map = get_block_to_zone_mapping(settings, state.year, workspace)

    # Load UrbanSim data
    usim_data_path = get_merged_usim_input_datastore_path(
        settings, workspace.get_usim_mutable_data_dir()
    )
    usim_data_in = provenance_tracker.record_input_file(
        "activitysim_preprocessor",
        usim_data_path,
        short_name="usim_data",
        description="UrbanSim H5 data",
        model_run_id=model_run_hash,
    )

    # Read tables from UrbanSim H5
    store, prefix = read_datastore(settings, state.start_year)
    households = store[prefix + "/households"]
    persons = store[prefix + "/persons"]
    jobs = store[prefix + "/jobs"]
    blocks = store[prefix + "/blocks"]

    # Add zone id to blocks table
    blocks[asim_zone_id_col] = blocks.index.map(block_to_zone_map)

    # Update tables
    households, unassigned_households = _update_households_table(
        households, blocks, asim_zone_id_col
    )
    persons = _update_persons_table(
        persons, households, unassigned_households, blocks, asim_zone_id_col
    )
    num_reassigned_jobs, jobs = _update_jobs_table(
        jobs,
        blocks,
        settings,
        settings.shared.geography.local_crs,
        asim_zone_id_col,
        workspace,
    )
    geoid_to_zone_mapping_updated, blocks = _update_blocks_table(
        settings, state.year, blocks, households, jobs, asim_zone_id_col, workspace
    )

    # Create land use table
    land_use = _create_land_use_table(
        settings, zones, households, persons, jobs, blocks, asim_zone_id_col
    )

    # Write to CSV and record provenance
    for df, name in [
        (households, "households"),
        (persons, "persons"),
        (land_use, "land_use"),
    ]:
        output_csv_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(output_csv_path, index=True)
        output_records.append(
            provenance_tracker.record_output_file(
                "activitysim_preprocessor",
                output_csv_path,
                short_name=f"{name}_asim_in",
                description=f"ActivitySim {name} input from UrbanSim H5",
                model_run_id=model_run_hash,
            )
        )

    return output_records
