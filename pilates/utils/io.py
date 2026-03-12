"""
Input/Output utility functions for PILATES.

This module provides functions for reading and parsing configuration settings,
determining data store paths, and locating various input/output files
for ActivitySim, UrbanSim, and BEAM models.
"""

import argparse
import logging
import os
import pandas as pd
from typing import Any, Dict, Optional, Tuple

from pilates.utils.settings_helper import get as get_setting
from pilates.config.models import load_config, PilatesConfig

logger = logging.getLogger(__name__)


def get_traffic_assignment_model(settings: Any) -> Optional[str]:
    """
    Resolve the configured traffic-assignment model across legacy/new schemas.
    """
    def _resolve_path(obj: Any, path: str) -> Optional[str]:
        current = obj
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                return None
        return current

    return (
        _resolve_path(settings, "run.models.traffic_assignment")
        or _resolve_path(settings, "run.models.travel")
        or getattr(settings, "travel_model", None)
    )


def compute_model_enabled_flags(settings: Any) -> Dict[str, bool]:
    """
    Compute which models are enabled based on the provided settings configuration.

    This function determines the operational status of various simulation models
    (land use, vehicle ownership, activity demand, traffic assignment, and replanning)
    by inspecting the PILATES settings object. It supports both legacy (flat) and
    new (nested Pydantic) configuration formats for model names.

    Args:
        settings (Any): The PILATES settings object, which can be a Pydantic model
            instance or a dictionary-like object containing configuration parameters.

    Returns:
        Dict[str, bool]: A dictionary where keys are model-related flags and
            values are booleans indicating whether the respective model/feature is enabled.
            Keys include:
            - "land_use_enabled": True if a land use model is specified and warm start skims are not enabled.
            - "vehicle_ownership_model_enabled": True if a vehicle ownership model is specified.
            - "activity_demand_enabled": True if an activity demand model is specified.
            - "traffic_assignment_enabled": True if a travel model is specified and static skims are not enabled.
            - "replanning_enabled": True if replanning iterations are greater than 0,
              unless the activity demand model is "polaris".
    """
    # Get model names from nested (Pydantic) or legacy (flat) config locations.
    # The `get_setting` utility handles this by trying the nested path first, then the flat path.
    land_use_model = get_setting(
        settings, "run.models.land_use", default=get_setting(settings, "land_use_model")
    )
    vehicle_ownership_model = get_setting(
        settings,
        "run.models.vehicle_ownership",
        default=get_setting(settings, "vehicle_ownership_model"),
    )
    activity_demand_model = get_setting(
        settings,
        "run.models.activity_demand",
        default=get_setting(settings, "activity_demand_model"),
    )
    travel_model = get_traffic_assignment_model(settings)

    # Retrieve flags related to skim processing and replanning iterations.
    warm_start_skims = get_setting(settings, "warm_start_skims", False)
    static_skims = get_setting(settings, "static_skims", False)
    replan_iters = get_setting(
        settings,
        "activitysim.replan_iters",
        default=get_setting(settings, "replan_iters", 0),
    )

    # Compute enabled flags based on model presence and other settings.
    # Land use model is enabled if a model is specified and warm start skims are not used.
    land_use_enabled = bool(land_use_model) and (not warm_start_skims)

    # Vehicle ownership model is enabled if a model is specified.
    vehicle_ownership_model_enabled = bool(vehicle_ownership_model)

    # Activity demand model is enabled if a model is specified.
    activity_demand_enabled = bool(activity_demand_model)

    # Traffic assignment is enabled if a travel model is specified and static skims are not used.
    traffic_assignment_enabled = bool(travel_model) and (not static_skims)

    # Replanning is enabled if replan_iters is greater than 0.
    replanning_enabled = replan_iters > 0

    # Special condition: if ActivitySim is using Polaris, replanning is disabled.
    if activity_demand_enabled and activity_demand_model == "polaris":
        replanning_enabled = False

    # Return a dictionary of all computed enabled flags.
    return {
        "land_use_enabled": land_use_enabled,
        "vehicle_ownership_model_enabled": vehicle_ownership_model_enabled,
        "activity_demand_enabled": activity_demand_enabled,
        "traffic_assignment_enabled": traffic_assignment_enabled,
        "replanning_enabled": replanning_enabled,
    }


def parse_args_and_settings(settings_file: str = "settings.yaml") -> PilatesConfig:
    """
    Parses command-line arguments and loads the PILATES settings configuration.

    This function sets up an argument parser to handle command-line options
    for specifying the settings file and the current stage file. It then
    loads the main settings from the specified YAML file into a Pydantic model,
    attaches runtime-specific file paths, and computes model enabled flags.
    It also performs validation checks for conflicting settings.

    Args:
        settings_file (str): The default name of the settings YAML file.
            Defaults to "settings.yaml".

    Returns:
        settings (PilatesConfig): The loaded and configured PILATES settings object (Pydantic model).

    Raises:
        ValueError: If conflicting settings are detected (e.g., land use models
            enabled with a non-zero household sample size, or invalid atlas_beamac
            configuration for the region/zone type).
    """
    # Initialize argument parser.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-c", "--config", default="settings.yaml", help="config file name"
    )
    parser.add_argument(
        "-S",
        "--stage",
        default=None,
        help="current state file to pick up from",
    )
    parser.add_argument(
        "--allow-rewind-resume",
        action="store_true",
        help="allow resuming from an earlier year than the archive run state",
    )
    args = parser.parse_args()

    # Override default settings_file if provided via command-line argument.
    if args.config:
        settings_file = args.config

    stage_file_loc = args.stage
    if args.stage is not None and not os.path.exists(stage_file_loc):
        raise FileNotFoundError(
            f"Explicit stage file provided via -S/--stage does not exist: {stage_file_loc}"
        )

    # Load settings from the specified config file into a Pydantic model for validation and structured access.
    settings = load_config(settings_file)

    # Attach runtime-specific file paths to the settings object for easy access later.
    settings.state_file_loc = stage_file_loc
    settings.settings_file = settings_file
    settings.allow_rewind_resume = args.allow_rewind_resume is True

    # Compute and attach model enabled flags to the settings object.
    enabled_flags = compute_model_enabled_flags(settings)
    settings.land_use_enabled = enabled_flags["land_use_enabled"]
    settings.vehicle_ownership_model_enabled = enabled_flags[
        "vehicle_ownership_model_enabled"
    ]
    settings.activity_demand_enabled = enabled_flags["activity_demand_enabled"]
    settings.traffic_assignment_enabled = enabled_flags["traffic_assignment_enabled"]
    settings.replanning_enabled = enabled_flags["replanning_enabled"]

    # Raise errors/warnings for conflicting settings to ensure valid simulation configurations.
    household_sample_size = get_setting(
        settings, "activitysim.household_sample_size", 0
    )

    if (household_sample_size > 0) and enabled_flags["land_use_enabled"]:
        raise ValueError(
            f'Land use models must be disabled (explicitly or via "warm '
            f'start" mode to use a non-zero household sample size. The '
            f"household sample size you specified is {household_sample_size}"
        )

    atlas_beamac = get_setting(settings, "atlas.beamac", 0)
    region = get_setting(settings, "run.region")
    skims_zone_type = get_setting(settings, "shared.geography.zones.zone_type")

    if (atlas_beamac > 0) and ((region != "sfbay") or (skims_zone_type != "taz")):
        raise ValueError(
            "atlas_beamac must be 0 (read accessibility from RData) "
            "unless region = sfbay and skims_zone_type = taz. When"
            "atlas_beamac = 1, accessibility is calculated internally. "
        )

    logger.info(f"Loading config with Pydantic validation: {settings_file}")
    pydantic_config = load_config(
        settings_file
    )  # Reload config to ensure Pydantic model is fully initialized
    logger.info("✓ Config validated successfully!")
    logger.info(f"  Region: {pydantic_config.run.region}")
    logger.info(
        f"  Years: {pydantic_config.run.start_year}-{pydantic_config.run.end_year}"
    )
    logger.info(f"  Enabled models: {pydantic_config.get_enabled_models()}")

    return settings


def datastore_path(
    settings: Any, year: Optional[int] = None, mutable_data_dir: Optional[str] = None
) -> str:
    """
    Constructs and returns the file path to the UrbanSim H5 data store.

    The path can be for either the base year input data store or a forecast
    year output data store, depending on the `year` parameter.

    Args:
        settings (Any): The PILATES settings object.
        year (Optional[int]): The simulation year for which to get the datastore path.
            If None, the path for the base year input datastore is returned.
            If an integer year is provided, the path for the forecast year
            output datastore is returned. Defaults to None.
        mutable_data_dir (Optional[str]): An optional directory to use as the
            base for the data store. If None, it defaults to the path specified
            in `settings.urbansim.local_data_input_folder`. Defaults to None.

    Returns:
        str: The absolute file path to the UrbanSim H5 data store.
    """
    region = get_setting(settings, "run.region")
    region_id_map = get_setting(
        settings, "urbansim.region_mappings.region_to_region_id", {}
    )
    region_id = region_id_map.get(region)

    # Determine the base data location. If mutable_data_dir is provided, use it;
    # otherwise, use the default from settings.
    if mutable_data_dir is None:
        data_loc = get_setting(settings, "urbansim.local_data_input_folder")
    else:
        data_loc = mutable_data_dir
        os.makedirs(data_loc, exist_ok=True)  # Ensure the directory exists.

    # Construct the datastore filename based on whether a specific year is provided.
    if year is None:
        # For base year, use the input file template.
        usim_datastore = get_setting(settings, "urbansim.input_file_template").format(
            region_id=region_id
        )
    else:
        # For a specific forecast year, use the output file template.
        usim_datastore = get_setting(settings, "urbansim.output_file_template").format(
            year=year
        )

    # Combine the data location and the datastore filename to get the full path.
    return os.path.join(data_loc, usim_datastore)


def read_datastore(
    settings: Any,
    year: Optional[int] = None,
    warm_start: bool = False,
    mutable_data_dir: Optional[str] = None,
    mode: str = "r",
) -> Tuple[pd.HDFStore, str]:
    """
    Opens and returns an UrbanSim H5 data store (pd.HDFStore) along with its table prefix.

    This function intelligently determines which H5 file to open based on the
    simulation year, warm start setting, and whether UrbanSim is enabled. It
    prioritizes year-specific input files if they exist, falling back to a
    base input file or an UrbanSim output file.

    Args:
        settings (Any): The PILATES settings object.
        year (Optional[int]): The current simulation year. If None, it implies
            reading the base input datastore. Defaults to None.
        warm_start (bool): If True, indicates a warm start scenario, which
            might influence which datastore is read. Defaults to False.
        mutable_data_dir (Optional[str]): An optional directory to use as the
            base for the data store. If None, it defaults to the path specified
            in `settings.urbansim.local_data_input_folder`. Defaults to None.
        mode (str): HDFStore open mode. Defaults to read-only ("r").

    Returns:
        Tuple[pd.HDFStore, str]: A tuple containing:
            - pd.HDFStore: An opened pandas HDFStore object for the UrbanSim data.
            - str: The table prefix (e.g., year string) to be used when accessing
              tables within the HDFStore.

    Raises:
        ValueError: If no land use data is found at the constructed path.
        KeyError: If expected tables (e.g., 'households') are not found in the
            opened HDFStore.
    """
    region = get_setting(settings, "run.region")
    region_id_map = get_setting(
        settings, "urbansim.region_mappings.region_to_region_id", {}
    )
    region_id = region_id_map.get(region)

    if mutable_data_dir is None:
        data_loc = get_setting(settings, "urbansim.local_data_input_folder")
    else:
        data_loc = mutable_data_dir

    urbansim_enabled = get_setting(settings, "run.models.land_use") is not None
    start_year = get_setting(settings, "run.start_year")

    # Determine which H5 file to read based on year, warm_start, and UrbanSim enablement.
    if (year == start_year) or warm_start or not urbansim_enabled:
        logger.info(
            f"Attempting to read input datastore: Year {year}, start year {start_year}, warm_start {warm_start}, urbansim_enabled {urbansim_enabled}"
        )

        # Try year-specific input file first if available and configured.
        year_template = get_setting(settings, "urbansim.input_file_template_year", None)
        use_year_specific = False

        if year_template:
            usim_datastore_year = year_template.format(
                region_id=region_id, year=year, start_year=year
            )
            usim_datastore_year_fpath = os.path.join(data_loc, usim_datastore_year)

            if os.path.exists(usim_datastore_year_fpath):
                logger.info(f"Using year-specific input file: {usim_datastore_year}")
                store = pd.HDFStore(usim_datastore_year_fpath, mode=mode)
                # Year-specific files typically have year-prefixed tables (e.g., '2018/households').
                table_prefix_yr = str(year)
                if f"{table_prefix_yr}/households" not in store:
                    # Fallback: if year-prefix not found, check if tables are at the root level.
                    if "households" in store:
                        table_prefix_yr = ""
                    else:
                        raise KeyError(
                            f"No households table found in year-specific file {usim_datastore_year_fpath}. Tables: {store.keys()}"
                        )
                usim_datastore = usim_datastore_year  # For logging purposes.
                usim_datastore_fpath = usim_datastore_year_fpath
                use_year_specific = True
            else:
                logger.info(
                    f"Year-specific file not found: {usim_datastore_year_fpath}"
                )
                logger.info("Falling back to base input file")

        # Fall back to base input file if year-specific doesn't exist or isn't configured.
        if not use_year_specific:
            table_prefix_yr = (
                ""  # Base input data store tables usually have no year prefix.
            )
            usim_datastore = get_setting(
                settings, "urbansim.input_file_template"
            ).format(region_id=region_id)
            usim_datastore_fpath = os.path.join(data_loc, usim_datastore)
            store = pd.HDFStore(usim_datastore_fpath, mode=mode)
            # Check for households table with and without year prefix.
            if "households" not in store:
                table_prefix_yr = str(year)
                if f"{table_prefix_yr}/households" not in store:
                    raise KeyError(
                        f"No households table of either format found in {usim_datastore_fpath}. Tables: {store.keys()}"
                    )
                else:
                    logger.info(f"Using {table_prefix_yr}/households table")

    # Otherwise, if not base year, warm start, or UrbanSim enabled, read from UrbanSim outputs.
    else:
        usim_datastore = get_setting(settings, "urbansim.output_file_template").format(
            year=year
        )
        table_prefix_yr = str(
            year
        )  # UrbanSim output tables are typically year-prefixed.
        usim_datastore_fpath = os.path.join(data_loc, usim_datastore)
        store = pd.HDFStore(usim_datastore_fpath, mode=mode)

    logger.info(f"Opening urbansim datastore at {usim_datastore_fpath}")

    if not os.path.exists(usim_datastore_fpath):
        raise ValueError(f"No land use data found at {usim_datastore_fpath}!")

    return store, table_prefix_yr


def get_merged_usim_input_datastore_path(
    settings: Any, mutable_data_dir: Optional[str] = None
) -> str:
    """
    Constructs and returns the full file path to the merged UrbanSim input datastore.

    This path is typically used for ActivitySim to read UrbanSim's base year
    data. It uses the region and a configured input file template to form the
    filename.

    Args:
        settings (Any): The PILATES settings object.
        mutable_data_dir (Optional[str]): An optional directory to use as the
            base for the data store. If None, it defaults to the path specified
            in `settings.urbansim.local_data_input_folder`. Defaults to None.

    Returns:
        str: The full file path to the merged UrbanSim input datastore.
    """
    region = get_setting(settings, "run.region")
    region_id_map = get_setting(
        settings, "urbansim.region_mappings.region_to_region_id", {}
    )
    region_id = region_id_map.get(region)
    usim_base_fname = get_setting(settings, "urbansim.input_file_template")
    # Format the base filename with the region ID.
    datastore_name = usim_base_fname.format(region_id=region_id)
    # Determine the data location, prioritizing mutable_data_dir if provided.
    data_loc = (
        mutable_data_dir
        if mutable_data_dir
        else get_setting(settings, "urbansim.local_data_input_folder")
    )
    # Combine the data location and the datastore name to get the full path.
    return os.path.join(data_loc, datastore_name)


def locate_asim_file(
    asim_output_data_dir: str, file_name: str, fmt: Optional[str]
) -> str:
    """
    Constructs the full path to an ActivitySim output file based on its format.

    ActivitySim outputs can be in CSV or Parquet format, and their naming
    conventions differ. This function abstracts away these differences to
    provide the correct file path.

    Args:
        asim_output_data_dir (str): The base directory where ActivitySim
            stores its output data.
        file_name (str): The logical name of the file (e.g., "households", "persons", "plans").
        fmt (Optional[str]): The format of the file ("csv", "parquet", or None).
            - If "csv", expects a file named "final_<file_name>.csv".
            - If "parquet", expects a file within a "final_pipeline" subdirectory.
            - If None, assumes the file is directly named `file_name` within the
              output directory.

    Returns:
        str: The full, absolute path to the ActivitySim output file.

    Raises:
        ValueError: If an unsupported file format (`fmt`) is provided.
    """
    if fmt == "csv":
        # CSV files are typically named "final_<file_name>.csv" directly in the output directory.
        return os.path.join(asim_output_data_dir, "final_" + file_name + ".csv")
    elif fmt == "parquet":
        # Parquet files have a more complex structure, often nested within "final_pipeline".
        # Special handling for "plans" file name.
        if file_name == "plans":
            a_n = "beam_plans"
        else:
            a_n = file_name
        return os.path.join(
            asim_output_data_dir, "final_pipeline", a_n, "final.parquet"
        )
    elif fmt is None:
        # If no format is specified, assume the file is directly named `file_name`.
        return os.path.join(asim_output_data_dir, file_name)
    else:
        # Raise an error for unsupported formats.
        raise ValueError(f"Unsupported file format: {fmt}")


def locate_beam_file(beam_scenario_folder: str, file_name: str, fmt: str) -> str:
    """
    Constructs the full path to a BEAM output file based on its format.

    BEAM output files can be in CSV (gzipped) or Parquet format. This function
    provides the correct file path based on the specified format.

    Args:
        beam_scenario_folder (str): The base directory for the BEAM scenario outputs.
        file_name (str): The logical name of the file (e.g., "events", "households").
        fmt (str): The format of the file ("csv" or "parquet").
            - If "csv", expects a file named "<file_name>.csv.gz".
            - If "parquet", expects a file named "<file_name>.parquet".

    Returns:
        str: The full, absolute path to the BEAM output file.

    Raises:
        ValueError: If an unsupported file format (`fmt`) is provided.
    """
    if fmt == "csv":
        # CSV files from BEAM are typically gzipped and named "<file_name>.csv.gz".
        return os.path.join(beam_scenario_folder, file_name + ".csv.gz")
    elif fmt == "parquet":
        # Parquet files from BEAM are typically named "<file_name>.parquet".
        return os.path.join(beam_scenario_folder, file_name + ".parquet")
    else:
        # Raise an error for unsupported formats.
        raise ValueError(f"Unsupported file format: {fmt}")
