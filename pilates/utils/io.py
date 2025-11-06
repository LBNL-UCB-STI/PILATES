import argparse
import logging
import os
import pandas as pd
import yaml

from pilates.utils.settings_helper import get as get_setting
from pilates.config.models import load_config

logger = logging.getLogger(__name__)


def compute_model_enabled_flags(settings):
    """
    Compute which models are enabled based on config.

    Works with both legacy (flat) and new (nested Pydantic) config formats.

    Args:
        settings: Settings object (Pydantic model or dict)

    Returns:
        dict: Flags for land_use_enabled, vehicle_ownership_model_enabled, etc.
    """
    # Get model names from nested or legacy locations
    land_use_model = get_setting(settings, "run.models.land_use", default=get_setting(settings, "land_use_model"))
    vehicle_ownership_model = get_setting(settings, "run.models.vehicle_ownership", default=get_setting(settings, "vehicle_ownership_model"))
    activity_demand_model = get_setting(settings, "run.models.activity_demand", default=get_setting(settings, "activity_demand_model"))
    travel_model = get_setting(settings, "run.models.travel", default=get_setting(settings, "travel_model"))

    # These flags are now read directly from settings, not CLI args.
    warm_start_skims = get_setting(settings, "warm_start_skims", False)
    static_skims = get_setting(settings, "static_skims", False)
    replan_iters = get_setting(settings, "activitysim.replan_iters", default=get_setting(settings, "replan_iters", 0))

    # Compute enabled flags
    land_use_enabled = (
        bool(land_use_model)
        and (not warm_start_skims)
    )

    vehicle_ownership_model_enabled = bool(vehicle_ownership_model)

    activity_demand_enabled = bool(activity_demand_model)

    traffic_assignment_enabled = (
        bool(travel_model)
        and (not static_skims)
    )

    replanning_enabled = replan_iters > 0

    if activity_demand_enabled and activity_demand_model == "polaris":
        replanning_enabled = False

    return {
        "land_use_enabled": land_use_enabled,
        "vehicle_ownership_model_enabled": vehicle_ownership_model_enabled,
        "activity_demand_enabled": activity_demand_enabled,
        "traffic_assignment_enabled": traffic_assignment_enabled,
        "replanning_enabled": replanning_enabled,
    }


def parse_args_and_settings(settings_file="settings.yaml"):
    # parse command-line args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-c", "--config", default="settings.yaml", help="config file name"
    )
    parser.add_argument(
        "-S",
        "--stage",
        default="current_stage.yaml",
        help="current state file to pick up from",
    )
    args = parser.parse_args()

    if args.config:
        settings_file = args.config

    # read settings from config file and load into Pydantic model
    settings = load_config(settings_file)

    # Attach runtime-specific file paths to the settings object
    settings.state_file_loc = args.stage
    settings.settings_file = settings_file

    # Compute model enabled flags using shared function
    # This no longer needs the `disabled_models` argument
    enabled_flags = compute_model_enabled_flags(settings)
    settings.land_use_enabled = enabled_flags["land_use_enabled"]
    settings.vehicle_ownership_model_enabled = enabled_flags["vehicle_ownership_model_enabled"]
    settings.activity_demand_enabled = enabled_flags["activity_demand_enabled"]
    settings.traffic_assignment_enabled = enabled_flags["traffic_assignment_enabled"]
    settings.replanning_enabled = enabled_flags["replanning_enabled"]

    # raise errors/warnings for conflicting settings
    household_sample_size = get_setting(settings, "activitysim.household_sample_size", 0)

    if (household_sample_size > 0) and enabled_flags["land_use_enabled"]:
        raise ValueError(
            'Land use models must be disabled (explicitly or via "warm '
            'start" mode to use a non-zero household sample size. The '
            "household sample size you specified is {0}".format(
                household_sample_size
            )
        )

    atlas_beamac = get_setting(settings, "atlas.beamac", 0)
    region = get_setting(settings, "run.region")
    skims_zone_type = get_setting(settings, "shared.skims.zone_type")

    if (atlas_beamac > 0) and (
        (region != "sfbay") or (skims_zone_type != "taz")
    ):
        raise ValueError(
            "atlas_beamac must be 0 (read accessibility from RData) "
            "unless region = sfbay and skims_zone_type = taz. When"
            "atlas_beamac = 1, accessibility is calculated internally. "
        )

    return settings


def datastore_path(settings, year=None, mutable_data_dir=None):
    """
    Returns the path to the land use .H5 data store.
    If `year` is None, returns the base year data store.
    If `year` is specified, returns the forecast year data store.
    """
    region = get_setting(settings, "run.region")
    region_id_map = get_setting(settings, "urbansim.region_mappings.region_to_region_id", {})
    region_id = region_id_map.get(region)

    if mutable_data_dir is None:
        data_loc = get_setting(settings, "urbansim.local_data_input_folder")
    else:
        data_loc = mutable_data_dir
        os.makedirs(data_loc, exist_ok=True)

    if year is None:
        usim_datastore = get_setting(settings, "urbansim.input_file_template").format(
            region_id=region_id
        )
    else:
        usim_datastore = get_setting(settings, "urbansim.output_file_template").format(year=year)

    return os.path.join(data_loc, usim_datastore)


def read_datastore(settings, year=None, warm_start=False, mutable_data_dir=None):
    """
    Access to the land use .H5 data store
    """
    region = get_setting(settings, "run.region")
    region_id_map = get_setting(settings, "urbansim.region_mappings.region_to_region_id", {})
    region_id = region_id_map.get(region)

    if mutable_data_dir is None:
        data_loc = get_setting(settings, "urbansim.local_data_input_folder")
    else:
        data_loc = mutable_data_dir

    urbansim_enabled = get_setting(settings, "run.models.land_use") is not None
    start_year = get_setting(settings, "run.start_year")

    if (year == start_year) or warm_start or not urbansim_enabled:
        logger.info(
            f"Year {year}, start year {start_year}, warm_start {warm_start}, urbansim_enabled {urbansim_enabled}"
        )
        table_prefix_yr = ""  # input data store tables have no year prefix
        usim_datastore = get_setting(settings, "urbansim.input_file_template").format(
            region_id=region_id
        )
        usim_datastore_fpath = os.path.join(data_loc, usim_datastore)
        store = pd.HDFStore(usim_datastore_fpath)
        if "households" not in store:
            table_prefix_yr = str(year)
            if f"{table_prefix_yr}/households" not in store:
                raise KeyError(
                    f"No households table of either format found in {usim_datastore_fpath}. Tables: {store.keys()}"
                )
            else:
                logger.info(f"Using {table_prefix_yr}/households table")

    # Otherwise we read from the land use outputs
    else:
        usim_datastore = get_setting(settings, "urbansim.output_file_template").format(year=year)
        table_prefix_yr = str(year)
        usim_datastore_fpath = os.path.join(data_loc, usim_datastore)
        store = pd.HDFStore(usim_datastore_fpath)

    logger.info(f"Opening urbansim datastore at {usim_datastore}")

    if not os.path.exists(usim_datastore_fpath):
        raise ValueError(f"No land use data found at {usim_datastore_fpath}!")

    return store, table_prefix_yr


def get_merged_usim_input_datastore_path(settings, mutable_data_dir=None):
    region = get_setting(settings, "run.region")
    region_id_map = get_setting(settings, "urbansim.region_mappings.region_to_region_id", {})
    region_id = region_id_map.get(region)
    usim_base_fname = get_setting(settings, "urbansim.input_file_template")
    datastore_name = usim_base_fname.format(region_id=region_id)
    data_loc = mutable_data_dir if mutable_data_dir else get_setting(settings, "urbansim.local_data_input_folder")
    return os.path.join(data_loc, datastore_name)

def locate_asim_file(asim_output_data_dir, file_name, fmt):
    if fmt == "csv":
        return os.path.join(asim_output_data_dir, "final_" + file_name + ".csv")
    elif fmt == "parquet":
        if file_name == "plans":
            a_n = "beam_plans"
        else:
            a_n = file_name
        return os.path.join(
            asim_output_data_dir, "final_pipeline", a_n, "final.parquet"
        )
    elif fmt is None:
        return os.path.join(asim_output_data_dir, file_name)
    else:
        raise ValueError()


def locate_beam_file(beam_scenario_folder, file_name, fmt):
    if fmt == "csv":
        return os.path.join(beam_scenario_folder, file_name + ".csv.gz")
    elif fmt == "parquet":
        return os.path.join(beam_scenario_folder, file_name + ".parquet")
    else:
        raise ValueError()
