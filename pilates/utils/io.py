import argparse
import logging
import os
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_model_enabled_flags(settings, disabled_models=""):
    """
    Compute which models are enabled based on config and command-line args.

    Works with both legacy (flat) and new (nested Pydantic) config formats.

    Args:
        settings: Settings dict (either flat or nested format)
        disabled_models: String of disabled model letters from -d flag (e.g., "lt")

    Returns:
        dict: Flags for land_use_enabled, vehicle_ownership_model_enabled, etc.
    """
    # Helper to get nested or flat values
    def get(nested_path, legacy_key, default=None):
        value = settings
        for key in nested_path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = settings.get(legacy_key, default)
                break
        return value

    # Get model names from nested or legacy locations
    land_use_model = get("run.models.land_use", "land_use_model", False)
    vehicle_ownership_model = get("run.models.vehicle_ownership", "vehicle_ownership_model", False)
    activity_demand_model = get("run.models.activity_demand", "activity_demand_model", False)
    travel_model = get("run.models.travel", "travel_model", False)

    warm_start_skims = settings.get("warm_start_skims", False)
    static_skims = settings.get("static_skims", False)
    replan_iters = get("activitysim.replan_iters", "replan_iters", 0)

    # Compute enabled flags
    land_use_enabled = (
        bool(land_use_model)
        and (not warm_start_skims)
        and ("l" not in disabled_models)
    )

    vehicle_ownership_model_enabled = bool(vehicle_ownership_model)

    activity_demand_enabled = bool(activity_demand_model) and ("a" not in disabled_models)

    traffic_assignment_enabled = (
        bool(travel_model)
        and (not static_skims)
        and ("t" not in disabled_models)
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
        "-v", "--verbose", action="store_true", help="print docker stdout"
    )
    parser.add_argument(
        "-p",
        "--pull_latest",
        action="store_true",
        help="pull latest docker images before running",
    )
    parser.add_argument(
        "-h", "--household_sample_size", action="store", help="household sample size"
    )
    parser.add_argument(
        "-s",
        "--static_skims",
        action="store_true",
        help="bypass traffic assignment, use same skims for every run.",
    )
    parser.add_argument(
        "-w",
        "--warm_start_skims",
        action="store_true",
        help="generate full activity plans for the base year only.",
    )
    parser.add_argument(
        "-f", "--figures", action="store_true", help="outputs validation figures"
    )
    parser.add_argument(
        "-d",
        "--disable_model",
        action="store",
        help=(
            '"l" for land use, "a" for activity demand, '
            '"t" for traffic assignment. Can specify multiple (e.g. "at")'
        ),
    )
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

    # read settings from config file
    with open(settings_file) as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    disabled_models = "" if args.disable_model is None else args.disable_model

    # command-line only settings:
    settings.update(
        {
            "static_skims": args.static_skims,
            "warm_start_skims": args.warm_start_skims,
            "asim_validation": args.figures,
            "state_file_loc": args.stage,
            "settings_file": settings_file,
            "_disabled_models": disabled_models,  # Store for later use
        }
    )

    # override .yaml settings with command-line values if command-line
    # values are not False/None
    if args.verbose:
        settings.update({"docker_stdout": args.verbose})
    if args.pull_latest:
        settings.update({"pull_latest": args.pull_latest})
    if args.household_sample_size:
        settings.update({"household_sample_size": args.household_sample_size})

    # Compute model enabled flags using shared function (disabled_models already stored above)
    enabled_flags = compute_model_enabled_flags(settings, disabled_models)
    settings.update(enabled_flags)

    # raise errors/warnings for conflicting settings
    # Check both legacy (top-level) and new nested (activitysim.household_sample_size) locations
    household_sample_size = settings.get("household_sample_size")
    if household_sample_size is None:
        # Try nested location for Pydantic-style configs
        household_sample_size = settings.get("activitysim", {}).get("household_sample_size", 0)

    if (household_sample_size > 0) and land_use_enabled:
        raise ValueError(
            'Land use models must be disabled (explicitly or via "warm '
            'start" mode to use a non-zero household sample size. The '
            "household sample size you specified is {0}".format(
                household_sample_size
            )
        )
    # Check both legacy and nested locations for atlas_beamac, region, skims_zone_type
    atlas_beamac = settings.get("atlas_beamac")
    if atlas_beamac is None:
        atlas_beamac = settings.get("atlas", {}).get("beamac", 0)

    region = settings.get("region")
    if region is None:
        region = settings.get("run", {}).get("region")

    skims_zone_type = settings.get("skims_zone_type")
    if skims_zone_type is None:
        skims_zone_type = settings.get("shared", {}).get("skims", {}).get("zone_type")

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
    region = settings["region"]
    region_id = settings["region_to_region_id"][region]
    if mutable_data_dir is None:
        data_loc = settings["usim_local_data_input_folder"]
    else:
        data_loc = mutable_data_dir # Corrected: Use mutable_data_dir directly
        os.makedirs(data_loc, exist_ok=True)
    if year is None:
        usim_datastore = settings["usim_formattable_input_file_name"].format(
            region_id=region_id
        )
    else:
        usim_datastore = settings["usim_formattable_output_file_name"].format(year=year)
    return os.path.join(data_loc, usim_datastore)


def read_datastore(settings, year=None, warm_start=False, mutable_data_dir=None):
    """
    Access to the land use .H5 data store
    """
    region = settings["region"]
    region_id = settings["region_to_region_id"][region]
    if mutable_data_dir is None:
        data_loc = settings["usim_local_data_input_folder"]
    else:
        data_loc = mutable_data_dir # Corrected: Use mutable_data_dir directly
    urbansim_enabled = settings.get("land_use_model") is not None

    if (year == settings["start_year"]) or warm_start or not urbansim_enabled:
        logger.info(
            "Year {0}, start year {1}, warm_start {2}, urbansim_enabled {3}".format(
                year, settings["start_year"], warm_start, urbansim_enabled
            )
        )
        table_prefix_yr = ""  # input data store tables have no year prefix
        usim_datastore = settings["usim_formattable_input_file_name"].format(
            region_id=region_id
        )
        usim_datastore_fpath = os.path.join(data_loc, usim_datastore)
        store = pd.HDFStore(usim_datastore_fpath)
        if "households" not in store:
            table_prefix_yr = str(year)
            if "{0}/households".format(table_prefix_yr) not in store:
                raise KeyError(
                    "No households table of either format found in {0}. Tables: {1}".format(
                        usim_datastore_fpath, store.keys()
                    )
                )
            else:
                logger.info("Using {0}/households table".format(table_prefix_yr))

    # Otherwise we read from the land use outputs
    else:
        usim_datastore = settings["usim_formattable_output_file_name"].format(year=year)
        table_prefix_yr = str(year)
        usim_datastore_fpath = os.path.join(data_loc, usim_datastore)
        store = pd.HDFStore(usim_datastore_fpath)

    logger.info("Opening urbansim datastore at {0}".format(usim_datastore))

    if not os.path.exists(usim_datastore_fpath):
        raise ValueError("No land use data found at {0}!".format(usim_datastore_fpath))

    return store, table_prefix_yr


def get_merged_usim_input_datastore_path(settings, mutable_data_dir=None):
    region = settings["region"]
    region_id = settings["region_to_region_id"][region]
    usim_base_fname = settings["usim_formattable_input_file_name"]
    datastore_name = usim_base_fname.format(region_id=region_id)
    data_loc = mutable_data_dir if mutable_data_dir else settings["usim_local_data_input_folder"]
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
