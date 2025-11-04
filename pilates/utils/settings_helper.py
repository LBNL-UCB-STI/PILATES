"""
Helper utilities for accessing settings in both legacy (flat) and new (nested) formats.

This module provides a backward-compatible way to access configuration values
whether they come from legacy YAML files or new Pydantic-validated nested configs.
"""


def get_nested_or_legacy(settings, nested_path, legacy_key=None, default=None):
    """
    Get a setting value from nested path (Pydantic format) or legacy flat key.

    Args:
        settings: Settings dict (can be nested or flat)
        nested_path: Dot-separated path for nested access (e.g., "run.region")
        legacy_key: Legacy flat key (e.g., "region"). If None, uses last part of nested_path
        default: Default value if not found

    Returns:
        The setting value, or default if not found

    Examples:
        >>> settings = {"run": {"region": "sfbay"}}
        >>> get_nested_or_legacy(settings, "run.region", "region")
        'sfbay'

        >>> settings = {"region": "sfbay"}  # legacy format
        >>> get_nested_or_legacy(settings, "run.region", "region")
        'sfbay'
    """
    if legacy_key is None:
        legacy_key = nested_path.split('.')[-1]

    # Try nested path first
    value = settings
    for key in nested_path.split('.'):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            # Not found in nested path, try legacy key
            return settings.get(legacy_key, default)

    return value


# Common settings paths mapping
# Maps nested path -> legacy key for common settings
COMMON_SETTINGS = {
    # Run-level settings
    "run.region": "region",
    "run.scenario": "scenario",
    "run.start_year": "start_year",
    "run.end_year": "end_year",
    "run.land_use_freq": "land_use_freq",
    "run.travel_model_freq": "travel_model_freq",
    "run.vehicle_ownership_freq": "vehicle_ownership_freq",
    "run.supply_demand_iters": "supply_demand_iters",
    "run.output_directory": "output_directory",
    "run.output_run_name": "output_run_name",

    # Model selection
    "run.models.land_use": "land_use_model",
    "run.models.travel": "travel_model",
    "run.models.activity_demand": "activity_demand_model",
    "run.models.vehicle_ownership": "vehicle_ownership_model",

    # Shared settings
    "shared.skims.zone_type": "skims_zone_type",
    "shared.skims.fname": "skims_fname",
    "shared.skims.origin_fname": "origin_skims_fname",
    "shared.skims.geoms_fname": "skims_geoms_fname",
    "shared.skims.geoms_index_col": "skims_geoms_index_col",

    # Infrastructure
    "infrastructure.container_manager": "container_manager",
    "infrastructure.docker.stdout": "docker_stdout",
    "infrastructure.docker.pull_latest": "pull_latest",

    # Database
    "shared.database.enabled": "database_enabled",
    "shared.database.path": "database_path",

    # UrbanSim
    "urbansim.region_id": "region_id",

    # ATLAS
    "atlas.sample_size": "atlas_sample_size",
    "atlas.num_processes": "atlas_npe",
    "atlas.beamac": "atlas_beamac",
    "atlas.mod": "atlas_mod",
    "atlas.scenario": "atlas_scenario",
    "atlas.adscen": "atlas_adscen",
    "atlas.rebfactor": "atlas_rebfactor",
    "atlas.taxfactor": "atlas_taxfactor",
    "atlas.discIncent": "atlas_discIncent",

    # ActivitySim
    "activitysim.household_sample_size": "household_sample_size",
    "activitysim.chunk_size": "chunk_size",
    "activitysim.num_processes": "num_processes",
    "activitysim.replan_iters": "replan_iters",

    # BEAM
    "beam.sample": "beam_sample",
    "beam.memory": "beam_memory",
    "beam.config": "beam_config",
    "beam.local_input_folder": "beam_local_input_folder",
    "beam.local_output_folder": "beam_local_output_folder",
    "beam.router_directory": "beam_router_directory",

    # Note: Some settings like region_to_region_id, usim_formattable_input_file_name, etc.
    # are kept at top-level as they're mappings/templates that don't fit the nested structure.
    # They can be accessed directly with settings["key"] for now.
}


def get(settings, nested_path, default=None):
    """
    Convenient shorthand for getting common settings.

    Uses the COMMON_SETTINGS mapping to automatically determine the legacy key.

    Args:
        settings: Settings dict
        nested_path: Nested path (e.g., "run.region")
        default: Default value if not found

    Returns:
        The setting value, or default if not found

    Examples:
        >>> settings = {"run": {"region": "sfbay"}}
        >>> get(settings, "run.region")
        'sfbay'

        >>> settings = {"region": "sfbay"}  # legacy
        >>> get(settings, "run.region")
        'sfbay'
    """
    legacy_key = COMMON_SETTINGS.get(nested_path)
    if legacy_key is None:
        # Not in common settings, just use last part of path as legacy key
        legacy_key = nested_path.split('.')[-1]

    return get_nested_or_legacy(settings, nested_path, legacy_key, default)
