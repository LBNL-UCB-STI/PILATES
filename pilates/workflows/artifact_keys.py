"""
Canonical artifact key registry for workflow coupler/provenance integration.
"""

from __future__ import annotations

from pilates.utils.usim_h5 import (
    USIM_POPULATION_BLOCKS_TABLE as USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE as USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE as USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE as USIM_POPULATION_PERSONS_TABLE,
)


# UrbanSim
USIM_DATASTORE_CURRENT_H5 = "usim_datastore_h5"
USIM_DATASTORE_BASE_H5 = "usim_datastore_base_h5"
# ``usim_datastore_h5`` is the canonical "current mutable datastore" handle
# used across workflow boundaries. ``USIM_DATASTORE_H5`` remains as a
# compatibility alias for that current-role key, not as a distinct third
# datastore concept.
USIM_DATASTORE_H5 = USIM_DATASTORE_CURRENT_H5
USIM_FORECAST_OUTPUT = "usim_forecast_output"
USIM_POPULATION_SOURCE_H5 = "usim_population_source_h5"
USIM_INPUT_MERGED_PREFIX = "usim_input_merged_"
USIM_INPUT_ARCHIVE_PREFIX = "usim_input_archive_"

# ATLAS
ATLAS_OUTPUT_DIR = "atlas_output_dir"
ATLAS_VEHICLES2_INPUT = "atlas_vehicles2_input"
ATLAS_VEHICLES2_OUTPUT = "atlas_vehicles2_output"

# ActivitySim
ASIM_MUTABLE_DATA_DIR = "asim_mutable_data_dir"
ASIM_OUTPUT_DIR = "asim_output_dir"
ASIM_LAND_USE_IN = "land_use_asim_in"
ASIM_HOUSEHOLDS_IN = "households_asim_in"
ASIM_PERSONS_IN = "persons_asim_in"
ASIM_OMX_SKIMS = "omx_skims"
# ``omx_skims`` is a cross-model semantic role used by UrbanSim, ATLAS, and
# ActivitySim. ``ASIM_OMX_SKIMS`` remains for compatibility with the existing
# ActivitySim-oriented name, while ``OMX_SKIMS`` is the preferred generic alias.
OMX_SKIMS = ASIM_OMX_SKIMS
ASIM_SHARROW_CACHE_DIR = "asim_sharrow_cache_dir"

# BEAM
BEAM_OUTPUT_DIR = "beam_output_dir"
BEAM_MUTABLE_DATA_DIR = "beam_mutable_data_dir"
BEAM_NETWORK_FINAL = "beam_network_final"
LINKSTATS = "linkstats"
BEAM_PLANS_OUT = "beam_plans_out"
BEAM_OUTPUT_PLANS_XML = "beam_output_plans_xml"
BEAM_EXPERIENCED_PLANS_XML = "beam_experienced_plans_xml"
BEAM_OUTPUT_EXPERIENCED_PLANS_XML = "beam_output_experienced_plans_xml"
BEAM_CONFIG_FILE = "beam_config_file"
BEAM_R5_OSM_FILE = "beam_r5_osm_file"
BEAM_PLANS_IN = "plans_beam_in"
BEAM_HOUSEHOLDS_IN = "households_beam_in"
BEAM_PERSONS_IN = "persons_beam_in"
BEAM_VEHICLES_IN = "vehicles_beam_in"
LINKSTATS_WARMSTART = "linkstats_warmstart"
BEAM_INPUT_PLANS_ARCHIVED = "beam_input_plans_archived"
BEAM_INPUT_HOUSEHOLDS_ARCHIVED = "beam_input_households_archived"
BEAM_INPUT_PERSONS_ARCHIVED = "beam_input_persons_archived"
BEAM_INPUT_CONFIG_ARCHIVED = "beam_input_config_archived"
BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED = "beam_input_config_references_archived"
BEAM_INPUT_VEHICLES_ARCHIVED = "beam_input_vehicles_archived"
BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED = "beam_input_linkstats_warmstart_archived"
BEAM_INPUT_PLANS_WARMSTART_ARCHIVED = "beam_input_plans_warmstart_archived"
BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED = (
    "beam_input_experienced_plans_warmstart_archived"
)

# Cross-model skims
ZARR_SKIMS = "zarr_skims"
FINAL_SKIMS_OMX = "final_skims_omx"
BEAM_FULL_SKIMS = "beam_full_skims"

# IMPACTS step inputs and outputs
IMPACTS_PRESIM_SETTINGS = "impacts_presim_settings"
IMPACTS_POSTSIM_SETTINGS = "impacts_postsim_settings"
IMPACTS_PRESIM_BRIDGE = "impacts_presim_bridge"
IMPACTS_POSTSIM_BRIDGE = "impacts_postsim_bridge"
IMPACTS_POSTSIM_DIR = "impacts_postsim_dir"
IMPACTS_ACTIVITIES_DIR = "impacts_activities_dir"
IMPACTS_FLEET_DIR = "impacts_fleet_dir"
IMPACTS_FLEET_MANIFEST = "impacts_fleet_manifest"
IMPACTS_PIPELINE_MANIFEST = "impacts_pipeline_manifest"
IMPACTS_POSTPROCESS_MANIFEST = "impacts_postprocess_manifest"
IMPACTS_EXPOSURE_TABLE = "impacts_exposure_table"
BEAM_EMISSIONS_SKIMS = "skims_emissions_parquet"
