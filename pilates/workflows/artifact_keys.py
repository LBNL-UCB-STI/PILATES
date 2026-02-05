"""
Canonical artifact key registry for workflow coupler/provenance integration.
"""

from __future__ import annotations

from typing import Dict, List


class ArtifactKeys:
    # UrbanSim
    USIM_DATASTORE_H5 = "usim_datastore_h5"
    USIM_H5_UPDATED = "usim_h5_updated"
    USIM_MUTABLE_DATA_DIR = "usim_mutable_data_dir"
    USIM_FORECAST_OUTPUT = "usim_forecast_output"
    USIM_INPUT_MERGED_PREFIX = "usim_input_merged_"
    USIM_INPUT_ARCHIVE_PREFIX = "usim_input_archive_"

    # ATLAS
    ATLAS_OUTPUT_DIR = "atlas_output_dir"
    ATLAS_VEHICLES2_INPUT = "atlas_vehicles2_input"

    # ActivitySim
    ASIM_MUTABLE_DATA_DIR = "asim_mutable_data_dir"
    ASIM_OUTPUT_DIR = "asim_output_dir"
    ASIM_LAND_USE_IN = "land_use_asim_in"
    ASIM_HOUSEHOLDS_IN = "households_asim_in"
    ASIM_PERSONS_IN = "persons_asim_in"
    ASIM_OMX_SKIMS = "omx_skims"

    # BEAM
    BEAM_OUTPUT_DIR = "beam_output_dir"
    BEAM_MUTABLE_DATA_DIR = "beam_mutable_data_dir"
    LINKSTATS = "linkstats"
    BEAM_PLANS_OUT = "beam_plans_out"
    BEAM_OUTPUT_PLANS_XML = "beam_output_plans_xml"
    BEAM_EXPERIENCED_PLANS_XML = "beam_experienced_plans_xml"
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML = "beam_output_experienced_plans_xml"
    BEAM_R5_OSM_FILE = "beam_r5_osm_file"
    BEAM_PLANS_IN = "plans_beam_in"
    BEAM_HOUSEHOLDS_IN = "households_beam_in"
    BEAM_PERSONS_IN = "persons_beam_in"
    LINKSTATS_WARMSTART = "linkstats_warmstart"

    # Cross-model skims
    ZARR_SKIMS = "zarr_skims"
    FINAL_SKIMS_OMX = "final_skims_omx"

    @classmethod
    def as_dict(cls) -> Dict[str, str]:
        return {
            name: value
            for name, value in vars(cls).items()
            if name.isupper() and isinstance(value, str)
        }

    @classmethod
    def all(cls) -> List[str]:
        return list(cls.as_dict().values())


# Backward-compatible module-level constants for direct imports, e.g.:
# `from pilates.workflows.artifact_keys import USIM_DATASTORE_H5`
for _name, _value in ArtifactKeys.as_dict().items():
    globals()[_name] = _value

del _name
del _value

__all__ = ["ArtifactKeys", *ArtifactKeys.as_dict().keys()]
