import os
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from unittest.mock import MagicMock
import yaml
import json

from pilates.config.models import load_config

# Define a canonical order for GEOIDs that our test will enforce
CANONICAL_GEOID_ORDER = [f"5303300{i:04d}" for i in range(5)]


@pytest.fixture
def mock_workspace(tmp_path):
    """Create a mock workspace with temporary directories."""
    workspace = MagicMock()

    # Set full_path to tmp_path for canonical_zones.csv
    workspace.full_path = tmp_path

    # Create temp dirs for beam and urbansim
    beam_mutable_dir = tmp_path / "beam" / "input"
    usim_mutable_dir = tmp_path / "urbansim" / "data"
    os.makedirs(beam_mutable_dir / "seattle" / "shape", exist_ok=True)
    os.makedirs(usim_mutable_dir, exist_ok=True)

    workspace.get_beam_mutable_data_dir.return_value = beam_mutable_dir
    workspace.get_usim_mutable_data_dir.return_value = usim_mutable_dir
    workspace.get_asim_mutable_data_dir.return_value = tmp_path

    # Create a dummy canonical_zones.geojson
    geojson_path = tmp_path / "canonical_zones.geojson"
    geojson_content = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"zone_key": geoid},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
            for idx, geoid in enumerate(CANONICAL_GEOID_ORDER)
        ],
    }
    with open(geojson_path, "w") as f:
        json.dump(geojson_content, f)

    return workspace


@pytest.fixture
def mock_h5_datastore(mock_workspace):
    """Create a mock HDF5 datastore that defines the canonical zone order."""
    h5_path = os.path.join(mock_workspace.get_usim_mutable_data_dir(), "usim_data.h5")

    # The order here defines the canonical order
    blocks_df = pd.DataFrame(index=CANONICAL_GEOID_ORDER)
    bgs_df = pd.DataFrame({"geoid10": CANONICAL_GEOID_ORDER})
    # Add a dummy households table to satisfy the read_datastore function
    households_df = pd.DataFrame({"id": [1, 2, 3]})

    with pd.HDFStore(h5_path, "w") as store:
        store.put("blocks", blocks_df)
        store.put("block_group_zone_geoms", bgs_df)
        store.put("households", households_df)

    return h5_path


@pytest.fixture
def mock_beam_shapefile(mock_workspace):
    """Create a mock BEAM shapefile with zones in the wrong order."""
    shapefile_dir = os.path.join(
        mock_workspace.get_beam_mutable_data_dir(), "seattle", "shape"
    )
    shapefile_path = os.path.join(shapefile_dir, "test_zones.shp")

    # Create GEOIDs in reverse order to test sorting
    scrambled_geoids = CANONICAL_GEOID_ORDER[::-1]

    # Create a simple geometry for each feature
    polygons = [
        Polygon([(i, i), (i + 1, i), (i + 1, i + 1), (i, i + 1)])
        for i in range(len(scrambled_geoids))
    ]

    gdf = gpd.GeoDataFrame({"geoid10": scrambled_geoids, "geometry": polygons})

    gdf.to_file(shapefile_path, driver="ESRI Shapefile")
    return shapefile_path


@pytest.fixture
def mock_settings(tmp_path, mock_h5_datastore, mock_beam_shapefile):
    """Create a mock hierarchical config file and load it."""
    config_dict = {
        "run": {
            "region": "seattle",
            "scenario": "test",
            "start_year": 2020,
            "end_year": 2021,
            "land_use_freq": 1,
            "travel_model_freq": 1,
            "vehicle_ownership_freq": 1,
            "supply_demand_iters": 1,
            "output_directory": str(tmp_path),
            "output_run_name": "test_run",
            "models": {"travel": "beam"},
        },
        "shared": {
            "geography": {
                "FIPS": {"seattle": {}},
                "local_crs": "EPSG:32048",
                "zones": {
                    "source_file": str(tmp_path / "canonical_zones.geojson"),
                    "activitysim_index_col": "TAZ",
                    "zone_type": "block_group",  # Added for Pydantic validation
                    "canonical_id_col": "zone_key",  # Added for Pydantic validation
                },
            },
            "skims": {
                "zone_type": "block_group",
                "fname": "",
                "geoms_fname": "",
                "geoms_index_col": "",
            },
            "database": {"enabled": False, "type": "duckdb", "path": ""},
        },
        "infrastructure": {
            "container_manager": "docker",
            "docker_images": {},
            "docker_config": {},
        },
        "beam": {
            "config": "beam.conf",
            "local_input_folder": "pilates/beam/production",  # Not used by test, but required by model
            "local_mutable_data_folder": "beam/input",
            "skims_shapefile": "shape/test_zones.shp",
            "skim_zone_geoid_col": "TAZ",
            # Add other required fields with dummy values
            "sample": 1.0,
            "replanning_portion": 0.1,
            "memory": "1g",
            "local_output_folder": "",
            "scenario_folder": "",
            "router_directory": "",
            "skim_zone_source_id_col": "",
            "discard_plans_every_year": False,
            "max_plans_memory": 1,
            "simulated_hwy_paths": [],
            "asim_hwy_measure_map": {},
            "asim_transit_measure_map": {},
            "asim_ridehail_measure_map": {},
            "ridehail_path_map": {},
        },
        "urbansim": {
            "local_data_input_folder": os.path.dirname(mock_h5_datastore),
            "input_file_template": os.path.basename(mock_h5_datastore),
            # Add other required fields
            "local_mutable_data_folder": "",
            "client_base_folder": "",
            "client_data_folder": "",
            "input_file_template_year": "",
            "output_file_template": "",
            "command_template": "",
            "region_mappings": {},
        },
    }
    config_path = tmp_path / "test_settings.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return load_config(config_path)


import pytest
from types import SimpleNamespace  # Import SimpleNamespace for mocking WorkflowState

from pilates.beam.preprocessor import (
    BeamPreprocessor,
)  # Import BeamPreprocessor

# Define a canonical order for GEOIDs that our test will enforce
CANONICAL_GEOID_ORDER = [f"5303300{i:04d}" for i in range(5)]


@pytest.fixture
def mock_workspace(tmp_path):
    """Create a mock workspace with temporary directories."""
    workspace = MagicMock()

    # Set full_path to tmp_path for canonical_zones.csv
    workspace.full_path = tmp_path

    # Create temp dirs for beam and urbansim
    beam_mutable_dir = tmp_path / "beam" / "input"
    usim_mutable_dir = tmp_path / "urbansim" / "data"
    os.makedirs(beam_mutable_dir / "seattle" / "shape", exist_ok=True)
    os.makedirs(usim_mutable_dir, exist_ok=True)

    workspace.get_beam_mutable_data_dir.return_value = beam_mutable_dir
    workspace.get_usim_mutable_data_dir.return_value = usim_mutable_dir
    workspace.get_asim_mutable_data_dir.return_value = tmp_path

    # Create a dummy canonical_zones.geojson
    geojson_path = tmp_path / "canonical_zones.geojson"
    geojson_content = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"zone_key": geoid},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
            for idx, geoid in enumerate(CANONICAL_GEOID_ORDER)
        ],
    }
    with open(geojson_path, "w") as f:
        json.dump(geojson_content, f)

    return workspace


@pytest.fixture
def mock_h5_datastore(mock_workspace):
    """Create a mock HDF5 datastore that defines the canonical zone order."""
    h5_path = os.path.join(mock_workspace.get_usim_mutable_data_dir(), "usim_data.h5")

    # The order here defines the canonical order
    blocks_df = pd.DataFrame(index=CANONICAL_GEOID_ORDER)
    bgs_df = pd.DataFrame({"geoid10": CANONICAL_GEOID_ORDER})
    # Add a dummy households table to satisfy the read_datastore function
    households_df = pd.DataFrame({"id": [1, 2, 3]})

    with pd.HDFStore(h5_path, "w") as store:
        store.put("blocks", blocks_df)
        store.put("block_group_zone_geoms", bgs_df)
        store.put("households", households_df)

    return h5_path


@pytest.fixture
def mock_settings(tmp_path, mock_h5_datastore, mock_beam_shapefile):
    """Create a mock hierarchical config file and load it."""
    config_dict = {
        "run": {
            "region": "seattle",
            "scenario": "test",
            "start_year": 2020,
            "end_year": 2021,
            "land_use_freq": 1,
            "travel_model_freq": 1,
            "vehicle_ownership_freq": 1,
            "supply_demand_iters": 1,
            "output_directory": str(tmp_path),
            "output_run_name": "test_run",
            "models": {"travel": "beam"},
        },
        "shared": {
            "geography": {
                "FIPS": {"seattle": {}},
                "local_crs": "EPSG:32048",
                "zones": {
                    "source_file": str(tmp_path / "canonical_zones.geojson"),
                    "activitysim_index_col": "TAZ",
                    "zone_type": "block_group",  # Added for Pydantic validation
                    "canonical_id_col": "zone_key",  # Added for Pydantic validation
                },
            },
            "skims": {
                "zone_type": "block_group",
                "fname": "",
                "geoms_fname": "",
                "geoms_index_col": "",
            },
            "database": {"enabled": False, "type": "duckdb", "path": ""},
        },
        "infrastructure": {
            "container_manager": "docker",
            "docker_images": {},
            "docker_config": {},
        },
        "beam": {
            "config": "beam.conf",
            "local_input_folder": "pilates/beam/production",  # Not used by test, but required by model
            "local_mutable_data_folder": "beam/input",
            "skims_shapefile": "shape/test_zones.shp",
            "skim_zone_geoid_col": "TAZ",
            # Add other required fields with dummy values
            "sample": 1.0,
            "replanning_portion": 0.1,
            "memory": "1g",
            "local_output_folder": "",
            "scenario_folder": "",
            "router_directory": "",
            "skim_zone_source_id_col": "",
            "discard_plans_every_year": False,
            "max_plans_memory": 1,
            "simulated_hwy_paths": [],
            "asim_hwy_measure_map": {},
            "asim_transit_measure_map": {},
            "asim_ridehail_measure_map": {},
            "ridehail_path_map": {},
        },
        "urbansim": {
            "local_data_input_folder": os.path.dirname(mock_h5_datastore),
            "input_file_template": os.path.basename(mock_h5_datastore),
            # Add other required fields
            "local_mutable_data_folder": "",
            "client_base_folder": "",
            "client_data_folder": "",
            "input_file_template_year": "",
            "output_file_template": "",
            "command_template": "",
            "region_mappings": {},
        },
    }
    config_path = tmp_path / "test_settings.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return load_config(config_path)


class TestBeamPreprocessor:
    """Tests for the BEAM Preprocessor's zone order alignment."""

    def test_shapefile_sorting_and_indexing(
        self, mock_settings, mock_workspace, mock_beam_shapefile
    ):
        """
        Test that the preprocessor correctly sorts the BEAM shapefile and adds
        the asim_idx column, and tracks provenance.
        """
        # Arrange
        mock_state = SimpleNamespace(
            full_settings=mock_settings,
            current_year=2020,
            current_inner_iter=0,
            run_info_path=None,  # Mock as needed
        )
        provenance_tracker = MagicMock()
        model_run_hash = "test_hash_123"

        # Instantiate the preprocessor
        preprocessor = BeamPreprocessor(
            model_name="beam",
            state=mock_state,
            provenance_tracker=provenance_tracker,
            major_stage=None,
        )

        # Act
        # Call the public method that performs the sorting and provenance tracking
        output_shapefile_path = preprocessor.prepare_beam_zone_shapefile(
            mock_workspace, model_run_hash
        )

        # Assert
        # 1. Verify the shapefile was modified correctly
        sorted_gdf = gpd.read_file(output_shapefile_path)

        # 1a. Check if the order of GEOIDs now matches the canonical order
        final_order = sorted_gdf[mock_settings.beam.skim_zone_geoid_col].tolist()
        assert (
            final_order == CANONICAL_GEOID_ORDER
        ), "Shapefile GEOID order does not match canonical order after sorting."

        # 2. Verify provenance tracking for the output file
        provenance_tracker.record_output_file.assert_called_once_with(
            "beam_preprocessor",
            output_shapefile_path,
            short_name="beam_zone_shapefile_sorted",
            description="BEAM zone shapefile created from sorted canonical zones.",
            model_run_id=model_run_hash,
        )

        # Ensure record_output_file_with_inputs was not called by this method
        provenance_tracker.record_output_file_with_inputs.assert_not_called()
