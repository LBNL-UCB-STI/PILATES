import os
import pytest
import pandas as pd
import xarray as xr
import numpy as np
import yaml
from unittest.mock import patch, MagicMock

from pilates.beam.postprocessor import _merge_beam_skims_to_zarr, verify_skim_zone_order
from pilates.config.models import load_config

# Use the same canonical order as the preprocessor test for consistency
CANONICAL_GEOID_ORDER = [f"5303300{i:04d}" for i in range(5)]
NUM_ZONES = len(CANONICAL_GEOID_ORDER)
TIME_PERIODS = ["EA", "AM", "MD", "PM", "EV"]

@pytest.fixture
def mock_settings(tmp_path):
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
            "models": {"travel": "beam"}
        },
        "shared": {
            "geography": {"FIPS": {"seattle": {}}, "local_crs": "EPSG:32048"},
            "skims": {"zone_type": "block_group", "fname": "", "geoms_fname": "", "geoms_index_col": "", "periods": TIME_PERIODS},
            "database": {"enabled": False, "type": "duckdb", "path": ""}
        },
        "infrastructure": {"container_manager": "docker", "docker_images": {}, "docker_config": {}},
        "beam": {
            "config": "",
            "local_input_folder": "pilates/beam/production",
            "local_mutable_data_folder": "beam/input",
            "skims_shapefile": "shape/test_zones.shp",
            "skim_zone_geoid_col": "geoid10",
            "sample": 1.0, "replanning_portion": 0.1, "memory": "1g", "local_output_folder": "",
            "scenario_folder": "", "router_directory": "", "skim_zone_source_id_col": "",
            "discard_plans_every_year": False, "max_plans_memory": 1, "simulated_hwy_paths": [],
            "asim_hwy_measure_map": {}, "asim_transit_measure_map": {}, "asim_ridehail_measure_map": {},
            "ridehail_path_map": {}
        },
        "urbansim": {
            "local_data_input_folder": str(tmp_path), # Dummy path
            "input_file_template": "usim_data.h5", # Dummy file
            "local_mutable_data_folder": "", "client_base_folder": "", "client_data_folder": "",
            "input_file_template_year": "", "output_file_template": "", "command_template": "", "region_mappings": {}
        }
    }
    config_path = tmp_path / "test_settings_postproc.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    return load_config(config_path)

@pytest.fixture
def initial_main_zarr(tmp_path):
    """
    Create the main Zarr skim file that will be updated.
    Represents activitysim/output/cache/skims.zarr
    """
    main_zarr_path = tmp_path / "main_skims.zarr"
    
    # Create an initial Zarr store with the expected structure and zeroed data
    # This mimics what ActivitySim might produce initially
    initial_sov_trips_data = np.zeros((len(TIME_PERIODS), NUM_ZONES, NUM_ZONES), dtype=np.float32)
    initial_sov_failures_data = np.zeros((len(TIME_PERIODS), NUM_ZONES, NUM_ZONES), dtype=np.float32)

    ds = xr.Dataset(
        {
            "SOV_TRIPS": (("time_period", "otaz", "dtaz"), initial_sov_trips_data),
            "SOV_FAILURES": (("time_period", "otaz", "dtaz"), initial_sov_failures_data),
        },
        coords={
            "time_period": TIME_PERIODS,
            "otaz": np.arange(NUM_ZONES),
            "dtaz": np.arange(NUM_ZONES),
        },
        attrs={
            "description": "Initial main skims",
            "original_zone_ids": CANONICAL_GEOID_ORDER # Must have this for verification
        }
    )
    ds.to_zarr(main_zarr_path, mode='w', consolidated=True)
    return str(main_zarr_path)

@pytest.fixture
def beam_iteration_zarr_base(tmp_path):
    """
    Base fixture to create a partial Zarr skim file from a BEAM iteration.
    """
    beam_zarr_path = tmp_path / "beam_iteration.zarr"
    
    # Create sample data for one skim measure
    sov_trips_data = np.ones((len(TIME_PERIODS), NUM_ZONES, NUM_ZONES), dtype=np.float32)
    
    ds = xr.Dataset(
        {
            "SOV_TRIPS": (("time_period", "otaz", "dtaz"), sov_trips_data),
        },
        coords={
            "time_period": TIME_PERIODS,
            "otaz": np.arange(NUM_ZONES),
            "dtaz": np.arange(NUM_ZONES),
        },
        attrs={
            "description": "Partial skims from BEAM iteration",
        }
    )
    return ds, str(beam_zarr_path)

@pytest.fixture
def beam_iteration_zarr_valid(beam_iteration_zarr_base):
    """Creates a valid BEAM Zarr with original_zone_ids in canonical order."""
    ds, path = beam_iteration_zarr_base
    ds.otaz.attrs['original_zone_ids'] = CANONICAL_GEOID_ORDER
    ds.to_zarr(path, mode='w', consolidated=True)
    return path

@pytest.fixture
def beam_iteration_zarr_scrambled(beam_iteration_zarr_base):
    """Creates a BEAM Zarr with original_zone_ids in a scrambled order."""
    ds, path = beam_iteration_zarr_base
    scrambled_order = CANONICAL_GEOID_ORDER[::-1] # Reverse order
    ds.otaz.attrs['original_zone_ids'] = scrambled_order
    ds.to_zarr(path, mode='w', consolidated=True)
    return path

@pytest.fixture
def beam_iteration_zarr_missing_attr(beam_iteration_zarr_base):
    """Creates a BEAM Zarr without the original_zone_ids attribute."""
    ds, path = beam_iteration_zarr_base
    # Do not set ds.otaz.attrs['original_zone_ids']
    ds.to_zarr(path, mode='w', consolidated=True)
    return path

@pytest.fixture
def beam_iteration_zarr_0_based_int_ids(beam_iteration_zarr_base):
    """Creates a BEAM Zarr with original_zone_ids as [0, 1, 2, ...]."""
    ds, path = beam_iteration_zarr_base
    ds.otaz.attrs['original_zone_ids'] = [str(i) for i in range(NUM_ZONES)]
    ds.to_zarr(path, mode='w', consolidated=True)
    return path

@pytest.fixture
def beam_iteration_zarr_1_based_int_ids(beam_iteration_zarr_base):
    """Creates a BEAM Zarr with original_zone_ids as [1, 2, 3, ...]."""
    ds, path = beam_iteration_zarr_base
    ds.otaz.attrs['original_zone_ids'] = [str(i + 1) for i in range(NUM_ZONES)]
    ds.to_zarr(path, mode='w', consolidated=True)
    return path


class TestBeamPostprocessor:
    """Tests for the BEAM Postprocessor's skim merging logic."""

    @patch('pilates.beam.postprocessor.geoid_to_zone_map')
    def test_merge_beam_skims_to_zarr_valid(
        self, mock_geoid_map, mock_settings, initial_main_zarr, beam_iteration_zarr_valid
    ):
        """
        Test the basic merge functionality with valid BEAM skims:
        - Verifies zone order before merging.
        - Checks that data from the partial skim is merged into the main skim.
        """
        # Arrange
        mock_geoid_map.return_value = {geoid: i for i, geoid in enumerate(CANONICAL_GEOID_ORDER)}

        # Act
        _merge_beam_skims_to_zarr(
            all_skims_path=initial_main_zarr,
            iteration_skims_path=beam_iteration_zarr_valid,
            beam_output_dir="", # Not used in this logic path
            settings=mock_settings,
        )

        # Assert
        mock_geoid_map.assert_called_once_with(mock_settings)

        updated_ds = xr.open_zarr(initial_main_zarr)
        assert "SOV_TRIPS" in updated_ds.data_vars, "SOV_TRIPS was not merged into the main Zarr file."
        expected_data = np.ones((len(TIME_PERIODS), NUM_ZONES, NUM_ZONES), dtype=np.float32)
        np.testing.assert_array_equal(
            updated_ds["SOV_TRIPS"].values,
            expected_data,
            "Data for SOV_TRIPS in main Zarr file is incorrect after merge."
        )
        assert "original_zone_ids" in updated_ds.attrs
        assert updated_ds.attrs["original_zone_ids"] == CANONICAL_GEOID_ORDER

    @patch('pilates.beam.postprocessor.geoid_to_zone_map')
    def test_merge_beam_skims_to_zarr_scrambled_order_raises_error(
        self, mock_geoid_map, mock_settings, initial_main_zarr, beam_iteration_zarr_scrambled
    ):
        """
        Test that merging BEAM skims with scrambled zone order raises a ValueError.
        """
        # Arrange
        mock_geoid_map.return_value = {geoid: i for i, geoid in enumerate(CANONICAL_GEOID_ORDER)}

        # Act & Assert
        with pytest.raises(ValueError, match="FATAL: Skim zone order \\(from original_zone_ids attribute\\) does not match canonical order!"):
            _merge_beam_skims_to_zarr(
                all_skims_path=initial_main_zarr,
                iteration_skims_path=beam_iteration_zarr_scrambled,
                beam_output_dir="",
                settings=mock_settings,
            )
        mock_geoid_map.assert_called_once_with(mock_settings)

    @patch('pilates.beam.postprocessor.geoid_to_zone_map')
    def test_merge_beam_skims_to_zarr_missing_attr_raises_error(
        self, mock_geoid_map, mock_settings, initial_main_zarr, beam_iteration_zarr_missing_attr
    ):
        """
        Test that merging BEAM skims missing the original_zone_ids attribute raises a ValueError.
        """
        # Arrange
        mock_geoid_map.return_value = {geoid: i for i, geoid in enumerate(CANONICAL_GEOID_ORDER)}

        # Act & Assert
        with pytest.raises(ValueError, match="Zarr file does not contain 'original_zone_ids' metadata."):
            _merge_beam_skims_to_zarr(
                all_skims_path=initial_main_zarr,
                iteration_skims_path=beam_iteration_zarr_missing_attr,
                beam_output_dir="",
                settings=mock_settings,
            )
        mock_geoid_map.assert_called_once_with(mock_settings)

    @patch('pilates.beam.postprocessor.geoid_to_zone_map')
    def test_merge_beam_skims_to_zarr_0_based_int_ids_raises_error(
        self, mock_geoid_map, mock_settings, initial_main_zarr, beam_iteration_zarr_0_based_int_ids
    ):
        """
        Test that merging BEAM skims with 0-based integer zone IDs raises a ValueError.
        """
        # Arrange
        mock_geoid_map.return_value = {geoid: i for i, geoid in enumerate(CANONICAL_GEOID_ORDER)}

        # Act & Assert
        with pytest.raises(ValueError, match="FATAL: Skim zone order \\(from original_zone_ids attribute\\) does not match canonical order!"):
            _merge_beam_skims_to_zarr(
                all_skims_path=initial_main_zarr,
                iteration_skims_path=beam_iteration_zarr_0_based_int_ids,
                beam_output_dir="",
                settings=mock_settings,
            )
        mock_geoid_map.assert_called_once_with(mock_settings)

    @patch('pilates.beam.postprocessor.geoid_to_zone_map')
    def test_merge_beam_skims_to_zarr_1_based_int_ids_raises_error(
        self, mock_geoid_map, mock_settings, initial_main_zarr, beam_iteration_zarr_1_based_int_ids
    ):
        """
        Test that merging BEAM skims with 1-based integer zone IDs raises a ValueError
        due to non-0-based otaz coordinates.
        """
        # Arrange
        mock_geoid_map.return_value = {geoid: i for i, geoid in enumerate(CANONICAL_GEOID_ORDER)}

        # Act & Assert
        with pytest.raises(ValueError, match="FATAL: Skim zone order \\(from original_zone_ids attribute\\) does not match canonical order!"):
            _merge_beam_skims_to_zarr(
                all_skims_path=initial_main_zarr,
                iteration_skims_path=beam_iteration_zarr_1_based_int_ids,
                beam_output_dir="",
                settings=mock_settings,
            )
        mock_geoid_map.assert_called_once_with(mock_settings)