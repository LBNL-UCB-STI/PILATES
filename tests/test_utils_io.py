import pytest
from unittest.mock import MagicMock, patch

from pilates.utils.io import (
    compute_model_enabled_flags,
    parse_args_and_settings,
)

# Mock the settings_helper.get and config.models.load_config for all tests
@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("pilates.utils.io.get_setting") as mock_get_setting, \
         patch("pilates.utils.io.load_config") as mock_load_config:
        yield mock_get_setting, mock_load_config

class MockSettings:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_enabled_models(self):
        return ["model1", "model2"]

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


@pytest.fixture
def mock_settings_data():
    return {
        "run": {
            "models": {
                "land_use": "urbansim",
                "vehicle_ownership": "atlas",
                "activity_demand": "activitysim",
                "travel": "beam",
            }
        },
        "warm_start_skims": False,
        "static_skims": False,
        "activitysim": {"replan_iters": 1},
        "replan_iters": 1, # legacy
        "land_use_model": "urbansim", # legacy
        "vehicle_ownership_model": "atlas", # legacy
        "activity_demand_model": "activitysim", # legacy
        "travel_model": "beam", # legacy
    }


class TestComputeModelEnabledFlags:
    def setup_method(self):
        # This side effect simulates get_setting on a nested dict-like object.
        # It's defined here to be reused across all tests in this class.
        def side_effect(settings_obj, key, default=None):
            # The settings_obj is a MockSettings instance which holds the data dict.
            val = settings_obj._data
            try:
                for part in key.split("."):
                    val = val[part]
                return val
            except (KeyError, TypeError):
                # If nested access fails, try to get the key from the top level (for legacy flat keys)
                return settings_obj._data.get(key, default)

        self.side_effect = side_effect

    def test_all_models_enabled_nested_settings(self, mock_dependencies, mock_settings_data):
        mock_get_setting, _ = mock_dependencies
        mock_get_setting.side_effect = self.side_effect

        settings = MockSettings(mock_settings_data)
        flags = compute_model_enabled_flags(settings)
        assert flags["land_use_enabled"] is True
        assert flags["vehicle_ownership_model_enabled"] is True
        assert flags["activity_demand_enabled"] is True
        assert flags["traffic_assignment_enabled"] is True
        assert flags["replanning_enabled"] is True

    def test_all_models_enabled_flat_settings(self, mock_dependencies, mock_settings_data):
        mock_get_setting, _ = mock_dependencies
        mock_get_setting.side_effect = self.side_effect

        settings = MockSettings(mock_settings_data)
        flags = compute_model_enabled_flags(settings)
        assert flags["land_use_enabled"] is True
        assert flags["vehicle_ownership_model_enabled"] is True
        assert flags["activity_demand_enabled"] is True
        assert flags["traffic_assignment_enabled"] is True
        assert flags["replanning_enabled"] is True

    def test_land_use_disabled_by_warm_start_skims(self, mock_dependencies, mock_settings_data):
        mock_get_setting, _ = mock_dependencies
        mock_settings_data["warm_start_skims"] = True
        mock_get_setting.side_effect = self.side_effect

        settings = MockSettings(mock_settings_data)
        flags = compute_model_enabled_flags(settings)
        assert flags["land_use_enabled"] is False
        assert flags["traffic_assignment_enabled"] is True # Should not be affected

    def test_traffic_assignment_disabled_by_static_skims(self, mock_dependencies, mock_settings_data):
        mock_get_setting, _ = mock_dependencies
        mock_settings_data["static_skims"] = True
        mock_get_setting.side_effect = self.side_effect

        settings = MockSettings(mock_settings_data)
        flags = compute_model_enabled_flags(settings)
        assert flags["traffic_assignment_enabled"] is False
        assert flags["land_use_enabled"] is True # Should not be affected

    def test_replanning_disabled_by_replan_iters_zero(self, mock_dependencies, mock_settings_data):
        mock_get_setting, _ = mock_dependencies
        mock_settings_data["activitysim"]["replan_iters"] = 0
        mock_settings_data["replan_iters"] = 0 # legacy
        mock_get_setting.side_effect = self.side_effect

        settings = MockSettings(mock_settings_data)
        flags = compute_model_enabled_flags(settings)
        assert flags["replanning_enabled"] is False

    def test_replanning_disabled_for_polaris_activity_demand(self, mock_dependencies, mock_settings_data):
        mock_get_setting, _ = mock_dependencies
        mock_settings_data["run"]["models"]["activity_demand"] = "polaris"
        mock_settings_data["activity_demand_model"] = "polaris" # legacy
        mock_get_setting.side_effect = self.side_effect

        settings = MockSettings(mock_settings_data)
        flags = compute_model_enabled_flags(settings)
        assert flags["replanning_enabled"] is False
        
        
@pytest.fixture
def mock_settings_pydantic():
            settings = MagicMock()
            settings.run.region = "sfbay"
            settings.run.start_year = 2020
            settings.run.end_year = 2025
            settings.get_enabled_models.return_value = ["urbansim", "atlas"]
            return settings
        
        
class TestParseArgsAndSettings:
            @patch("argparse.ArgumentParser")
            @patch("pilates.utils.io.compute_model_enabled_flags")
            def test_default_args_and_settings(self, mock_compute_flags, mock_arg_parser, mock_dependencies, mock_settings_pydantic):
                mock_get_setting, mock_load_config = mock_dependencies
                mock_load_config.return_value = mock_settings_pydantic
        
                # Mock argparse
                mock_args = MagicMock()
                mock_args.config = "settings.yaml"
                mock_args.stage = "current_stage.yaml"
                mock_arg_parser.return_value.parse_args.return_value = mock_args
        
                # Mock compute_model_enabled_flags
                mock_compute_flags.return_value = {
                    "land_use_enabled": True,
                    "vehicle_ownership_model_enabled": True,
                    "activity_demand_enabled": True,
                    "traffic_assignment_enabled": True,
                    "replanning_enabled": True,
                }
        
                # Mock get_setting for the ValueError checks
                mock_get_setting.side_effect = [
                    0, # activitysim.household_sample_size
                    0, # atlas.beamac
                    "sfbay", # run.region
                    "taz", # shared.skims.zone_type
                ]
        
                settings = parse_args_and_settings()
        
                mock_arg_parser.assert_called_once()
                mock_arg_parser.return_value.parse_args.assert_called_once()
                mock_load_config.assert_called_with("settings.yaml")
                mock_compute_flags.assert_called_once_with(settings)
        
                assert settings.state_file_loc == "current_stage.yaml"
                assert settings.settings_file == "settings.yaml"
                assert settings.land_use_enabled is True
                assert settings.vehicle_ownership_model_enabled is True
        
            @patch("argparse.ArgumentParser")
            @patch("pilates.utils.io.compute_model_enabled_flags")
            def test_custom_config_and_stage_files(self, mock_compute_flags, mock_arg_parser, mock_dependencies, mock_settings_pydantic):
                mock_get_setting, mock_load_config = mock_dependencies
                mock_load_config.return_value = mock_settings_pydantic
        
                # Mock argparse
                mock_args = MagicMock()
                mock_args.config = "custom_settings.yaml"
                mock_args.stage = "custom_stage.yaml"
                mock_arg_parser.return_value.parse_args.return_value = mock_args
        
                # Mock compute_model_enabled_flags
                mock_compute_flags.return_value = {
                    "land_use_enabled": True,
                    "vehicle_ownership_model_enabled": True,
                    "activity_demand_enabled": True,
                    "traffic_assignment_enabled": True,
                    "replanning_enabled": True,
                }
        
                # Mock get_setting for the ValueError checks
                mock_get_setting.side_effect = [
                    0, # activitysim.household_sample_size
                    0, # atlas.beamac
                    "sfbay", # run.region
                    "taz", # shared.skims.zone_type
                ]
        
                settings = parse_args_and_settings()
        
                mock_load_config.assert_called_with("custom_settings.yaml")
                assert settings.state_file_loc == "custom_stage.yaml"
                assert settings.settings_file == "custom_settings.yaml"
        
            @patch("argparse.ArgumentParser")
            @patch("pilates.utils.io.compute_model_enabled_flags")
            def test_household_sample_size_value_error(self, mock_compute_flags, mock_arg_parser, mock_dependencies, mock_settings_pydantic):
                mock_get_setting, mock_load_config = mock_dependencies
                mock_load_config.return_value = mock_settings_pydantic
        
                # Mock argparse
                mock_args = MagicMock()
                mock_args.config = "settings.yaml"
                mock_args.stage = "current_stage.yaml"
                mock_arg_parser.return_value.parse_args.return_value = mock_args
        
                # Mock compute_model_enabled_flags to enable land use
                mock_compute_flags.return_value = {
                    "land_use_enabled": True,
                    "vehicle_ownership_model_enabled": True,
                    "activity_demand_enabled": True,
                    "traffic_assignment_enabled": True,
                    "replanning_enabled": True,
                }
        
                # Mock get_setting to trigger the ValueError
                mock_get_setting.side_effect = [
                    100, # activitysim.household_sample_size > 0
                    0, # atlas.beamac
                    "sfbay", # run.region
                    "taz", # shared.skims.zone_type
                ]
        
                with pytest.raises(ValueError, match='Land use models must be disabled'):
                    parse_args_and_settings()
        
            @patch("argparse.ArgumentParser")
            @patch("pilates.utils.io.compute_model_enabled_flags")
            def test_atlas_beamac_value_error(self, mock_compute_flags, mock_arg_parser, mock_dependencies, mock_settings_pydantic):
                mock_get_setting, mock_load_config = mock_dependencies
                mock_load_config.return_value = mock_settings_pydantic
        
                # Mock argparse
                mock_args = MagicMock()
                mock_args.config = "settings.yaml"
                mock_args.stage = "current_stage.yaml"
                mock_arg_parser.return_value.parse_args.return_value = mock_args
        
                # Mock compute_model_enabled_flags
                mock_compute_flags.return_value = {
                    "land_use_enabled": True,
                    "vehicle_ownership_model_enabled": True,
                    "activity_demand_enabled": True,
                    "traffic_assignment_enabled": True,
                    "replanning_enabled": True,
                }
        
                # Mock get_setting to trigger the ValueError
                mock_get_setting.side_effect = [
                    0, # activitysim.household_sample_size
                    1, # atlas.beamac > 0
                    "sfbay", # run.region
                    "other_zone", # shared.skims.zone_type != "taz"
                ]
                
