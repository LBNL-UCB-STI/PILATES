import pytest
import xarray as xr
import numpy as np
import yaml
from unittest.mock import patch, MagicMock
import json
import pandas as pd

from pilates.beam.postprocessor import (
    BeamPostprocessor,
    _merge_beam_skims_to_zarr,
    split_events_parquet_by_type,
    write_zarr_skim_as_omx_new,
)
from pilates.beam.outputs import BeamPostprocessOutputs
from pilates.generic.records import RecordStore, FileRecord
from pilates.config.models import load_config

# Use the same canonical order as the preprocessor test for consistency
CANONICAL_GEOID_ORDER = [f"5303300{i:04d}" for i in range(5)]
NUM_ZONES = len(CANONICAL_GEOID_ORDER)
TIME_PERIODS = ["EA", "AM", "MD", "PM", "EV"]


class _FakeOmxMatrix:
    def __init__(self, data):
        self.data = data
        self.attrs = {}


class _FakeOmxFile:
    def __init__(self, path: str):
        self.path = path
        self.mapping = None
        self.matrices = {}
        self.closed = False

    def create_mapping(self, name, values, overwrite=False):
        self.mapping = (name, list(values), overwrite)

    def __setitem__(self, key, value):
        self.matrices[key] = _FakeOmxMatrix(np.array(value))

    def __getitem__(self, key):
        return self.matrices[key]

    def close(self):
        self.closed = True


@pytest.fixture
def canonical_zones_geojson(tmp_path):
    """Create a dummy canonical_zones.geojson file."""
    geojson_path = tmp_path / "canonical_zones.geojson"
    # Create a minimal GeoJSON content
    geojson_content = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"zone_key": geoid, "TAZ": idx + 1},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
            for idx, geoid in enumerate(CANONICAL_GEOID_ORDER)
        ],
    }
    with open(geojson_path, "w") as f:
        json.dump(geojson_content, f)
    return geojson_path


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
                "periods": TIME_PERIODS,
            },
            "database": {"enabled": False, "type": "duckdb", "path": ""},
        },
        "infrastructure": {
            "container_manager": "docker",
            "docker_images": {},
            "docker_config": {},
        },
        "beam": {
            "config": "",
            "local_input_folder": "pilates/beam/production",
            "local_mutable_data_folder": "beam/input",
            "skims_shapefile": "shape/test_zones.shp",
            "skim_zone_geoid_col": "geoid10",
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
            "local_data_input_folder": str(tmp_path),  # Dummy path
            "input_file_template": "usim_data.h5",  # Dummy file
            "local_mutable_data_folder": "",
            "client_base_folder": "",
            "client_data_folder": "",
            "input_file_template_year": "",
            "output_file_template": "",
            "command_template": "",
            "region_mappings": {},
        },
    }
    config_path = tmp_path / "test_settings_postproc.yaml"
    with open(config_path, "w") as f:
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
    initial_sov_trips_data = np.zeros(
        (len(TIME_PERIODS), NUM_ZONES, NUM_ZONES), dtype=np.float32
    )
    initial_sov_failures_data = np.zeros(
        (len(TIME_PERIODS), NUM_ZONES, NUM_ZONES), dtype=np.float32
    )

    ds = xr.Dataset(
        {
            "SOV_TRIPS": (("time_period", "otaz", "dtaz"), initial_sov_trips_data),
            "SOV_FAILURES": (
                ("time_period", "otaz", "dtaz"),
                initial_sov_failures_data,
            ),
        },
        coords={
            "time_period": TIME_PERIODS,
            "otaz": np.arange(NUM_ZONES),
            "dtaz": np.arange(NUM_ZONES),
        },
        attrs={
            "description": "Initial main skims",
            "original_zone_ids": CANONICAL_GEOID_ORDER,  # Must have this for verification
        },
    )
    ds.to_zarr(main_zarr_path, mode="w", consolidated=True, zarr_format=2)
    return str(main_zarr_path)


@pytest.fixture
def beam_iteration_zarr_base(tmp_path):
    """
    Base fixture to create a partial Zarr skim file from a BEAM iteration.
    """
    beam_zarr_path = tmp_path / "beam_iteration.zarr"

    # Create sample data for one skim measure
    sov_trips_data = np.ones(
        (len(TIME_PERIODS), NUM_ZONES, NUM_ZONES), dtype=np.float32
    )

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
        },
    )
    return ds, str(beam_zarr_path)


@pytest.fixture
def beam_iteration_zarr_valid(beam_iteration_zarr_base):
    """Creates a valid BEAM Zarr with original_zone_ids in canonical order."""
    ds, path = beam_iteration_zarr_base
    ds.otaz.attrs["original_zone_ids"] = CANONICAL_GEOID_ORDER
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)
    return path


@pytest.fixture
def beam_iteration_zarr_scrambled(beam_iteration_zarr_base):
    """Creates a BEAM Zarr with original_zone_ids in a scrambled order."""
    ds, path = beam_iteration_zarr_base
    scrambled_order = CANONICAL_GEOID_ORDER[::-1]  # Reverse order
    ds.otaz.attrs["original_zone_ids"] = scrambled_order
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)
    return path


def test_split_events_parquet_by_type_filtered(tmp_path):
    pytest.importorskip("pyarrow")

    events_df = pd.DataFrame(
        {
            "type": ["EventA", "EventA", "EventB", "Event B", "PathTraversal"],
            "col_a": [1.0, 2.0, None, None, None],
            "col_b": [None, None, 3.0, 4.0, None],
            "col_c": [None, None, None, 5.0, None],
            "links": [None, None, None, None, "1,2,3"],
            "linkTravelTime": [None, None, None, None, "10,20,30"],
        }
    )
    events_path = tmp_path / "2.events.parquet"
    events_df.to_parquet(events_path, index=False)

    written, links_path = split_events_parquet_by_type(
        str(events_path),
        event_types=["EventA", "EventB", "Event B", "PathTraversal"],
        create_path_traversal_links=True,
    )

    assert len(written) == 4
    assert written["EventA"].endswith("2.events.EventA.parquet")
    assert written["EventB"].endswith("2.events.EventB.parquet")
    assert written["Event B"].endswith("2.events.Event_B.parquet")
    assert written["PathTraversal"].endswith("2.events.PathTraversal.parquet")
    assert links_path is not None
    assert links_path.endswith("2.events.PathTraversal.links.parquet")

    event_a_path = tmp_path / "2.events.EventA.parquet"
    event_b_path = tmp_path / "2.events.EventB.parquet"
    event_b_space_path = tmp_path / "2.events.Event_B.parquet"

    event_a = pd.read_parquet(event_a_path)
    assert event_a["type"].unique().tolist() == ["EventA"]
    assert "col_a" in event_a.columns
    assert "EventAEventId" in event_a.columns
    assert event_a["EventAEventId"].tolist() == list(range(len(event_a)))
    assert "col_b" not in event_a.columns
    assert "col_c" not in event_a.columns

    event_b = pd.read_parquet(event_b_path)
    assert event_b["type"].unique().tolist() == ["EventB"]
    assert "col_b" in event_b.columns
    assert "EventBEventId" in event_b.columns
    assert event_b["EventBEventId"].tolist() == list(range(len(event_b)))
    assert "col_a" not in event_b.columns
    assert "col_c" not in event_b.columns

    event_b_space = pd.read_parquet(event_b_space_path)
    assert event_b_space["type"].unique().tolist() == ["Event B"]
    assert "col_b" in event_b_space.columns
    assert "col_c" in event_b_space.columns
    assert "Event_BEventId" in event_b_space.columns
    assert event_b_space["Event_BEventId"].tolist() == list(range(len(event_b_space)))
    assert "col_a" not in event_b_space.columns

    path_traversal_path = tmp_path / "2.events.PathTraversal.parquet"
    path_traversal = pd.read_parquet(path_traversal_path)
    assert path_traversal["type"].unique().tolist() == ["PathTraversal"]
    assert "PathTraversalEventId" in path_traversal.columns
    assert "links" not in path_traversal.columns
    assert "linkTravelTime" not in path_traversal.columns

    link_table = pd.read_parquet(tmp_path / "2.events.PathTraversal.links.parquet")
    assert link_table["PathTraversalEventId"].unique().tolist() == [0]
    assert link_table["link_index"].tolist() == [0, 1, 2]
    assert link_table["linkId"].tolist() == [1, 2, 3]
    assert link_table["travelTimeSeconds"].tolist() == [10.0, 20.0, 30.0]

    record_store = RecordStore(
        recordList=[
            FileRecord(
                file_path=written["EventA"],
                short_name="events_parquet_2018_1_type_EventA",
            ),
            FileRecord(
                file_path=written["EventB"],
                short_name="events_parquet_2018_1_type_EventB",
            ),
            FileRecord(
                file_path=links_path,
                short_name="path_traversal_links_2018_1",
            ),
        ]
    )
    outputs = BeamPostprocessOutputs.from_record_store(record_store, tmp_path)
    assert "events_parquet_2018_1_type_EventA" in outputs.split_events
    assert "events_parquet_2018_1_type_EventB" in outputs.split_events
    assert "path_traversal_links_2018_1" in outputs.split_event_links


def test_write_zarr_skim_as_omx_new_uses_workspace_path_descales_transit_and_writes_original_lookup(
    tmp_path, mock_settings
):
    zarr_path = tmp_path / "skims.zarr"
    target_root = tmp_path / "workspace" / "beam" / "input"
    opened = {}

    ds = xr.Dataset(
        {
            "WLK_LOC_WLK_TOTIVT": (
                ("otaz", "dtaz", "time_period"),
                np.array(
                    [
                        [[100.0, 200.0], [300.0, 400.0]],
                        [[500.0, 600.0], [700.0, 800.0]],
                    ],
                    dtype=np.float32,
                ),
            )
        },
        coords={
            "otaz": np.arange(2),
            "dtaz": np.arange(2),
            "time_period": ["EA", "AM"],
        },
        attrs={"original_zone_ids": [101, 205]},
    )
    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)

    def _fake_open_file(path, mode):
        assert mode == "w"
        fake = _FakeOmxFile(path)
        opened["file"] = fake
        return fake

    workspace = MagicMock()
    workspace.get_beam_mutable_data_dir.return_value = str(target_root)

    with patch("pilates.beam.postprocessor.omx.open_file", side_effect=_fake_open_file):
        out = write_zarr_skim_as_omx_new(
            str(zarr_path),
            mock_settings,
            "final_skims.omx",
            workspace=workspace,
        )

    assert out == str(target_root / "seattle" / "final_skims.omx")
    assert (target_root / "seattle").is_dir()

    fake = opened["file"]
    assert fake.mapping == ("zone_id", [1, 2], False)
    np.testing.assert_array_equal(
        fake["WLK_LOC_WLK_TOTIVT__EA"].data,
        np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
    )
    assert fake["WLK_LOC_WLK_TOTIVT__EA"].attrs["mode"] == "WLK_LOC_WLK"
    assert fake["WLK_LOC_WLK_TOTIVT__EA"].attrs["measure"] == "TOTIVT"
    assert fake["WLK_LOC_WLK_TOTIVT__EA"].attrs["timePeriod"] == "EA"
    assert fake.closed is True


@patch("pilates.utils.zone_utils.load_canonical_zones")
def test_write_zarr_skim_as_omx_new_reconstructs_zone_ids_from_workspace(
    mock_load_canonical_zones, tmp_path, mock_settings
):
    zarr_path = tmp_path / "skims.zarr"
    target_root = tmp_path / "workspace" / "beam" / "input"
    opened = {}

    ds = xr.Dataset(
        {
            "SOV_TIME": (
                ("otaz", "dtaz"),
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
        },
        coords={"otaz": np.arange(2), "dtaz": np.arange(2)},
    )
    ds["otaz"].attrs["preprocessed"] = "zero-based-contiguous"
    ds["dtaz"].attrs["preprocessed"] = "zero-based-contiguous"
    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)

    mock_load_canonical_zones.return_value = pd.DataFrame(
        {"zone_key": [9001, 9002]}, index=[9001, 9002]
    )

    def _fake_open_file(path, mode):
        fake = _FakeOmxFile(path)
        opened["file"] = fake
        return fake

    workspace = MagicMock()
    workspace.get_beam_mutable_data_dir.return_value = str(target_root)

    with patch("pilates.beam.postprocessor.omx.open_file", side_effect=_fake_open_file):
        out = write_zarr_skim_as_omx_new(
            str(zarr_path),
            mock_settings,
            "final_skims.omx",
            workspace=workspace,
        )

    assert out == str(target_root / "seattle" / "final_skims.omx")
    assert opened["file"].mapping == ("zone_id", [1, 2], False)
    np.testing.assert_array_equal(
        opened["file"]["SOV_TIME"].data,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    mock_load_canonical_zones.assert_called_once_with(mock_settings, workspace)


def test_beam_postprocessor_expected_outputs_uses_omx_name_for_zarr_shared_skims(
    tmp_path, mock_settings
):
    mock_settings.shared.skims.fname = "skims.zarr"
    mock_settings.run.models.land_use = "urbansim"

    workspace = MagicMock()
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")

    outputs = BeamPostprocessor.expected_outputs(
        mock_settings,
        state=MagicMock(),
        workspace=workspace,
    )

    assert outputs == {
        "final_skims_omx": str(tmp_path / "beam" / "input" / "seattle" / "skims.omx")
    }


@pytest.fixture
def beam_iteration_zarr_missing_attr(beam_iteration_zarr_base):
    """Creates a BEAM Zarr without the original_zone_ids attribute."""
    ds, path = beam_iteration_zarr_base
    # Do not set ds.otaz.attrs['original_zone_ids']
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)
    return path


@pytest.fixture
def beam_iteration_zarr_0_based_int_ids(beam_iteration_zarr_base):
    """Creates a BEAM Zarr with original_zone_ids as [0, 1, 2, ...]."""
    ds, path = beam_iteration_zarr_base
    ds.otaz.attrs["original_zone_ids"] = [str(i) for i in range(NUM_ZONES)]
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)
    return path


@pytest.fixture
def beam_iteration_zarr_1_based_int_ids(beam_iteration_zarr_base):
    """Creates a BEAM Zarr with original_zone_ids as [1, 2, 3, ...]."""
    ds, path = beam_iteration_zarr_base
    ds.otaz.attrs["original_zone_ids"] = [str(i + 1) for i in range(NUM_ZONES)]
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)
    return path


class TestBeamPostprocessor:
    """Tests for the BEAM Postprocessor's skim merging logic."""

    @patch("pilates.utils.zone_utils.load_canonical_zones")
    @patch("pilates.beam.postprocessor.verify_skim_zone_order")
    def test_merge_beam_skims_to_zarr_valid(
        self,
        mock_verify_skim_zone_order,
        mock_load_canonical_zones,
        mock_settings,
        initial_main_zarr,
        beam_iteration_zarr_valid,
        tmp_path,
        canonical_zones_geojson,
    ):
        """
        Test the basic merge functionality with valid BEAM skims:
        - Verifies zone order before merging.
        - Checks that data from the partial skim is merged into the main skim.
        """
        # Arrange
        mock_load_canonical_zones.return_value = pd.DataFrame(
            {"zone_key": CANONICAL_GEOID_ORDER}, index=CANONICAL_GEOID_ORDER
        )
        mock_verify_skim_zone_order.return_value = (
            CANONICAL_GEOID_ORDER  # Mock its return value
        )
        mock_workspace = MagicMock()  # Create a mock workspace
        mock_workspace.get_asim_mutable_data_dir.return_value = tmp_path

        # Act
        _merge_beam_skims_to_zarr(
            all_skims_path=initial_main_zarr,
            iteration_skims_path=beam_iteration_zarr_valid,
            beam_output_dir="",  # Not used in this logic path
            settings=mock_settings,
            workspace=mock_workspace,  # Pass the mock workspace
        )

        # Assert
        mock_load_canonical_zones.assert_called_once_with(
            mock_settings, mock_workspace
        )  # Assert with workspace
        mock_verify_skim_zone_order.assert_called_once_with(
            mock_settings, beam_iteration_zarr_valid, mock_workspace
        )

        updated_ds = xr.open_zarr(initial_main_zarr)
        assert "SOV_TRIPS" in updated_ds.data_vars, (
            "SOV_TRIPS was not merged into the main Zarr file."
        )
        expected_data = np.ones(
            (len(TIME_PERIODS), NUM_ZONES, NUM_ZONES), dtype=np.float32
        )
        np.testing.assert_array_equal(
            updated_ds["SOV_TRIPS"].values,
            expected_data,
            "Data for SOV_TRIPS in main Zarr file is incorrect after merge.",
        )
        assert "original_zone_ids" in updated_ds.attrs
        assert updated_ds.attrs["original_zone_ids"] == CANONICAL_GEOID_ORDER

    @patch("pilates.utils.zone_utils.load_canonical_zones")
    @patch("pilates.beam.postprocessor.verify_skim_zone_order")
    def test_merge_beam_skims_to_zarr_scrambled_order_raises_error(
        self,
        mock_verify_skim_zone_order,
        mock_load_canonical_zones,
        mock_settings,
        initial_main_zarr,
        beam_iteration_zarr_scrambled,
        tmp_path,
        canonical_zones_geojson,
    ):
        """
        Test that merging BEAM skims with scrambled zone order raises a ValueError.
        """
        # Arrange
        mock_load_canonical_zones.return_value = pd.DataFrame(
            {"zone_key": CANONICAL_GEOID_ORDER}, index=CANONICAL_GEOID_ORDER
        )
        mock_workspace = MagicMock()
        mock_workspace.get_asim_mutable_data_dir.return_value = tmp_path
        mock_verify_skim_zone_order.side_effect = ValueError(
            "FATAL: Skim zone order (from original_zone_ids attribute) does not match canonical order!"
        )

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="FATAL: Skim zone order \\(from original_zone_ids attribute\\) does not match canonical order!",
        ):
            _merge_beam_skims_to_zarr(
                all_skims_path=initial_main_zarr,
                iteration_skims_path=beam_iteration_zarr_scrambled,
                beam_output_dir="",
                settings=mock_settings,
                workspace=mock_workspace,
            )
        mock_load_canonical_zones.assert_called_once_with(mock_settings, mock_workspace)
        mock_verify_skim_zone_order.assert_called_once_with(
            mock_settings, beam_iteration_zarr_scrambled, mock_workspace
        )

    @patch("pilates.utils.zone_utils.load_canonical_zones")
    @patch("pilates.beam.postprocessor.verify_skim_zone_order")
    def test_merge_beam_skims_to_zarr_missing_attr_raises_error(
        self,
        mock_verify_skim_zone_order,
        mock_load_canonical_zones,
        mock_settings,
        initial_main_zarr,
        beam_iteration_zarr_missing_attr,
        tmp_path,
        canonical_zones_geojson,
    ):
        """
        Test that merging BEAM skims missing the original_zone_ids attribute raises a ValueError.
        """
        # Arrange
        mock_load_canonical_zones.return_value = pd.DataFrame(
            {"zone_key": CANONICAL_GEOID_ORDER}, index=CANONICAL_GEOID_ORDER
        )
        mock_workspace = MagicMock()
        mock_workspace.get_asim_mutable_data_dir.return_value = tmp_path
        mock_verify_skim_zone_order.side_effect = ValueError(
            "Zarr file does not contain 'original_zone_ids' metadata."
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Zarr file does not contain 'original_zone_ids' metadata."
        ):
            _merge_beam_skims_to_zarr(
                all_skims_path=initial_main_zarr,
                iteration_skims_path=beam_iteration_zarr_missing_attr,
                beam_output_dir="",
                settings=mock_settings,
                workspace=mock_workspace,
            )
        mock_load_canonical_zones.assert_called_once_with(mock_settings, mock_workspace)
        mock_verify_skim_zone_order.assert_called_once_with(
            mock_settings, beam_iteration_zarr_missing_attr, mock_workspace
        )

    @patch("pilates.utils.zone_utils.load_canonical_zones")
    @patch("pilates.beam.postprocessor.verify_skim_zone_order")
    def test_merge_beam_skims_to_zarr_0_based_int_ids_raises_error(
        self,
        mock_verify_skim_zone_order,
        mock_load_canonical_zones,
        mock_settings,
        initial_main_zarr,
        beam_iteration_zarr_0_based_int_ids,
        tmp_path,
        canonical_zones_geojson,
    ):
        """
        Test that merging BEAM skims with 0-based integer zone IDs raises a ValueError.
        """
        # Arrange
        mock_load_canonical_zones.return_value = pd.DataFrame(
            {"zone_key": CANONICAL_GEOID_ORDER}, index=CANONICAL_GEOID_ORDER
        )
        mock_workspace = MagicMock()
        mock_workspace.get_asim_mutable_data_dir.return_value = tmp_path
        mock_verify_skim_zone_order.side_effect = ValueError(
            "FATAL: Zarr 'otaz' coordinates are not 0-based as expected!"
        )

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="FATAL: Zarr 'otaz' coordinates are not 0-based as expected!",
        ):
            _merge_beam_skims_to_zarr(
                all_skims_path=initial_main_zarr,
                iteration_skims_path=beam_iteration_zarr_0_based_int_ids,
                beam_output_dir="",
                settings=mock_settings,
                workspace=mock_workspace,
            )
        mock_load_canonical_zones.assert_called_once_with(mock_settings, mock_workspace)
        mock_verify_skim_zone_order.assert_called_once_with(
            mock_settings, beam_iteration_zarr_0_based_int_ids, mock_workspace
        )

    @patch("pilates.utils.zone_utils.load_canonical_zones")
    @patch("pilates.beam.postprocessor.verify_skim_zone_order")
    def test_merge_beam_skims_to_zarr_1_based_int_ids_raises_error(
        self,
        mock_verify_skim_zone_order,
        mock_load_canonical_zones,
        mock_settings,
        initial_main_zarr,
        beam_iteration_zarr_1_based_int_ids,
        tmp_path,
        canonical_zones_geojson,
    ):
        """
        Test that merging BEAM skims with 1-based integer zone IDs raises a ValueError
        due to non-0-based otaz coordinates.
        """
        # Arrange
        mock_load_canonical_zones.return_value = pd.DataFrame(
            {"zone_key": CANONICAL_GEOID_ORDER}, index=CANONICAL_GEOID_ORDER
        )
        mock_workspace = MagicMock()
        mock_workspace.get_asim_mutable_data_dir.return_value = tmp_path
        mock_verify_skim_zone_order.side_effect = ValueError(
            "FATAL: Zarr 'otaz' coordinates are not 0-based as expected!"
        )

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="FATAL: Zarr 'otaz' coordinates are not 0-based as expected!",
        ):
            _merge_beam_skims_to_zarr(
                all_skims_path=initial_main_zarr,
                iteration_skims_path=beam_iteration_zarr_1_based_int_ids,
                beam_output_dir="",
                settings=mock_settings,
                workspace=mock_workspace,
            )
        mock_load_canonical_zones.assert_called_once_with(mock_settings, mock_workspace)
        mock_verify_skim_zone_order.assert_called_once_with(
            mock_settings, beam_iteration_zarr_1_based_int_ids, mock_workspace
        )
