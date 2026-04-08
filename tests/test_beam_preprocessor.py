import os
import pytest
import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock
import yaml
import json

gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely.geometry")
from shapely.geometry import Polygon

from pilates.config.models import load_config
from pilates.activitysim.outputs import normalize_asim_output_key
from pilates.beam.preprocessor import BeamPreprocessor

# Define a canonical order for GEOIDs that our test will enforce
CANONICAL_GEOID_ORDER = [f"5303300{i:04d}" for i in range(5)]


# GeoPandas/Shapely can be present but non-functional (GEOS/compiled deps mismatch).
# Skip these tests rather than failing with confusing TypeErrors during setup.
try:
    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
except Exception as e:
    pytest.skip(
        f"Shapely Polygon is not usable in this environment: {e}",
        allow_module_level=True,
    )

try:
    gpd.GeoDataFrame({"a": [1, 2]})
except Exception as e:
    pytest.skip(
        f"GeoPandas GeoDataFrame is not usable in this environment: {e}",
        allow_module_level=True,
    )


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

    try:
        gdf.to_file(shapefile_path, driver="ESRI Shapefile")
    except Exception as e:
        pytest.skip(f"GeoPandas cannot write shapefiles in this environment: {e}")
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


from pilates.generic.records import FileRecord, RecordStore

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
        the asim_idx column, and updates the BEAM config.
        """
        # Arrange
        mock_state = SimpleNamespace(
            full_settings=mock_settings,
            current_year=2020,
            current_inner_iter=0,
            run_info_path=None,  # Mock as needed
        )
        # Instantiate the preprocessor
        preprocessor = BeamPreprocessor(
            model_name="beam",
            state=mock_state,
        )

        # Act
        # Call the public method that performs the sorting and config update
        output_shapefile_path = preprocessor.prepare_beam_zone_shapefile(mock_workspace)

        # Assert
        # 1. Verify the shapefile was modified correctly
        sorted_gdf = gpd.read_file(output_shapefile_path)

        # 1a. Check if the order of GEOIDs now matches the canonical order
        final_order = sorted_gdf[mock_settings.beam.skim_zone_geoid_col].tolist()
        assert (
            final_order == CANONICAL_GEOID_ORDER
        ), "Shapefile GEOID order does not match canonical order after sorting."

        # 2. Verify the new shapefile is used in the BEAM config
        assert mock_settings.beam.skim_zone_geoid_col in sorted_gdf.columns


def test_prepare_beam_zone_shapefile_preserves_canonical_order_when_sort_col_missing(
    mock_settings, mock_workspace, caplog
):
    mock_state = SimpleNamespace(
        full_settings=mock_settings,
        current_year=2020,
        current_inner_iter=0,
        run_info_path=None,
    )
    preprocessor = BeamPreprocessor(model_name="beam", state=mock_state)
    object.__setattr__(mock_settings.beam, "skim_zone_geoid_col", "geoid10")

    with caplog.at_level("WARNING"):
        output_shapefile_path = preprocessor.prepare_beam_zone_shapefile(mock_workspace)

    sorted_gdf = gpd.read_file(output_shapefile_path)

    assert sorted_gdf["TAZ"].astype(str).tolist() == CANONICAL_GEOID_ORDER
    assert "not found in canonical zones export" in caplog.text


def test_prepare_beam_zone_shapefile_preserves_canonical_order_when_sort_col_conflicts(
    mock_settings, mock_workspace, monkeypatch, caplog
):
    mock_state = SimpleNamespace(
        full_settings=mock_settings,
        current_year=2020,
        current_inner_iter=0,
        run_info_path=None,
    )
    preprocessor = BeamPreprocessor(model_name="beam", state=mock_state)
    object.__setattr__(mock_settings.beam, "skim_zone_geoid_col", "geoid10")

    scrambled_geoids = CANONICAL_GEOID_ORDER[::-1]
    canonical_gdf = gpd.GeoDataFrame(
        {
            "district": range(len(CANONICAL_GEOID_ORDER)),
            "geoid10": scrambled_geoids,
            "geometry": [None] * len(CANONICAL_GEOID_ORDER),
        },
        index=pd.Index(CANONICAL_GEOID_ORDER, name="TAZ"),
    )

    monkeypatch.setattr(
        "pilates.utils.zone_utils.load_canonical_zones",
        lambda settings, workspace: canonical_gdf,
    )

    with caplog.at_level("WARNING"):
        output_shapefile_path = preprocessor.prepare_beam_zone_shapefile(mock_workspace)

    sorted_gdf = gpd.read_file(output_shapefile_path)

    assert sorted_gdf["TAZ"].astype(str).tolist() == CANONICAL_GEOID_ORDER
    assert "would reorder canonical zones" in caplog.text


def test_preprocess_ignores_workspace_beam_output_cache(monkeypatch, mock_settings, mock_workspace):
    state = SimpleNamespace(
        full_settings=mock_settings,
        current_year=2020,
        current_inner_iter=0,
        forecast_year=2020,
        run_info_path=None,
    )
    preprocessor = BeamPreprocessor(
        model_name="beam",
        state=state,
    )
    object.__setattr__(preprocessor.settings, "vehicle_ownership_model_enabled", False)
    object.__setattr__(
        preprocessor.settings,
        "activitysim",
        SimpleNamespace(file_format="parquet"),
    )

    cached_beam_record = FileRecord(
        file_path="/tmp/cached_beam_plans.parquet",
        short_name="beam_plans",
        description="stale BEAM cache record",
    )
    mock_workspace.output_data = {
        "beam": RecordStore(recordList=[cached_beam_record]),
    }

    previous_records = RecordStore(
        recordList=[
            FileRecord(
                file_path="/tmp/households.parquet",
                short_name="households_asim_out",
                description="fresh ActivitySim households",
            )
        ]
    )

    captured = {}

    monkeypatch.setattr(preprocessor, "_update_beam_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessor,
        "prepare_beam_zone_shapefile",
        lambda _workspace: None,
    )
    monkeypatch.setattr(
        preprocessor,
        "_copy_vehicles_from_atlas",
        lambda _workspace: None,
    )
    monkeypatch.setattr(
        preprocessor,
        "_handle_linkstats",
        lambda _workspace, _previous_beam_records, _store: None,
    )
    monkeypatch.setattr(preprocessor, "_activity_demand_enabled", lambda: True)

    def _capture_input_records(input_records, _workspace):
        captured["keys"] = [record.short_name for record in input_records.all_records()]
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path="/tmp/plans.parquet",
                    short_name="plans_beam_in",
                    description="mock staged plans",
                ),
                FileRecord(
                    file_path="/tmp/households.parquet",
                    short_name="households_beam_in",
                    description="mock staged households",
                ),
                FileRecord(
                    file_path="/tmp/persons.parquet",
                    short_name="persons_beam_in",
                    description="mock staged persons",
                ),
            ]
        )

    monkeypatch.setattr(preprocessor, "_copy_plans_from_asim", _capture_input_records)

    preprocessor._preprocess(mock_workspace, previous_records=previous_records)

    assert captured["keys"] == ["households_asim_out"]


def test_normalize_asim_output_key_maps_plans_alias() -> None:
    assert normalize_asim_output_key("plans") == "beam_plans_asim_out"


def test_copy_plans_from_asim_accepts_plans_asim_out_alias(
    monkeypatch, mock_settings, mock_workspace, tmp_path
):
    state = SimpleNamespace(
        full_settings=mock_settings,
        current_year=2020,
        current_inner_iter=0,
        forecast_year=2020,
        run_info_path=None,
    )
    preprocessor = BeamPreprocessor(
        model_name="beam",
        state=state,
    )
    object.__setattr__(preprocessor.settings, "vehicle_ownership_model_enabled", False)
    object.__setattr__(
        preprocessor.settings,
        "activitysim",
        SimpleNamespace(file_format="parquet"),
    )
    mock_workspace.get_asim_output_dir.return_value = str(tmp_path / "activitysim" / "output")

    plans_path = tmp_path / "activitysim" / "output" / "year-2020-iteration-0" / "plans.parquet"
    plans_path.parent.mkdir(parents=True, exist_ok=True)
    plans_path.write_text("plans", encoding="utf-8")

    input_records = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(plans_path),
                short_name="plans_asim_out",
                description="ActivitySim plans alias",
            )
        ]
    )

    captured = {}

    def _capture_copy_initial(asim_file_paths, _file_format, _workspace):
        captured["asim_file_paths"] = dict(asim_file_paths)
        return []

    monkeypatch.setattr(preprocessor, "_copy_initial_asim_files", _capture_copy_initial)

    preprocessor._copy_plans_from_asim(input_records, mock_workspace)

    assert "beam_plans" in captured["asim_file_paths"]
    assert captured["asim_file_paths"]["beam_plans"][0] == str(plans_path)
