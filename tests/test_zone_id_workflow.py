import os
import shutil
import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from unittest.mock import patch
import xarray as xr

from pilates.config import PilatesConfig
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.zone_utils import load_canonical_zones
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.beam.postprocessor import verify_skim_zone_order
from pilates.utils.provenance import FileProvenanceTracker


class TestZoneIdWorkflow:
    """
    This test class provides a comprehensive, narrative-style test of the
    end-to-end zone ID management and alignment workflow in PILATES.

    It is designed to serve as both a validation of the system's correctness
    and as executable documentation for developers.

    The test simulates a workflow where the initial zone definitions are
    intentionally out of order, and then asserts that each component of the
    framework correctly sorts, processes, and verifies this data, ensuring
    all models operate on a consistent and canonical zone order.
    """

    def setup_method(self):
        """
        Set up a temporary environment for the test run.
        This method creates all necessary directories, configuration files, and
        dummy data required to execute the workflow steps.
        """
        # Create a temporary root directory for the test
        self.test_dir = os.path.join(os.path.dirname(__file__), "tmp_zone_test")
        os.makedirs(self.test_dir, exist_ok=True)

        # --- Create Dummy Source Data and Directories ---
        self.source_data_dir = os.path.join(self.test_dir, "source_data")
        os.makedirs(self.source_data_dir, exist_ok=True)

        # Create a dummy canonical zone file with OUT-OF-ORDER zone IDs
        self.canonical_zone_source_path = os.path.join(
            self.source_data_dir, "canonical_zones.geojson"
        )
        # Define zones with intentionally unordered IDs: 3, 1, 2
        zones_data = {
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            ],
            "CANONICAL_ID": [3, 1, 2],
        }
        self.gdf_source = gpd.GeoDataFrame(zones_data, crs="EPSG:4326")
        self.gdf_source.to_file(self.canonical_zone_source_path, driver="GeoJSON")

        # Expected sorted order for assertions
        self.expected_sorted_ids = ["1", "2", "3"]

        # Create a minimal settings object
        self.settings = self._create_dummy_config()

        # Define run folder
        run_folder_name = "test_run"
        run_dir = os.path.join(self.test_dir, run_folder_name)
        run_id_for_provenance = "test_run_id"  # A simple ID for the test run

        # Initialize a dummy provenance tracker
        self.provenance_tracker = FileProvenanceTracker(
            run_id=run_id_for_provenance,
            output_path=self.test_dir,
            folder_name=run_folder_name,
        )

        # --- Create Workspace and Pilates Objects ---
        self.workspace = Workspace(
            settings=self.settings,
            output_path=self.test_dir,
            folder_name=run_folder_name,
            provenance_tracker=self.provenance_tracker,
        )

        # Set a path for the state file and initialize WorkflowState
        self.settings.state_file_loc = os.path.join(self.test_dir, "current_stage.yaml")
        self.state = WorkflowState.from_settings(self.settings)

        # Manually create the directories that the preprocessors will need,
        # as the main `run.py` logic that normally does this is not executed.
        os.makedirs(self.workspace.get_usim_mutable_data_dir(), exist_ok=True)
        os.makedirs(self.workspace.get_beam_output_dir(), exist_ok=True)
        os.makedirs(self.workspace.get_asim_mutable_data_dir(), exist_ok=True)

    def teardown_method(self):
        """
        Clean up the temporary test environment after the test run.
        """
        shutil.rmtree(self.test_dir)

    def _create_dummy_config(self):
        """
        Creates a minimal PilatesConfig object for the test.
        """
        config_dict = {
            "run": {
                "region": "test_region",
                "scenario": "test_scenario",
                "start_year": 2020,
                "end_year": 2020,
                "output_directory": self.test_dir,
                "output_run_name": "test_run",
                "models": {
                    "land_use": "urbansim",
                    "travel": "beam",
                    "activity_demand": "activitysim",
                    "vehicle_ownership": "atlas",
                },
            },
            "shared": {
                "geography": {
                    "zones": {
                        "source_file": self.canonical_zone_source_path,
                        "canonical_id_col": "CANONICAL_ID",
                        "activitysim_index_col": "TAZ",
                        "zone_type": "taz",
                    },
                    "FIPS": {"state": "00", "counties": ["001"]},
                    "local_crs": "EPSG:4326",
                },
                "skims": {"fname": "skims.zarr", "hwy_paths": [], "periods": []},
                "database": {"enabled": False, "path": "dummy.db"},
            },
            "infrastructure": {
                "container_manager": "docker",
                "docker_images": {"pilates": "pilates:latest"},
            },
            "urbansim": {
                "datastore": "dummy.h5",
                "local_data_input_folder": "urbansim/data",
                "local_mutable_data_folder": "urbansim/data",
                "client_base_folder": "/urbansim",
                "client_data_folder": "data",
                "input_file_template": "urbansim_data.h5",
                "input_file_template_year": "urbansim_data_{year}.h5",
                "output_file_template": "urbansim_output_{year}.h5",
                "command_template": "python run.py",
            },
            "activitysim": {
                "local_mutable_data_folder": "activitysim/data",
                "local_mutable_configs_folder": "activitysim/configs",
                "main_configs_dir": "configs",
                "local_input_folder": "activitysim/data",
                "local_output_folder": "activitysim/output",
                "local_configs_folder": "activitysim/configs",
                "validation_folder": "activitysim/validation",
                "command_template": "python run.py",
                "final_plans_folder": "plans",
            },
            "beam": {
                "config": "beam.conf",
                "local_input_folder": "beam/input",
                "local_output_folder": "beam/output",
                "local_mutable_data_folder": "beam/data",
                "router_directory": "r5",
                "scenario_folder": "beam/scenario",
                "skims_shapefile": "skims.shp",
                "skim_zone_geoid_col": "GEOID",
                "skim_zone_source_id_col": "TAZ",
            },
            "atlas": {
                "host_input_folder": "atlas/input",
                "warmstart_input_folder": "atlas/input",
                "host_mutable_input_folder": "atlas/input",
                "host_output_folder": "atlas/output",
                "container_input_folder": "/atlas/input",
                "container_output_folder": "/atlas/output",
                "basedir": "/atlas",
                "codedir": "/atlas/code",
                "command_template": "Rscript run.R",
            },
        }
        return PilatesConfig(**config_dict)

    @patch("pilates.activitysim.preprocessor._get_college_enrollment")
    @patch("pilates.activitysim.preprocessor._get_school_enrollment")
    @patch("pilates.utils.geog.get_county_block_geoms")
    @patch("pilates.beam.preprocessor.BeamPreprocessor.copy_data_to_mutable_location")
    @patch(
        "pilates.activitysim.preprocessor.ActivitysimPreprocessor.copy_data_to_mutable_location"
    )
    def test_zone_id_end_to_end_alignment_workflow(
        self,
        mock_asim_copy,
        mock_beam_copy,
        mock_download_geoms,
        mock_get_school_enrollment,
        mock_get_college_enrollment,
    ):
        """
        This test method executes the full zone ID management workflow,
        acting as a narrative test and executable documentation.
        """
        # Configure the mock to return a dummy DataFrame for college enrollment
        dummy_college_enrollment_data = pd.DataFrame(
            {
                "ncessch": ["college1", "college2"],
                "county_code": ["001", "001"],
                "latitude": [37.5, 37.6],
                "longitude": [-122.5, -122.6],
                "full_time_enrollment": [500, 1000],
                "part_time_enrollment": [50, 100],
                "x": [-122.5, -122.6],
                "y": [37.5, 37.6],
            }
        )
        mock_get_college_enrollment.return_value = dummy_college_enrollment_data

        # Configure the mock to return a dummy DataFrame for school enrollment
        dummy_enrollment_data = pd.DataFrame(
            {
                "ncessch": ["school1", "school2", "school3"],
                "county_code": [
                    "001",
                    "001",
                    "001",
                ],  # Assuming county_code '001' from settings
                "latitude": [37.0, 37.1, 37.2],
                "longitude": [-122.0, -122.1, -122.2],
                "enrollment": [100, 200, 150],
                "x": [-122.0, -122.1, -122.2],  # Add dummy x coordinates
                "y": [37.0, 37.1, 37.2],  # Add dummy y coordinates
            }
        )
        mock_get_school_enrollment.return_value = dummy_enrollment_data

        # Configure the mock to return a dummy GeoDataFrame for block geoms
        dummy_block_geoms = gpd.GeoDataFrame(
            {
                "GEOID": ["1", "2", "3"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                    Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
                ],
            },
            crs="EPSG:4326",
        )
        mock_download_geoms.return_value = dummy_block_geoms

        # This test follows the data flow of zone IDs, ensuring they are
        # correctly sorted and aligned at each stage of the process.

        # =====================================================================
        # STEP 1: Initial Data Loading and Canonical Sorting
        # =====================================================================
        # The first step in the workflow is to copy the source data to a
        # mutable location that the models can use. We then test the core
        # `load_canonical_zones` function to ensure it correctly reads our
        # intentionally out-of-order source file and produces a sorted,
        # canonical DataFrame. This is the foundation of the entire workflow.

        # 1a. Initialize the ActivitySim preprocessor and copy source data
        asim_preprocessor = ActivitysimPreprocessor(
            "activitysim", self.state, self.provenance_tracker
        )
        asim_preprocessor.copy_data_to_mutable_location(
            self.settings, self.workspace.get_asim_mutable_data_dir()
        )

        # Manually copy the file to simulate the outcome of the mocked function
        copied_zone_file = os.path.join(
            self.workspace.get_asim_mutable_data_dir(), "canonical_zones.geojson"
        )
        shutil.copy(self.canonical_zone_source_path, copied_zone_file)

        # 1b. Assert that the source file was copied to the mutable location
        assert os.path.exists(
            copied_zone_file
        ), "Canonical zone file was not copied to mutable directory."

        # 1c. Call the authoritative `load_canonical_zones` function
        sorted_zones_gdf = load_canonical_zones(self.settings, self.workspace)

        # 1d. Assert that the returned GeoDataFrame is correctly sorted
        assert isinstance(sorted_zones_gdf, gpd.GeoDataFrame)
        assert not sorted_zones_gdf.empty
        # CRITICAL ASSERTION: The index should now be sorted numerically,
        # overriding the out-of-order IDs [3, 1, 2] from the source file.
        # The IDs are strings, as defined by the function's contract.
        assert sorted_zones_gdf.index.tolist() == self.expected_sorted_ids
        assert (
            sorted_zones_gdf.index.name
            == self.settings.shared.geography.zones.activitysim_index_col
        )

        print("✅ STEP 1 PASSED: Canonical zone loading and sorting is correct.")

        # =====================================================================
        # STEP 2: BEAM Preprocessing - Preparing BEAM's Zone Input
        # =====================================================================
        # Now we verify that the BEAM preprocessor correctly uses the sorted
        # canonical zones to generate the zone shapefile that BEAM itself
        # will consume. This ensures the canonical order is passed to BEAM.

        # 2a. Initialize the BEAM preprocessor and run its zone preparation
        beam_preprocessor = BeamPreprocessor(
            "beam", self.state, self.provenance_tracker
        )
        beam_run_hash = self.provenance_tracker.start_model_run(
            "beam_preprocessor", year=2020
        )
        beam_zone_shapefile_path = beam_preprocessor.prepare_beam_zone_shapefile(
            self.workspace, beam_run_hash
        )

        # 2b. Assert that the shapefile was created
        assert os.path.exists(beam_zone_shapefile_path)

        # 2c. Read the generated shapefile and verify its order
        beam_input_gdf = gpd.read_file(beam_zone_shapefile_path)

        # 2d. Assert that the zones in the shapefile for BEAM are sorted
        # The preprocessor should have used the sorted data. The user's patch
        # correctly sets the ID field to the 'activitysim_index_col' ('TAZ').
        id_col = self.settings.shared.geography.zones.activitysim_index_col
        assert id_col in beam_input_gdf.columns
        assert beam_input_gdf[id_col].astype(str).tolist() == self.expected_sorted_ids

        print(
            "✅ STEP 2 PASSED: BEAM preprocessor correctly creates a sorted zone shapefile."
        )

        # =====================================================================
        # STEP 3: Skim Generation and Post-hoc Verification
        # =====================================================================
        # Here, we simulate the output of a BEAM run and then test the
        # `verify_skim_zone_order` function from the BEAM postprocessor. This
        # ensures that our safeguard correctly validates skim files.

        # 3a. Simulate BEAM's Zarr skim output with the CORRECT order
        mock_zarr_path = os.path.join(
            self.workspace.get_beam_output_dir(), "skims.zarr"
        )

        # Create a dummy xarray Dataset with 'otaz' as a coordinate
        ds = xr.Dataset(coords={"otaz": np.arange(len(self.expected_sorted_ids))})
        ds.attrs["original_zone_ids"] = self.expected_sorted_ids
        ds.to_zarr(mock_zarr_path, mode="w")

        # 3b. Verify the correctly generated skim file. This should pass.
        try:
            verify_skim_zone_order(self.settings, mock_zarr_path, self.workspace)
        except ValueError as e:
            pytest.fail(
                f"verify_skim_zone_order failed unexpectedly on a correct skim file: {e}"
            )

        print(
            "✅ STEP 3a PASSED: Skim verification correctly passed for an aligned skim file."
        )

        # 3c. Simulate a corrupted BEAM Zarr skim output with an INCORRECT order
        mock_bad_zarr_path = os.path.join(
            self.workspace.get_beam_output_dir(), "skims_bad.zarr"
        )

        # Create a dummy xarray Dataset with 'otaz' as a coordinate
        bad_ds = xr.Dataset(coords={"otaz": np.arange(len(self.expected_sorted_ids))})
        bad_ds.attrs["original_zone_ids"] = ["3", "1", "2"]  # Intentionally wrong order
        bad_ds.to_zarr(mock_bad_zarr_path, mode="w")

        # 3d. Assert that the verification function catches the mismatch and raises an error.
        with pytest.raises(
            ValueError, match="Skim zone order .* does not match canonical order"
        ):
            verify_skim_zone_order(self.settings, mock_bad_zarr_path, self.workspace)

        print(
            "✅ STEP 3b PASSED: Skim verification correctly failed for a misaligned skim file."
        )

        # =====================================================================
        # STEP 4: ActivitySim Preprocessing - Land Use Table Generation
        # =====================================================================
        # The final step is to verify that the ActivitySim preprocessor, when
        # generating its own input files from an UrbanSim source, also
        # respects the canonical zone order for the `land_use.csv` file.

        # 4a. Create minimal mock UrbanSim H5 data needed for the function to run.
        mock_usim_h5_path = os.path.join(
            self.workspace.get_usim_mutable_data_dir(),
            self.settings.urbansim.input_file_template.format(region_id=""),
        )
        os.makedirs(os.path.dirname(mock_usim_h5_path), exist_ok=True)

        # We need a 'blocks' table with IDs that can be mapped to our zones.
        # Let's create 3 blocks, one for each of our 3 zones.
        blocks_df = pd.DataFrame(
            {
                "square_meters_land": [10000, 10000, 10000],
                "x": [0.5, 1.5, 0.5],  # Add dummy x coordinates
                "y": [0.5, 0.5, 1.5],  # Add dummy y coordinates
            },
            index=pd.Index(["3", "1", "2"], name="block_id"),
        )

        # A 'households' table with a 'block_id' to link persons to a location.
        households_df = pd.DataFrame(
            {
                "block_id": [
                    "3",
                    "1",
                    "2",
                ],  # Updated to match block_id from previous fix
                "persons": [2, 3, 1],
                "income": [50000, 100000, 75000],
                "cars": [1, 2, 1],
                "workers": [1, 2, 1],  # Add dummy workers column
            },
            index=pd.Index([101, 102, 103], name="household_id"),
        )

        # A 'persons' table.
        persons_df = pd.DataFrame(
            {
                "household_id": [101, 101, 102, 102, 102, 103],
                "age": [30, 32, 45, 42, 12, 65],
                "worker": [1, 1, 1, 0, 0, 0],
                "student": [0, 0, 0, 0, 1, 0],
            },
            index=pd.Index(range(6), name="person_id"),
        )

        # A 'jobs' table.
        jobs_df = pd.DataFrame(
            {
                "block_id": ["block1", "block2", "block3"],
                "sector_id": ["A", "B", "A"],
            },
            index=pd.Index(range(3), name="job_id"),
        )

        with pd.HDFStore(mock_usim_h5_path, mode="w") as store:
            store.put("2020/households", households_df)
            store.put("2020/persons", persons_df)
            store.put("2020/jobs", jobs_df)
            store.put("2020/blocks", blocks_df)

        # 4b. Run the ActivitySim data creation process
        from pilates.activitysim.preprocessor import create_asim_data_from_h5

        create_asim_data_from_h5(
            self.settings, self.state, self.workspace, self.provenance_tracker
        )

        # 4c. Assert that the land_use.csv file was created
        land_use_path = os.path.join(
            self.workspace.get_asim_mutable_data_dir(), "land_use.csv"
        )
        assert os.path.exists(land_use_path), "land_use.csv was not created."

        # 4d. Read the land_use.csv and verify its index is sorted
        land_use_df = pd.read_csv(land_use_path, index_col="TAZ")
        assert land_use_df.index.astype(str).tolist() == self.expected_sorted_ids

        print("✅ STEP 4 PASSED: ActivitySim land_use.csv is correctly sorted.")
        print(
            "\n🎉 SUCCESS: End-to-end zone ID alignment workflow validated successfully."
        )
