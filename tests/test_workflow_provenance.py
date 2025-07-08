import os
import tempfile
import shutil
import uuid
import pytest

from pilates.generic.model_factory import ModelFactory
from pilates.utils.provenance import OpenLineageTracker
import json
from pilates.workspace import Workspace
from workflow_state import WorkflowState

def minimal_settings(tmpdir):
    # Minimal settings for Activitysim/BEAM workflow
    return {
        "region": "seattle",
        "start_year": 2020,
        "end_year": 2020,
        "travel_model_freq": 1,
        "land_use_enabled": False,
        "vehicle_ownership_model_enabled": False,
        "activity_demand_enabled": True,
        "traffic_assignment_enabled": True,
        "replanning_enabled": False,
        "state_file_loc": os.path.join(tmpdir, "run_state.yaml"),
        "output_directory": tmpdir,
        "run_name": "test-run",
        "container_manager": "singularity",
        "singularity_images": {
            "activitysim": "dummy_image",
            "beam": "dummy_image"
        },
        "docker_images": {},
        "asim_local_mutable_data_folder": "activitysim/data/",
        "asim_local_output_folder": "activitysim/output/",
        "asim_local_configs_folder": "pilates/activitysim/configs/",
        "asim_local_mutable_configs_folder": "activitysim/configs/",
        "beam_local_mutable_data_folder": "beam/input/",
        "beam_local_output_folder": "beam/beam_output/",
        "beam_local_input_folder": "beam/production/",
        "beam_geoms_fname": "dummy_geoms.csv",
        "beam_router_directory": "r5/",
        "beam_config": "dummy.conf",
        "skims_fname": "dummy_skims.omx",
        "origin_skims_fname": "dummy_origin_skims.csv.gz",
        "periods": ["AM"],
        "transit_paths": {},
        "hwy_paths": ["SOV"],
        "ridehail_path_map": {},
        "beam_asim_hwy_measure_map": {"TIME": "TIME_minutes", "DIST": "DIST_miles", "BTOLL": None, "VTOLL": "VTOLL_FAR"},
        "beam_asim_transit_measure_map": {},
        "beam_asim_ridehail_measure_map": {},
        "asim_output_tables": {"prefix": "final_", "tables": ["households", "persons"]},
        "region_to_region_id": {"seattle": "00000000"},
        "usim_local_data_input_folder": "pilates/urbansim/data/",
        "usim_local_mutable_data_folder": "urbansim/data/",
        "usim_client_data_folder": "/base/demos_urbansim/data/",
        "usim_formattable_input_file_name": "dummy.h5",
        "usim_formattable_output_file_name": "dummy_out.h5",
        "asim_validation_folder": "pilates/activitysim/validation",
        "FIPS": {"testregion": {"state": "00", "counties": ["001"]}},
        "local_crs": {"testregion": "EPSG:4326"},
        "geoms_index_col": "zone_id",
        "activity_demand_model": "activitysim",
        "travel_model": "beam",
    }

def create_dummy_file(path, content="dummy"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

@pytest.mark.parametrize("model_name", ["activitysim", "beam"])
def test_workflow_and_provenance_tracking(tmp_path, model_name):
    # Setup minimal settings and workspace
    tmpdir = str(tmp_path)
    settings = minimal_settings(tmpdir)
    original_cwd = os.getcwd()
    os.chdir(tmpdir)

    try:
        # Create dummy source file that the preprocessor expects to copy
        # This path is relative to the new CWD (tmpdir)
        dummy_geoms_source_path = os.path.join(
            settings["beam_local_input_folder"],
            settings["region"],
            "r5",
            settings["beam_geoms_fname"],
        )
        create_dummy_file(dummy_geoms_source_path)

        run_id = str(uuid.uuid4())
        provenance_tracker = OpenLineageTracker(
            run_id, settings["output_directory"], folder_name="test-run"
        )
        provenance_tracker.initialize_from_settings(settings)
        workspace = Workspace(
            settings,
            settings["output_directory"],
            folder_name="test-run",
            provenance_tracker=provenance_tracker,
        )
        state = WorkflowState.from_settings(settings)

        # Create dummy input files for preprocessors
        if model_name == "activitysim":
            dummy_input = os.path.join(
                workspace.get_asim_mutable_data_dir(), "dummy_input.csv"
            )
        else:
            dummy_input = os.path.join(
                workspace.get_beam_mutable_data_dir(), "dummy_input.csv"
            )
        create_dummy_file(dummy_input)

        # Get model factory and components
        factory = ModelFactory()
        preprocessor = factory.get_preprocessor(model_name)
        runner = factory.get_runner(model_name)
        postprocessor = factory.get_postprocessor(model_name)

        # Preprocess step
        pre_run_hash = provenance_tracker.start_model_run(
            f"{model_name}_preprocessor",
            state.current_year,
            state.current_inner_iter,
            description=f"Preprocessing for {model_name}",
        )
        input_data = preprocessor.preprocess(
            state, workspace, provenance_tracker, pre_run_hash
        )
        provenance_tracker.complete_model_run(pre_run_hash)

        # Simulate runner step (will not actually run model, but should not error)
        run_outputs, run_info = runner.run(
            input_data, state, workspace, provenance_tracker
        )

        # Postprocess step
        post_run_hash = provenance_tracker.start_model_run(
            f"{model_name}_postprocessor",
            state.current_year,
            state.current_inner_iter,
            description=f"Post-processing {model_name} outputs",
        )
        processed_outputs = postprocessor.postprocess(
            run_outputs, run_info, state, workspace, provenance_tracker, post_run_hash
        )
        provenance_tracker.complete_model_run(post_run_hash)

        # Check that provenance records exist for the run
        run_info_data = provenance_tracker.get_run_info()
        assert "model_runs" in run_info_data
        assert any(
            run["model"].startswith(model_name)
            for run in run_info_data["model_runs"].values()
        )
        assert "file_records" in run_info_data
        # There should be at least one file record (from dummy input or output)
        assert len(run_info_data["file_records"]) > 0

        # Check that the openlineage.jsonl file was created and is valid
        log_file_path = os.path.join(settings["output_directory"], "test-run", "openlineage.jsonl")
        assert os.path.exists(log_file_path)
        with open(log_file_path, "r") as f:
            lines = f.readlines()
            # Should be one START and one COMPLETE event for each of the 3 steps
            assert len(lines) == 6
            for line in lines:
                event = json.loads(line)
                assert "eventType" in event
                assert "run" in event
                assert "job" in event
                if event["eventType"] == "START":
                    assert "inputs" in event
                else:
                    assert "outputs" in event

    finally:
        os.chdir(original_cwd)
        # Clean up
        shutil.rmtree(tmpdir, ignore_errors=True)
