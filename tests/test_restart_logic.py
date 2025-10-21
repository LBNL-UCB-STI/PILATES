import pytest
import os
import shutil
import yaml
from unittest.mock import MagicMock, patch
from datetime import datetime

from pilates.generic.model_factory import ModelFactory
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.runner import GenericRunner
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import RecordStore, ModelRunInfo, FileRecord, PilatesRunInfo
from pilates.utils.provenance import FileProvenanceTracker, OpenLineageTracker
from workflow_state import WorkflowState
from pilates.workspace import Workspace

# Helper function to create a dummy run_info.json
def create_dummy_run_info(tmp_path, model_name, year, iteration, sub_stage_progress, output_records=None):
    run_id = "test_run_id"
    output_path = str(tmp_path)
    folder_name = "test_run"
    
    # Ensure the run folder exists
    run_folder = tmp_path / folder_name
    run_folder.mkdir(exist_ok=True)

    tracker = FileProvenanceTracker(run_id, output_path, folder_name)
    
    # Manually set up run_info for testing
    tracker.run_info = PilatesRunInfo(
        run_id=run_id,
        created_at=datetime.now().isoformat(),
        file_records={},
        model_runs={}
    )

    # Simulate a completed preprocessor run
    if sub_stage_progress in ["runner", "postprocessor"]:
        preprocessor_model_name = f"{model_name}_preprocessor"
        preprocessor_run_id = f"{preprocessor_model_name}_{datetime.now().strftime('%Y%m%d')}_abcde123"
        preprocessor_output_file_hash = "preprocessor_output_hash"
        
        # Create a dummy FileRecord for preprocessor output
        preprocessor_output_record = FileRecord(
            unique_id=preprocessor_output_file_hash,
            file_path="dummy/preprocessor_output.txt",
            short_name="preprocessor_output",
            description="Dummy preprocessor output",
            producing_run_id=preprocessor_run_id
        )
        tracker.run_info.file_records[preprocessor_output_file_hash] = preprocessor_output_record

        # Create a dummy ModelRunInfo for preprocessor
        preprocessor_run_info = ModelRunInfo(
            unique_id=preprocessor_run_id,
            model=preprocessor_model_name,
            year=year,
            iteration=iteration,
            status="completed",
            completed_at=datetime.now().isoformat(), # new
            output_record_hashes=[preprocessor_output_file_hash]
        )
        tracker.run_info.model_runs[preprocessor_run_id] = preprocessor_run_info

    # Simulate a completed runner run
    if sub_stage_progress == "postprocessor":
        runner_model_name = model_name
        runner_run_id = f"{runner_model_name}_{datetime.now().strftime('%Y%m%d')}_fghij456"
        runner_output_file_hash = "runner_output_hash"

        # Create a dummy FileRecord for runner output
        runner_output_record = FileRecord(
            unique_id=runner_output_file_hash,
            file_path="dummy/runner_output.txt",
            short_name="runner_output",
            description="Dummy runner output",
            producing_run_id=runner_run_id
        )
        tracker.run_info.file_records[runner_output_file_hash] = runner_output_record

        # Create a dummy ModelRunInfo for runner
        runner_run_info = ModelRunInfo(
            unique_id=runner_run_id,
            model=runner_model_name,
            year=year,
            iteration=iteration,
            status="completed",
            completed_at=datetime.now().isoformat(), # new
            output_record_hashes=[runner_output_file_hash]
        )
        tracker.run_info.model_runs[runner_run_id] = runner_run_info
    
    tracker._save_run_info() # Persist the run_info.json

    return tracker

@pytest.fixture
def setup_test_environment(tmp_path):
    # Create dummy settings.yaml
    settings_content = {
        "start_year": 2017,
        "end_year": 2017,
        "output_directory": str(tmp_path),
        "run_name": "test_run",
        "land_use_model": "urbansim",
        "vehicle_ownership_model": "atlas",
        "activity_demand_model": "activitysim",
        "travel_model": "beam",
        "docker_images": {
            "urbansim": "test_urbansim_image",
            "activitysim": "test_activitysim_image",
            "beam": "test_beam_image",
            "atlas": "test_atlas_image",
        }
    }
    settings_path = tmp_path / "settings.yaml"
    with open(settings_path, "w") as f:
        yaml.dump(settings_content, f)

    # Mock WorkflowState.from_settings to use our dummy settings
    with patch('workflow_state.WorkflowState.from_settings') as mock_from_settings:
        mock_state = MagicMock(spec=WorkflowState)
        mock_state.start_year = 2017
        mock_state.end_year = 2017
        mock_state.current_year = 2017
        mock_state.current_inner_iter = 0
        mock_state.current_major_stage = WorkflowState.Stage.land_use
        mock_state.sub_stage_progress = None # Default to fresh start
        mock_state.file_loc = tmp_path / "current_stage.yaml"
        mock_state.full_settings = settings_content
        mock_state.set_sub_stage_progress.side_effect = lambda x: setattr(mock_state, 'sub_stage_progress', x)
        mock_from_settings.return_value = mock_state
        
        yield mock_state, settings_path, tmp_path

def test_fresh_run_executes_all_sub_stages(setup_test_environment):
    """
    Test that a fresh run (no previous state) executes all preprocessor, runner, and postprocessor steps.
    """
    mock_state, settings_path, tmp_path = setup_test_environment
    mock_state.sub_stage_progress = None # Ensure fresh start

    # Mock the actual _preprocess, _run, _postprocess methods
    with patch('pilates.urbansim.preprocessor.UrbansimPreprocessor._preprocess') as mock_pre_preprocess:
        with patch('pilates.urbansim.runner.UrbansimRunner._run') as mock_run_run:
            with patch('pilates.urbansim.postprocessor.UrbansimPostprocessor._postprocess') as mock_post_postprocess:
                # Instantiate the ModelFactory
                factory = ModelFactory()

                # Get actual instances of the preprocessor, runner, postprocessor
                # These will have their _preprocess, _run, _postprocess methods mocked
                preprocessor_instance = factory.get_preprocessor("urbansim", mock_state, MagicMock(), major_stage=WorkflowState.Stage.land_use)
                runner_instance = factory.get_runner("urbansim", mock_state, MagicMock(), major_stage=WorkflowState.Stage.land_use)
                postprocessor_instance = factory.get_postprocessor("urbansim", mock_state, MagicMock(), major_stage=WorkflowState.Stage.land_use)

                # Set up return values for the mocked _preprocess, _run, _postprocess
                mock_pre_preprocess.return_value = RecordStore()
                mock_run_run.return_value = (RecordStore(), ModelRunInfo(model="urbansim", year=2017, unique_id="run_id"))
                mock_post_postprocess.return_value = RecordStore()

                # Call the actual preprocess, run, postprocess methods
                input_data = preprocessor_instance.preprocess(MagicMock())
                raw_outputs, run_info = runner_instance.run(input_data, MagicMock())
                postprocessor_instance.postprocess(raw_outputs, MagicMock(), run_info)

                # Assert that all _preprocess, _run, _postprocess were called
                mock_pre_preprocess.assert_called_once()
                mock_run_run.assert_called_once()
                mock_post_postprocess.assert_called_once()

                # Assert that sub_stage_progress was updated correctly
                assert mock_state.sub_stage_progress == "postprocessor" # Last one to be set

def test_resume_from_runner_failure_skips_preprocessor(setup_test_environment):
    """
    Test that when resuming from a runner failure, the preprocessor is skipped and its outputs are loaded from provenance.
    """
    mock_state, settings_path, tmp_path = setup_test_environment
    mock_state.current_major_stage = WorkflowState.Stage.land_use
    mock_state.sub_stage_progress = "runner" # Simulate failure after preprocessor

    # Create dummy run_info.json with a completed preprocessor run
    tracker = create_dummy_run_info(tmp_path, "urbansim", mock_state.current_year, mock_state.current_inner_iter, "runner")
    mock_state.run_info_path = tracker.run_info_path

    with patch('pilates.urbansim.preprocessor.UrbansimPreprocessor._preprocess') as mock_pre_preprocess:
        with patch('pilates.urbansim.runner.UrbansimRunner._run') as mock_run_run:
            with patch('pilates.urbansim.postprocessor.UrbansimPostprocessor._postprocess') as mock_post_postprocess:
                with patch('pilates.utils.provenance.FileProvenanceTracker._initialize_run_info', return_value=tracker.run_info):
                    # Instantiate the ModelFactory
                    factory = ModelFactory()

                    # Get actual instances of the preprocessor, runner, postprocessor
                    # These will have their _preprocess, _run, _postprocess methods mocked
                    preprocessor_instance = factory.get_preprocessor("urbansim", mock_state, tracker, major_stage=WorkflowState.Stage.land_use)
                    runner_instance = factory.get_runner("urbansim", mock_state, tracker, major_stage=WorkflowState.Stage.land_use)
                    postprocessor_instance = factory.get_postprocessor("urbansim", mock_state, tracker, major_stage=WorkflowState.Stage.land_use)

                    # Set up return values for the mocked _run, _postprocess
                    mock_run_run.return_value = (RecordStore(), ModelRunInfo(model="urbansim", year=2017, unique_id="run_id"))
                    mock_post_postprocess.return_value = RecordStore()

                    # Call the actual preprocess, run, postprocess methods
                    input_data = preprocessor_instance.preprocess(MagicMock())
                    raw_outputs, run_info = runner_instance.run(input_data, MagicMock())
                    postprocessor_instance.postprocess(raw_outputs, MagicMock(), run_info)

                    # Assert that _preprocess was NOT called
                    mock_pre_preprocess.assert_not_called()
                    # Assert that _run and _postprocess were called
                    mock_run_run.assert_called_once()
                    mock_post_postprocess.assert_called_once()

                    # Assert that the input_data for runner was loaded from provenance
                    assert isinstance(input_data, RecordStore)
                    assert len(input_data.all_records()) > 0
                    assert input_data.all_records()[0].unique_id == "preprocessor_output_hash"

                    # Assert that sub_stage_progress was updated correctly
                    assert mock_state.sub_stage_progress == "postprocessor" # Last one to be set

def test_resume_from_postprocessor_failure_skips_preprocessor_and_runner(setup_test_environment):
    """
    Test that when resuming from a postprocessor failure, the preprocessor and runner are skipped,
    and their outputs are loaded from provenance.
    """
    mock_state, settings_path, tmp_path = setup_test_environment
    mock_state.current_major_stage = WorkflowState.Stage.land_use
    mock_state.sub_stage_progress = "postprocessor" # Simulate failure after runner

    # Create dummy run_info.json with completed preprocessor and runner runs
    tracker = create_dummy_run_info(tmp_path, "urbansim", mock_state.current_year, mock_state.current_inner_iter, "postprocessor")
    mock_state.run_info_path = tracker.run_info_path

    with patch('pilates.urbansim.preprocessor.UrbansimPreprocessor._preprocess') as mock_pre_preprocess:
        with patch('pilates.urbansim.runner.UrbansimRunner._run') as mock_run_run:
            with patch('pilates.urbansim.postprocessor.UrbansimPostprocessor._postprocess') as mock_post_postprocess:
                with patch('pilates.utils.provenance.FileProvenanceTracker._initialize_run_info', return_value=tracker.run_info):
                    # Instantiate the ModelFactory
                    factory = ModelFactory()

                    # Get actual instances of the preprocessor, runner, postprocessor
                    # These will have their _preprocess, _run, _postprocess methods mocked
                    preprocessor_instance = factory.get_preprocessor("urbansim", mock_state, tracker, major_stage=WorkflowState.Stage.land_use)
                    runner_instance = factory.get_runner("urbansim", mock_state, tracker, major_stage=WorkflowState.Stage.land_use)
                    postprocessor_instance = factory.get_postprocessor("urbansim", mock_state, tracker, major_stage=WorkflowState.Stage.land_use)

                    # Set up return values for the mocked _postprocess
                    mock_post_postprocess.return_value = RecordStore()

                    # Call the actual preprocess, run, postprocess methods
                    input_data = preprocessor_instance.preprocess(MagicMock())
                    raw_outputs, run_info = runner_instance.run(input_data, MagicMock())
                    postprocessor_instance.postprocess(raw_outputs, MagicMock(), run_info)

                    # Assert that _preprocess and _run were NOT called
                    mock_pre_preprocess.assert_not_called()
                    mock_run_run.assert_not_called()
                    # Assert that _postprocess was called
                    mock_post_postprocess.assert_called_once()

                    # Assert that the input_data for runner and raw_outputs for postprocessor were loaded from provenance
                    assert isinstance(input_data, RecordStore)
                    assert len(input_data.all_records()) > 0
                    assert input_data.all_records()[0].unique_id == "preprocessor_output_hash"

                    assert isinstance(raw_outputs, RecordStore)
                    assert len(raw_outputs.all_records()) > 0
                    assert raw_outputs.all_records()[0].unique_id == "runner_output_hash"
                    assert isinstance(run_info, ModelRunInfo)
                    assert run_info.model == "urbansim"

                    # Assert that sub_stage_progress was updated correctly
                    assert mock_state.sub_stage_progress == "postprocessor" # Last one to be set