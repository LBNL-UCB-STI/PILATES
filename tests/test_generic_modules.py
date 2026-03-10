import pytest
from types import SimpleNamespace
from pilates.generic.model_factory import ModelFactory
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore
from pilates.generic.runner import GenericRunner
from pilates.workflows import step_exec


# ----------------------------------------------------------------------
# ModelFactory tests
# ----------------------------------------------------------------------
def test_model_factory_returns_correct_classes():
    mf = ModelFactory()
    # Verify that known model keys return the proper classes
    runner = mf.get_runner("activitysim", None)
    assert runner.__class__.__name__ == "ActivitysimRunner"

    mock_state = SimpleNamespace(full_settings=SimpleNamespace())
    pre = mf.get_preprocessor("beam", mock_state)
    assert pre.__class__.__name__ == "BeamPreprocessor"

    post = mf.get_postprocessor("urbansim", None)
    assert post.__class__.__name__ == "UrbansimPostprocessor"


def test_model_factory_unknown_model_raises():
    mf = ModelFactory()
    with pytest.raises(KeyError):
        mf.get_runner("nonexistent_model", None)


# ----------------------------------------------------------------------
# GenericRunner.get_model_and_image tests
# ----------------------------------------------------------------------
def test_get_model_and_image_success():
    settings = SimpleNamespace(
        infrastructure=SimpleNamespace(
            container_manager="docker", docker_images={"urbansim": "my_image"}
        ),
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use="urbansim",
                travel=None,
                activity_demand=None,
                vehicle_ownership=None,
            )
        ),
    )
    model, image = GenericRunner.get_model_and_image(settings, "land_use_model")
    assert model == "urbansim"
    assert image == "my_image"


def test_get_model_and_image_missing_manager():
    settings = SimpleNamespace(
        infrastructure=SimpleNamespace(container_manager=None),
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use="urbansim",
                travel=None,
                activity_demand=None,
                vehicle_ownership=None,
            )
        ),  # Need run.models to exist
    )
    with pytest.raises(ValueError, match="Container Manager not specified"):
        GenericRunner.get_model_and_image(settings, "land_use_model")


def test_get_model_and_image_missing_model():
    settings = SimpleNamespace(
        infrastructure=SimpleNamespace(
            container_manager="docker", docker_images={"urbansim": "my_image"}
        ),
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use=None,
                travel=None,
                activity_demand=None,
                vehicle_ownership=None,
            )
        ),
    )
    with pytest.raises(
        ValueError, match="No model land_use_model specified in settings."
    ):
        GenericRunner.get_model_and_image(settings, "land_use_model")


def test_get_model_and_image_missing_image():
    settings = SimpleNamespace(
        infrastructure=SimpleNamespace(container_manager="docker", docker_images={}),
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use="urbansim",
                travel=None,
                activity_demand=None,
                vehicle_ownership=None,
            )
        ),
    )
    with pytest.raises(
        ValueError, match="No docker image specified for model 'urbansim'"
    ):
        GenericRunner.get_model_and_image(settings, "land_use_model")


def test_preprocessor_protocol_shape_forwards_previous_records():
    workspace = object()
    previous_records = RecordStore()
    captured = {}

    class _Preprocessor:
        def preprocess(self, workspace_arg, previous_records_arg):
            captured["workspace"] = workspace_arg
            captured["previous_records"] = previous_records_arg
            return previous_records_arg

    result = _Preprocessor().preprocess(workspace, previous_records)

    assert captured == {
        "workspace": workspace,
        "previous_records": previous_records,
    }
    assert result is previous_records


def test_postprocessor_protocol_shape_forwards_model_run_hash():
    raw_outputs = RecordStore()
    workspace = object()
    captured = {}

    class _Postprocessor:
        def postprocess(self, raw_outputs_arg, workspace_arg, model_run_hash):
            captured["raw_outputs"] = raw_outputs_arg
            captured["workspace"] = workspace_arg
            captured["model_run_hash"] = model_run_hash
            return raw_outputs_arg

    result = _Postprocessor().postprocess(
        raw_outputs,
        workspace,
        model_run_hash="run-hash-123",
    )

    assert captured == {
        "raw_outputs": raw_outputs,
        "workspace": workspace,
        "model_run_hash": "run-hash-123",
    }
    assert result is raw_outputs


def test_warm_start_activities_forwards_preprocess_outputs_to_runner_and_updates_usim_inputs(
    monkeypatch,
):
    settings = SimpleNamespace()
    state = SimpleNamespace()
    workspace = object()
    preprocess_outputs = RecordStore()
    captured = {}

    class _Preprocessor:
        def preprocess(self, workspace_arg):
            captured["preprocess_workspace"] = workspace_arg
            return preprocess_outputs

    class _Runner:
        def run(self, input_data_arg, workspace_arg):
            captured["runner_input_data"] = input_data_arg
            captured["runner_workspace"] = workspace_arg
            return RecordStore()

    def _capture_update(settings_arg, state_arg, workspace_arg, model_run_hash=None):
        captured["update_call"] = {
            "settings": settings_arg,
            "state": state_arg,
            "workspace": workspace_arg,
            "model_run_hash": model_run_hash,
        }

    monkeypatch.setattr(
        step_exec.GenericRunner,
        "get_model_and_image",
        staticmethod(lambda settings, model_type: ("activitysim", "fake-image")),
    )
    monkeypatch.setattr(
        step_exec.ModelFactory,
        "get_preprocessor",
        lambda self, model_name, state: _Preprocessor(),
    )
    monkeypatch.setattr(
        step_exec.ModelFactory,
        "get_runner",
        lambda self, model_name, state: _Runner(),
    )
    monkeypatch.setattr(
        step_exec.asim_post,
        "update_usim_inputs_after_warm_start",
        _capture_update,
    )

    step_exec.warm_start_activities(settings, state, workspace)

    assert captured["preprocess_workspace"] is workspace
    assert captured["runner_input_data"] is preprocess_outputs
    assert captured["runner_workspace"] is workspace
    assert captured["update_call"] == {
        "settings": settings,
        "state": state,
        "workspace": workspace,
        "model_run_hash": None,
    }


class _StageTrackingState:
    def __init__(self):
        self.sub_stages = []

    def set_sub_stage_progress(self, progress):
        self.sub_stages.append(progress)


def test_generic_preprocessor_sets_stage_and_forwards_record_store():
    state = _StageTrackingState()
    workspace = object()
    previous_records = RecordStore()
    captured = {}

    class _GenericPreprocessor(GenericPreprocessor):
        def copy_data_to_mutable_location(self, settings, output_dir):
            return RecordStore(), RecordStore()

        def _preprocess(self, workspace_arg, previous_records_arg):
            captured["workspace"] = workspace_arg
            captured["previous_records"] = previous_records_arg
            return previous_records_arg

    preprocessor = _GenericPreprocessor("dummy", state)
    result = preprocessor.preprocess(workspace, previous_records)

    assert state.sub_stages == ["preprocessor"]
    assert captured == {
        "workspace": workspace,
        "previous_records": previous_records,
    }
    assert result is previous_records


def test_generic_preprocessor_defaults_previous_records_to_new_store():
    state = _StageTrackingState()
    workspace = object()
    captured = {}

    class _GenericPreprocessor(GenericPreprocessor):
        def copy_data_to_mutable_location(self, settings, output_dir):
            return RecordStore(), RecordStore()

        def _preprocess(self, workspace_arg, previous_records_arg):
            captured["workspace"] = workspace_arg
            captured["previous_records"] = previous_records_arg
            return previous_records_arg

    preprocessor = _GenericPreprocessor("dummy", state)
    first = preprocessor.preprocess(workspace)
    second = preprocessor.preprocess(workspace)

    assert state.sub_stages == ["preprocessor", "preprocessor"]
    assert captured["workspace"] is workspace
    assert isinstance(first, RecordStore)
    assert isinstance(second, RecordStore)
    assert first is not second


def test_generic_postprocessor_sets_stage_and_forwards_model_run_hash():
    state = _StageTrackingState()
    raw_outputs = RecordStore()
    workspace = object()
    captured = {}

    class _GenericPostprocessor(GenericPostprocessor):
        def _postprocess(self, raw_outputs_arg, workspace_arg, model_run_hash=None):
            captured["raw_outputs"] = raw_outputs_arg
            captured["workspace"] = workspace_arg
            captured["model_run_hash"] = model_run_hash
            return raw_outputs_arg

    postprocessor = _GenericPostprocessor("dummy", state)
    result = postprocessor.postprocess(
        raw_outputs,
        workspace,
        model_run_hash="run-hash-456",
    )

    assert state.sub_stages == ["postprocessor"]
    assert captured == {
        "raw_outputs": raw_outputs,
        "workspace": workspace,
        "model_run_hash": "run-hash-456",
    }
    assert result is raw_outputs
