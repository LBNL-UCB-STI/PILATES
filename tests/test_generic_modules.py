import pytest
from pilates.generic.model_factory import ModelFactory
from pilates.generic.runner import GenericRunner
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.postprocessor import GenericPostprocessor


# ----------------------------------------------------------------------
# ModelFactory tests
# ----------------------------------------------------------------------
def test_model_factory_returns_correct_classes():
    mf = ModelFactory()
    # Verify that known model keys return the proper classes
    runner = mf.get_runner("activitysim", None, None)
    assert runner.__class__.__name__ == "ActivitysimRunner"

    pre = mf.get_preprocessor("beam", None, None)
    assert pre.__class__.__name__ == "BeamPreprocessor"

    post = mf.get_postprocessor("urbansim", None, None)
    assert post.__class__.__name__ == "UrbansimPostprocessor"


def test_model_factory_unknown_model_raises():
    mf = ModelFactory()
    with pytest.raises(KeyError):
        mf.get_runner("nonexistent_model", None, None)


# ----------------------------------------------------------------------
# GenericRunner.get_model_and_image tests
# ----------------------------------------------------------------------
def test_get_model_and_image_success():
    settings = {
        "container_manager": "docker",
        "docker_images": {"my_model": "my_image"},
        "my_model": "my_model",
    }
    model, image = GenericRunner.get_model_and_image(settings, "my_model")
    assert model == "my_model"
    assert image == "my_image"


def test_get_model_and_image_missing_manager():
    settings = {"my_model": "my_model"}
    with pytest.raises(ValueError, match="Container Manager not specified"):
        GenericRunner.get_model_and_image(settings, "my_model")


def test_get_model_and_image_missing_model():
    settings = {
        "container_manager": "docker",
        "docker_images": {"my_model": "my_image"},
    }
    with pytest.raises(ValueError, match="No model my_model specified"):
        GenericRunner.get_model_and_image(settings, "my_model")


def test_get_model_and_image_missing_image():
    settings = {
        "container_manager": "docker",
        "docker_images": {},
        "my_model": "my_model",
    }
    with pytest.raises(ValueError, match="No docker image specified"):
        GenericRunner.get_model_and_image(settings, "my_model")


# ----------------------------------------------------------------------
# GenericPreprocessor & GenericPostprocessor abstract class tests
# ----------------------------------------------------------------------
# NOTE: These tests were removed because they attempted to test abstract classes
# directly without implementing required abstract methods. The actual preprocessor
# and postprocessor implementations are tested via integration tests and
# model-specific test suites.
