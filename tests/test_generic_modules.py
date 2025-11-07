import pytest
from types import SimpleNamespace
from pilates.generic.model_factory import ModelFactory
from pilates.generic.runner import GenericRunner



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
    settings = SimpleNamespace(
        infrastructure=SimpleNamespace(
            container_manager="docker",
            docker_images={"urbansim": "my_image"}
        ),
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use="urbansim",
                travel=None,
                activity_demand=None,
                vehicle_ownership=None,
            )
        )
    )
    model, image = GenericRunner.get_model_and_image(settings, "land_use_model")
    assert model == "urbansim"
    assert image == "my_image"


def test_get_model_and_image_missing_manager():
    settings = SimpleNamespace(
        infrastructure=SimpleNamespace(container_manager=None),
        run=SimpleNamespace(models=SimpleNamespace(
            land_use="urbansim",
            travel=None,
            activity_demand=None,
            vehicle_ownership=None,
        ))  # Need run.models to exist
    )
    with pytest.raises(ValueError, match="Container Manager not specified"):
        GenericRunner.get_model_and_image(settings, "land_use_model")


def test_get_model_and_image_missing_model():
    settings = SimpleNamespace(
        infrastructure=SimpleNamespace(
            container_manager="docker",
            docker_images={"urbansim": "my_image"}
        ),
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use=None,
                travel=None,
                activity_demand=None,
                vehicle_ownership=None,
            )
        )
    )
    with pytest.raises(ValueError, match="No model land_use_model specified in settings."):
        GenericRunner.get_model_and_image(settings, "land_use_model")


def test_get_model_and_image_missing_image():
    settings = SimpleNamespace(
        infrastructure=SimpleNamespace(
            container_manager="docker",
            docker_images={}
        ),
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use="urbansim",
                travel=None,
                activity_demand=None,
                vehicle_ownership=None,
            )
        )
    )
    with pytest.raises(ValueError, match="No docker image specified for model 'urbansim'"):
        GenericRunner.get_model_and_image(settings, "land_use_model")
