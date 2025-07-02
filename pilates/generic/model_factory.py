from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.runner import ActivitysimRunner
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.beam.runner import BeamRunner
from pilates.beam.postprocessor import BeamPostprocessor


class ModelFactory:
    """
    Factory class for creating model runner, preprocessor, and postprocessor instances
    based on model name or type.

    This is a token implementation. In a real implementation, this would
    dynamically import or look up the correct class for each model type.
    """

    def __init__(self):
        # In a real implementation, this might register available models
        self._registry = {}

    def get_runner(self, model_name):
        """
        Return a runner instance for the given model name.
        """
        if model_name.lower() == "activitysim":
            return ActivitysimRunner("activitysim")
        if model_name.lower() == "beam":
            return BeamRunner("beam")
        return None

    def get_preprocessor(self, model_name):
        """
        Return a preprocessor instance for the given model name.
        """
        if model_name.lower() == "activitysim":
            return ActivitysimPreprocessor()
        if model_name.lower() == "beam":
            return BeamPreprocessor()
        return None

    def get_postprocessor(self, model_name):
        """
        Return a postprocessor instance for the given model name.
        """
        if model_name.lower() == "activitysim":
            return ActivitysimPostprocessor()
        if model_name.lower() == "beam":
            return BeamPostprocessor()
        return None
