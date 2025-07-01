from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.runner import ActivitysimRunner
from pilates.activitysim.postprocessor import ActivitysimPostprocessor


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

    def get_runner(self, model_name, provenanceTracker=None):
        """
        Return a runner instance for the given model name, passing provenanceTracker.
        """
        if model_name.lower() == "activitysim":
            return ActivitysimRunner("activitysim", provenanceTracker)
        return None

    def get_preprocessor(self, model_name, provenanceTracker=None):
        """
        Return a preprocessor instance for the given model name, passing provenanceTracker.
        """
        if model_name.lower() == "activitysim":
            return ActivitysimPreprocessor(provenanceTracker)
        return None

    def get_postprocessor(self, model_name, provenanceTracker=None):
        """
        Return a postprocessor instance for the given model name, passing provenanceTracker.
        """
        if model_name.lower() == "activitysim":
            return  ActivitysimPostprocessor(provenanceTracker)
        return None
