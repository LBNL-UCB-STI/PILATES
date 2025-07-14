from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.runner import ActivitysimRunner
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.beam.runner import BeamRunner
from pilates.beam.postprocessor import BeamPostprocessor


class ModelFactory:
    _registry = {
        "activitysim": {
            "preprocessor": ActivitysimPreprocessor,
            "runner": ActivitysimRunner,
            "postprocessor": ActivitysimPostprocessor,
        },
        "beam": {
            "preprocessor": BeamPreprocessor,
            "runner": BeamRunner,
            "postprocessor": BeamPostprocessor,
        },
    }

    def get_runner(self, model_name):
        return self._registry[model_name.lower()]["runner"](model_name)

    def get_preprocessor(self, model_name):
        return self._registry[model_name.lower()]["preprocessor"]()

    def get_postprocessor(self, model_name):
        return self._registry[model_name.lower()]["postprocessor"]()
