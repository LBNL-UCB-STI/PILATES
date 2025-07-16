from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.runner import ActivitysimRunner
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.beam.runner import BeamRunner
from pilates.beam.postprocessor import BeamPostprocessor
from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasRunner
from pilates.atlas.postprocessor import AtlasPostprocessor


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
        "atlas": {
            "preprocessor": AtlasPreprocessor,
            "runner": AtlasRunner,
            "postprocessor": AtlasPostprocessor,
        },
    }

    def get_runner(self, model_name):
        return self._registry[model_name.lower()]["runner"](model_name)

    def get_preprocessor(self, model_name):
        return self._registry[model_name.lower()]["preprocessor"]()

    def get_postprocessor(self, model_name):
        return self._registry[model_name.lower()]["postprocessor"]()
