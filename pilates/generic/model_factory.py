from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.runner import ActivitysimRunner, ActivitysimCompileRunner
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.beam.runner import BeamRunner
from pilates.beam.postprocessor import BeamPostprocessor
from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasRunner
from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.urbansim.postprocessor import UrbansimPostprocessor
from pilates.urbansim.preprocessor import UrbansimPreprocessor
from pilates.urbansim.runner import UrbansimRunner


class ModelFactory:
    _registry = {
        "activitysim": {
            "preprocessor": ActivitysimPreprocessor,
            "runner": ActivitysimRunner,
            "postprocessor": ActivitysimPostprocessor,
        },
        "activitysim_compile": {
            "preprocessor": ActivitysimPreprocessor,
            "runner": ActivitysimCompileRunner,
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
        "urbansim": {
            "preprocessor": UrbansimPreprocessor,
            "runner": UrbansimRunner,
            "postprocessor": UrbansimPostprocessor,
        },
    }

    def get_runner(
        self,
        model_name,
        state: "WorkflowState" = None,
        major_stage: "WorkflowState.Stage" = None,
    ):
        return self._registry[model_name.lower()]["runner"](
            model_name, state, major_stage
        )

    def get_preprocessor(
        self,
        model_name,
        state: "WorkflowState" = None,
        major_stage: "WorkflowState.Stage" = None,
    ):
        return self._registry[model_name.lower()]["preprocessor"](
            model_name, state, major_stage
        )

    def get_postprocessor(
        self,
        model_name,
        state: "WorkflowState" = None,
        major_stage: "WorkflowState.Stage" = None,
    ):
        return self._registry[model_name.lower()]["postprocessor"](
            model_name, state, major_stage
        )
