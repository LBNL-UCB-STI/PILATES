from typing import TYPE_CHECKING, Tuple

from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.runner import ActivitysimRunner, ActivitysimCompileRunner
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.beam.runner import BeamRunner, BeamFullSkimRunner
from pilates.beam.postprocessor import BeamPostprocessor
from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasRunner
from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.urbansim.postprocessor import UrbansimPostprocessor
from pilates.urbansim.preprocessor import UrbansimPreprocessor
from pilates.urbansim.runner import UrbansimRunner

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.generic.postprocessor import GenericPostprocessor
    from pilates.generic.preprocessor import GenericPreprocessor
    from pilates.generic.runner import GenericRunner


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
        "beam_full_skim": {
            "preprocessor": BeamPreprocessor,
            "runner": BeamFullSkimRunner,
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
    ) -> "GenericRunner":
        return self._registry[model_name.lower()]["runner"](model_name, state)

    def get_preprocessor(
        self,
        model_name,
        state: "WorkflowState" = None,
    ) -> "GenericPreprocessor":
        return self._registry[model_name.lower()]["preprocessor"](model_name, state)

    def get_postprocessor(
        self,
        model_name,
        state: "WorkflowState" = None,
    ) -> "GenericPostprocessor":
        return self._registry[model_name.lower()]["postprocessor"](model_name, state)

    def get_components(
        self,
        model_name: str,
        state: "WorkflowState" = None,
    ) -> Tuple["GenericPreprocessor", "GenericRunner", "GenericPostprocessor"]:
        """
        Return preprocessor, runner, and postprocessor instances for a model.

        Parameters
        ----------
        model_name : str
            Model key registered in the factory (e.g., "urbansim").
        state : WorkflowState, optional
            Workflow state to pass through to component constructors.
        Returns
        -------
        tuple
            (preprocessor, runner, postprocessor) instances for the model.

        Notes
        -----
        This mirrors the standard PILATES component pattern in one call to reduce
        boilerplate when orchestrating steps.
        """
        preprocessor = self.get_preprocessor(model_name, state)
        runner = self.get_runner(model_name, state)
        postprocessor = self.get_postprocessor(model_name, state)
        return preprocessor, runner, postprocessor
