from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    TYPE_CHECKING,
    Type,
    TypeVar,
)

from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    log_and_set_input,
    log_and_set_output,
    log_input_only,
    log_output_only,
    record_store_to_outputs,
    resolve_artifact_from_value,
)
from pilates.workflows.outputs_base import (
    deserialize_step_outputs,
    serialize_step_outputs,
)
from pilates.workflows.artifact_constants import (
    ASIM_OMX_SKIMS,
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.step_exec import (
    Postprocessor,
    Preprocessor,
    Runner,
    run_postprocessor,
    run_preprocessor,
    run_runner,
    warm_start_activities,
)
from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.postprocessor import get_usim_datastore_fname
from pilates.beam.outputs import (
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
)
from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
)
from workflow_state import WorkflowState

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.generic.records import RecordStore
    from pilates.workspace import Workspace

logger = logging.getLogger(__name__)

StepOutputsT = TypeVar("StepOutputsT")
InputLogger = Callable[
    ["PilatesConfig", WorkflowState, "Workspace", "StepOutputsHolder"],
    Mapping[str, Any],
]
OutputLogger = Callable[
    [StepOutputsT, "PilatesConfig", WorkflowState, "Workspace", "StepOutputsHolder"],
    None,
]


@dataclass
class StepOutputsHolder:
    """
    Accumulates typed step outputs across the workflow.

    This holder acts as the in-memory handoff between granular steps so that
    each model phase can consume the outputs produced by its predecessors
    without re-querying the coupler or filesystem.

    Attributes
    ----------
    activitysim_preprocess : ActivitySimPreprocessOutputs, optional
        Preprocess outputs.
    activitysim_run : ActivitySimRunOutputs, optional
        Run outputs.
    activitysim_postprocess : ActivitySimPostprocessOutputs, optional
        Postprocess outputs.
    beam_preprocess : BeamPreprocessOutputs, optional
        Preprocess outputs.
    beam_run : BeamRunOutputs, optional
        Run outputs.
    beam_postprocess : BeamPostprocessOutputs, optional
        Postprocess outputs.
    urbansim_preprocess : UrbanSimPreprocessOutputs, optional
        Preprocess outputs.
    urbansim_run : UrbanSimRunOutputs, optional
        Run outputs.
    urbansim_postprocess : UrbanSimPostprocessOutputs, optional
        Postprocess outputs.
    atlas_preprocess : AtlasPreprocessOutputs, optional
        Preprocess outputs.
    atlas_run : AtlasRunOutputs, optional
        Run outputs.
    atlas_postprocess : AtlasPostprocessOutputs, optional
        Postprocess outputs.
    """

    activitysim_preprocess: Optional[ActivitySimPreprocessOutputs] = None
    activitysim_run: Optional[ActivitySimRunOutputs] = None
    activitysim_postprocess: Optional[ActivitySimPostprocessOutputs] = None
    beam_preprocess: Optional[BeamPreprocessOutputs] = None
    beam_run: Optional[BeamRunOutputs] = None
    beam_postprocess: Optional[BeamPostprocessOutputs] = None
    urbansim_preprocess: Optional[UrbanSimPreprocessOutputs] = None
    urbansim_run: Optional[UrbanSimRunOutputs] = None
    urbansim_postprocess: Optional[UrbanSimPostprocessOutputs] = None
    atlas_preprocess: Optional[AtlasPreprocessOutputs] = None
    atlas_run: Optional[AtlasRunOutputs] = None
    atlas_postprocess: Optional[AtlasPostprocessOutputs] = None

    def set_attribute(self, step_name: str, outputs: Any) -> None:
        """
        Set a holder attribute by step name.

        Parameters
        ----------
        step_name : str
            Step name key.
        outputs : Any
            Outputs object to store.
        """
        attr = step_name.replace("-", "_")
        setattr(self, attr, outputs)

    def get_attribute(self, step_name: str) -> Any:
        """
        Retrieve a holder attribute by step name.

        Parameters
        ----------
        step_name : str
            Step name key.

        Returns
        -------
        Any
            Outputs object or None if missing.
        """
        attr = step_name.replace("-", "_")
        return getattr(self, attr, None)


STEP_OUTPUTS_CLASSES = {
    "activitysim_preprocess": ActivitySimPreprocessOutputs,
    "activitysim_run": ActivitySimRunOutputs,
    "activitysim_postprocess": ActivitySimPostprocessOutputs,
    "beam_preprocess": BeamPreprocessOutputs,
    "beam_run": BeamRunOutputs,
    "beam_postprocess": BeamPostprocessOutputs,
    "urbansim_preprocess": UrbanSimPreprocessOutputs,
    "urbansim_run": UrbanSimRunOutputs,
    "urbansim_postprocess": UrbanSimPostprocessOutputs,
    "atlas_preprocess": AtlasPreprocessOutputs,
    "atlas_run": AtlasRunOutputs,
    "atlas_postprocess": AtlasPostprocessOutputs,
}


STEP_DEPENDENCIES = {
    "urbansim_preprocess": {
        "depends_on": [],
        "holder_inputs": [],
    },
    "urbansim_run": {
        "depends_on": ["urbansim_preprocess"],
        "holder_inputs": ["urbansim_preprocess"],
    },
    "urbansim_postprocess": {
        "depends_on": ["urbansim_run"],
        "holder_inputs": ["urbansim_run"],
    },
    "atlas_preprocess": {
        "depends_on": [],
        "holder_inputs": [],
    },
    "atlas_run": {
        "depends_on": ["atlas_preprocess"],
        "holder_inputs": ["atlas_preprocess"],
    },
    "atlas_postprocess": {
        "depends_on": ["atlas_run"],
        "holder_inputs": ["atlas_run"],
    },
    "activitysim_preprocess": {
        "depends_on": [],
        "holder_inputs": [],
    },
    "activitysim_run": {
        "depends_on": ["activitysim_preprocess"],
        "holder_inputs": ["activitysim_preprocess"],
    },
    "activitysim_postprocess": {
        "depends_on": ["activitysim_run"],
        "holder_inputs": ["activitysim_run"],
    },
    "beam_preprocess": {
        "depends_on": ["activitysim_postprocess"],
        "holder_inputs": ["activitysim_postprocess"],
    },
    "beam_run": {
        "depends_on": ["beam_preprocess"],
        "holder_inputs": ["beam_preprocess"],
    },
    "beam_postprocess": {
        "depends_on": ["beam_run"],
        "holder_inputs": ["beam_run"],
    },
}


def validate_step_ready(step_name: str, outputs_holder: StepOutputsHolder) -> None:
    """
    Validate that dependencies for a step are satisfied.

    This enforces the expected execution order (e.g., preprocess before run)
    so that downstream steps can rely on required outputs being present.

    Parameters
    ----------
    step_name : str
        Step name to validate.
    outputs_holder : StepOutputsHolder
        Holder containing upstream outputs.
    """
    spec = STEP_DEPENDENCIES.get(step_name)
    if not spec:
        logger.warning("No dependency spec for %s; skipping validation", step_name)
        return
    for holder_input_key in spec["holder_inputs"]:
        holder_attr = holder_input_key.replace("-", "_")
        if getattr(outputs_holder, holder_attr, None) is None:
            raise RuntimeError(
                f"{step_name} requires {holder_input_key} to complete first, "
                "but it has not been executed or failed."
            )


def require_common_runtime(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Require runtime kwargs shared across step functions.

    Parameters
    ----------
    *names : str
        Additional required runtime kwarg names.

    Returns
    -------
    callable
        Decorator that enforces the runtime kwargs.
    """
    return cr.require_runtime_kwargs("settings", "state", "workspace", *names)


def _make_generic_step_function(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
    model_name: str,
    phase: str,
    outputs_class: Type[StepOutputsT],
    component_getter: Callable[[ModelFactory, WorkflowState], Any],
    component_executor: Callable[..., RecordStore],
    outputs_holder_setter: Callable[[StepOutputsHolder, StepOutputsT], None],
    input_logger: Optional[InputLogger] = None,
    output_logger: Optional[OutputLogger] = None,
) -> Callable[..., None]:
    """
    Build a step function with common RecordStore-to-StepOutputs plumbing.

    The returned function executes a model component (preprocess/run/postprocess),
    converts its RecordStore outputs into a typed outputs dataclass, validates
    the outputs, logs any configured inputs/outputs for provenance, and stores
    the results in the shared outputs holder.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder used to store outputs for downstream steps.
    model_name : str
        Model identifier for logging.
    phase : str
        Step phase name (preprocess/run/postprocess).
    outputs_class : type
        StepOutputs dataclass type.
    component_getter : callable
        Callable that returns the component instance.
    component_executor : callable
        Callable that executes the component.
    outputs_holder_setter : callable
        Callback that stores outputs on the holder.
    input_logger : callable, optional
        Optional hook for logging step inputs.
    output_logger : callable, optional
        Optional hook for logging step outputs.

    Returns
    -------
    callable
        Step function compatible with Consist scenario execution.
    """

    @cr.require_runtime_kwargs("settings", "state", "workspace")
    def _step_func(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        **kwargs: Any,
    ) -> None:
        logger.debug("Starting %s %s step", model_name, phase)
        factory = ModelFactory()
        component = component_getter(factory, state)

        extra_kwargs: Dict[str, Any] = {}
        if input_logger is not None:
            extra_kwargs = (
                input_logger(settings, state, workspace, outputs_holder) or {}
            )
            logger.debug(
                "%s %s input logger keys: %s",
                model_name,
                phase,
                list(extra_kwargs.keys()),
            )

        record_store = component_executor(
            component,
            workspace,
            outputs_holder,
            **extra_kwargs,
            **kwargs,
        )
        if record_store is not None:
            try:
                record_keys = list(record_store.to_mapping().keys())
            except AttributeError:
                record_keys = []
            logger.debug(
                "%s %s record store keys: %s",
                model_name,
                phase,
                record_keys,
            )

        step_outputs = record_store_to_outputs(
            record_store=record_store,
            output_class=outputs_class,
            workspace=workspace,
        )
        step_outputs.validate()
        outputs_holder_setter(outputs_holder, step_outputs)

        if output_logger is not None:
            output_logger(step_outputs, settings, state, workspace, outputs_holder)

        logger.info("%s %s completed successfully", model_name, phase)

    return _step_func


def _execute_preprocess(
    preprocessor: Preprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a preprocessor using only the workspace.

    This phase typically prepares model-specific inputs (copying, formatting,
    or deriving tables) in the mutable workspace for the runner.

    Parameters
    ----------
    preprocessor : object
        Preprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder for upstream outputs (unused).

    Returns
    -------
    RecordStore
        Preprocessor outputs.
    """
    return run_preprocessor(preprocessor, workspace)


def _execute_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    extra_inputs: Optional[RecordStore] = None,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a runner using upstream preprocess outputs.

    This phase performs the core model simulation (e.g., activity demand,
    land use, or traffic assignment) using the prepared inputs.

    Parameters
    ----------
    runner : object
        Runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.
    extra_inputs : RecordStore, optional
        Additional inputs to merge into the runner input store.

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.activitysim_preprocess
    if upstream is None:
        raise RuntimeError("ActivitySim preprocess must complete first")
    input_store = upstream.to_record_store()
    if extra_inputs is not None:
        input_store += extra_inputs
    return run_runner(runner, input_store, workspace)


def _execute_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a postprocessor using upstream run outputs.

    This phase adapts raw model outputs for downstream consumption (e.g.,
    updating HDF5 inputs, producing summary outputs, or deriving skims).

    Parameters
    ----------
    postprocessor : object
        Postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.activitysim_run
    if upstream is None:
        raise RuntimeError("ActivitySim run must complete first")
    raw_outputs = upstream.to_record_store()
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def _execute_beam_preprocess(
    preprocessor: Preprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    activity_demand_outputs: Optional[RecordStore] = None,
    previous_beam_outputs: Optional[RecordStore] = None,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the BEAM preprocessor with upstream RecordStore inputs.

    BEAM preprocess builds the runnable scenario inputs by combining
    ActivitySim demand outputs with warm-start data and optional ATLAS
    vehicle ownership inputs.

    Parameters
    ----------
    preprocessor : object
        BEAM preprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder for upstream outputs (unused).
    activity_demand_outputs : RecordStore, optional
        ActivitySim postprocess outputs.
    previous_beam_outputs : RecordStore, optional
        Previous BEAM outputs for warm starts.

    Returns
    -------
    RecordStore
        Preprocessor outputs.
    """
    combined = RecordStore()
    if activity_demand_outputs is not None:
        combined += activity_demand_outputs
    if previous_beam_outputs is not None:
        combined += previous_beam_outputs
    return preprocessor.preprocess(workspace, combined)


def _execute_beam_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    extra_inputs: Optional[RecordStore] = None,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the BEAM runner using preprocess outputs.

    BEAM run performs the traffic assignment simulation, producing linkstats,
    skims, plans, and event outputs.

    Parameters
    ----------
    runner : object
        BEAM runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.
    extra_inputs : RecordStore, optional
        Additional inputs (e.g., zarr skims).

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")
    input_store = upstream.to_record_store()
    if extra_inputs is not None:
        input_store += extra_inputs
    return run_runner(runner, input_store, workspace)


def _execute_beam_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the BEAM postprocessor using run outputs.

    BEAM postprocess merges updated skims and writes final skim artifacts for
    downstream models.

    Parameters
    ----------
    postprocessor : object
        BEAM postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.beam_run
    if upstream is None:
        raise RuntimeError("BEAM run must complete first")
    raw_outputs = upstream.to_record_store()
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def _execute_urbansim_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the UrbanSim runner using preprocess outputs.

    UrbanSim run performs the land-use forecast between the base year and
    forecast year, writing the UrbanSim datastore for downstream steps.

    Parameters
    ----------
    runner : object
        UrbanSim runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.urbansim_preprocess
    if upstream is None:
        raise RuntimeError("UrbanSim preprocess must complete first")
    input_store = upstream.to_record_store()
    return run_runner(runner, input_store, workspace)


def _execute_urbansim_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the UrbanSim postprocessor using run outputs.

    UrbanSim postprocess prepares an updated input datastore for subsequent
    model stages (ActivitySim/ATLAS).

    Parameters
    ----------
    postprocessor : object
        UrbanSim postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.urbansim_run
    if upstream is None:
        raise RuntimeError("UrbanSim run must complete first")
    raw_outputs = upstream.to_record_store()
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def _execute_atlas_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the ATLAS runner using preprocess outputs.

    ATLAS run simulates vehicle ownership evolution for the sub-year and
    produces household vehicle ownership outputs.

    Parameters
    ----------
    runner : object
        ATLAS runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.atlas_preprocess
    if upstream is None:
        raise RuntimeError("ATLAS preprocess must complete first")
    input_store = upstream.to_record_store()
    return run_runner(runner, input_store, workspace)


def _execute_atlas_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute the ATLAS postprocessor using run outputs.

    ATLAS postprocess updates the UrbanSim HDF5 datastore and derives
    vehicles2 outputs for downstream BEAM runs.

    Parameters
    ----------
    postprocessor : object
        ATLAS postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.atlas_run
    if upstream is None:
        raise RuntimeError("ATLAS run must complete first")
    raw_outputs = upstream.to_record_store()
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def make_urbansim_preprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the UrbanSim preprocess step function.

    This step prepares land-use inputs by materializing the UrbanSim mutable
    data directory, ensuring warm-start activities (if enabled), and making
    required input tables available for the land-use runner.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for UrbanSim preprocess.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        if state.is_start_year() and settings.activitysim.warm_start_activities:
            logger.info("[Main] Running warm start activities for ActivitySim.")
            warm_start_activities(settings, state, workspace)
        return {}

    def _log_outputs(
        outputs: UrbanSimPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
            )
        usim_data_dir = outputs.usim_mutable_data_dir
        usim_input_fname = settings.urbansim.input_file_template.format(
            region_id=settings.urbansim.region_mappings["region_to_region_id"][
                settings.run.region
            ]
        )
        usim_input_path = usim_data_dir / usim_input_fname
        if usim_input_path.exists():
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(usim_input_path),
                description="UrbanSim input datastore for preprocessing",
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="urbansim",
        phase="preprocess",
        outputs_class=UrbanSimPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "urbansim", state, WorkflowState.Stage.land_use
        ),
        component_executor=_execute_preprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "urbansim_preprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_urbansim_run_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the UrbanSim run step function.

    This step executes the UrbanSim land-use simulation for the forecast year
    and produces the UrbanSim datastore output.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for UrbanSim run.
    """

    def _log_outputs(
        outputs: UrbanSimRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim datastore output for year {state.forecast_year}"
                ),
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="urbansim",
        phase="run",
        outputs_class=UrbanSimRunOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "urbansim", state, WorkflowState.Stage.land_use
        ),
        component_executor=_execute_urbansim_run,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "urbansim_run", outputs
        ),
        output_logger=_log_outputs,
    )


def make_urbansim_postprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the UrbanSim postprocess step function.

    This step merges UrbanSim outputs into the input datastore used by
    downstream models and prepares the HDF5 for the next stage.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for UrbanSim postprocess.
    """

    def _log_outputs(
        outputs: UrbanSimPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    "UrbanSim datastore prepared for next iteration "
                    f"(year {state.forecast_year})"
                ),
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="urbansim",
        phase="postprocess",
        outputs_class=UrbanSimPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "urbansim", state, WorkflowState.Stage.land_use
        ),
        component_executor=_execute_urbansim_postprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "urbansim_postprocess", outputs
        ),
        output_logger=_log_outputs,
    )


def make_atlas_preprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ATLAS preprocess step function.

    This step extracts UrbanSim HDF5 tables into ATLAS input CSVs and optionally
    computes accessibility metrics required by the ATLAS vehicle ownership model.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for ATLAS preprocess.
    """
    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="atlas",
        phase="preprocess",
        outputs_class=AtlasPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "atlas", state, WorkflowState.Stage.vehicle_ownership_model
        ),
        component_executor=_execute_preprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "atlas_preprocess", outputs
        ),
    )


def make_atlas_run_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ATLAS run step function.

    This step runs ATLAS to simulate vehicle ownership for the sub-year and
    produces household/vehicle output CSVs.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for ATLAS run.
    """

    def _log_outputs(
        outputs: AtlasRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="atlas",
        phase="run",
        outputs_class=AtlasRunOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "atlas", state, WorkflowState.Stage.vehicle_ownership_model
        ),
        component_executor=_execute_atlas_run,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "atlas_run", outputs
        ),
        output_logger=_log_outputs,
    )


def make_atlas_postprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ATLAS postprocess step function.

    This step updates the UrbanSim HDF5 datastore with ATLAS vehicle ownership
    results and writes vehicles2 outputs used by BEAM.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for ATLAS postprocess.
    """

    def _log_outputs(
        outputs: AtlasPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
            )
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    "UrbanSim datastore updated by ATLAS for year "
                    f"{state.forecast_year}"
                ),
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="atlas",
        phase="postprocess",
        outputs_class=AtlasPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "atlas", state, WorkflowState.Stage.vehicle_ownership_model
        ),
        component_executor=_execute_atlas_postprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "atlas_postprocess", outputs
        ),
        output_logger=_log_outputs,
    )


def make_activitysim_compile_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim compile step function.

    This step compiles ActivitySim inputs (prepared by preprocess) into
    optimized skim artifacts (Zarr) used by ActivitySim runs and BEAM.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.

    Returns
    -------
    callable
        Step function for ActivitySim compile.
    """

    @require_common_runtime("expected_outputs")
    def _run_activitysim_compile_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        expected_outputs: Dict[str, Any],
    ) -> None:
        factory = ModelFactory()
        from workflow_state import WorkflowState as _WorkflowState

        compile_runner = factory.get_runner(
            "activitysim_compile",
            state,
            major_stage=_WorkflowState.Stage.activity_demand,
        )

        upstream = outputs_holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError(
                "ActivitySim compile must run after activitysim_preprocess"
            )
        input_store = upstream.to_record_store()
        omx_record = None
        if input_store:
            for record in input_store.all_records():
                if getattr(record, "short_name", None) == ASIM_OMX_SKIMS:
                    omx_record = record
                    break
        if omx_record is not None:
            input_store = RecordStore(recordList=[omx_record])
        else:
            input_store = RecordStore()
        if omx_record is not None:
            omx_path = omx_record.get_absolute_path(base_path=workspace.full_path)
            if omx_path and os.path.exists(omx_path):
                cr.log_input(
                    omx_path,
                    key=ASIM_OMX_SKIMS,
                    description="ActivitySim compile input skims (OMX)",
                )
        compile_outputs = compile_runner.run(input_store, workspace)

        zarr_record = None
        if compile_outputs:
            for record in compile_outputs.all_records():
                if record.short_name == ZARR_SKIMS:
                    zarr_record = record
                    break
        zarr_output_path = expected_outputs.get(ZARR_SKIMS)
        if not zarr_output_path and zarr_record is not None:
            zarr_output_path = zarr_record.file_path
        if zarr_output_path and os.path.exists(zarr_output_path):
            log_and_set_output(
                key=ZARR_SKIMS,
                path=zarr_output_path,
                description="ActivitySim compiled zarr skims",
                coupler=coupler,
            )

    return _run_activitysim_compile_step


def make_activitysim_preprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim preprocess step function.

    This step prepares ActivitySim inputs from UrbanSim outputs and ensures
    the mutable input directory contains the required tables and skims.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for ActivitySim preprocess.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        usim_input = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            usim_input = resolve_artifact_from_value(
                get_value(USIM_DATASTORE_H5),
                key=USIM_DATASTORE_H5,
                workspace=workspace,
            )
        usim_path = artifact_to_path(usim_input, workspace)
        if usim_path and os.path.exists(usim_path):
            log_and_set_input(
                key=USIM_DATASTORE_H5,
                path=usim_path,
                description=(f"UrbanSim datastore for ActivitySim year {state.year}"),
                coupler=coupler,
            )
        return {}

    def _log_outputs(
        outputs: ActivitySimPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="preprocess",
        outputs_class=ActivitySimPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_preprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_preprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_activitysim_run_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim run step function.

    This step executes the activity demand model for the year/iteration and
    produces household/person/tour outputs consumed by BEAM and postprocessing.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for ActivitySim run.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError("ActivitySim preprocess must complete first")

        for short_name, path, description in upstream._iter_record_items():
            log_and_set_input(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
            )

        extra_inputs = RecordStore()
        zarr_value = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            zarr_value = resolve_artifact_from_value(
                get_value(ZARR_SKIMS),
                key=ZARR_SKIMS,
                workspace=workspace,
            )
        zarr_path = artifact_to_path(zarr_value, workspace)
        if not zarr_path:
            candidate = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
            if os.path.exists(candidate):
                zarr_path = candidate
        if zarr_path and os.path.exists(zarr_path):
            extra_inputs.add_record(
                FileRecord(
                    file_path=zarr_path,
                    short_name=ZARR_SKIMS,
                    description="Compiled ActivitySim skims (Zarr)",
                )
            )
            log_and_set_input(
                key=ZARR_SKIMS,
                path=zarr_path,
                description=(
                    f"ActivitySim compiled skims for year {state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
            )
        return {"extra_inputs": extra_inputs}

    def _log_outputs(
        outputs: ActivitySimRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            log_output_only(
                key=short_name,
                path=str(path),
                description=description,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="run",
        outputs_class=ActivitySimRunOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_run,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_run", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_activitysim_postprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim postprocess step function.

    This step converts ActivitySim outputs into downstream inputs and updates
    the UrbanSim datastore for the next model stages.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for ActivitySim postprocess.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.activitysim_run
        if upstream is None:
            raise RuntimeError("ActivitySim run must complete first")
        for short_name, path, description in upstream._iter_record_items():
            log_input_only(
                key=short_name,
                path=str(path),
                description=description,
            )
        usim_input_fname = get_usim_datastore_fname(settings, io="input")
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        if os.path.exists(usim_input_path):
            log_and_set_input(
                key=USIM_DATASTORE_H5,
                path=usim_input_path,
                description="UrbanSim datastore used by ActivitySim postprocess",
                coupler=coupler,
            )
        return {}

    def _log_outputs(
        outputs: ActivitySimPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim datastore updated by ActivitySim for year {state.forecast_year}"
                ),
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="postprocess",
        outputs_class=ActivitySimPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_postprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_postprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_beam_preprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM preprocess step function.

    This step builds the BEAM scenario inputs by transforming ActivitySim
    demand outputs, adding ATLAS vehicles (if enabled), and staging warm-start
    artifacts such as linkstats.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for BEAM preprocess.
    """

    def _log_outputs(
        outputs: BeamPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for key, path in outputs.prepared_inputs.items():
            log_and_set_output(
                key=key,
                path=str(path),
                description=(
                    f"BEAM prepared input {key} for year {state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="beam",
        phase="preprocess",
        outputs_class=BeamPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "beam", state, WorkflowState.Stage.traffic_assignment
        ),
        component_executor=_execute_beam_preprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "beam_preprocess", outputs
        ),
        output_logger=_log_outputs,
    )


def make_beam_run_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM run step function.

    This step performs the traffic assignment simulation for the current
    iteration and produces linkstats, skims, plans, and event outputs.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for BEAM run.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.beam_preprocess
        if upstream is None:
            raise RuntimeError("BEAM preprocess must complete first")
        for short_name, path, description in upstream._iter_record_items():
            log_and_set_input(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
            )

        extra_inputs = RecordStore()
        zarr_value = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            zarr_value = resolve_artifact_from_value(
                get_value(ZARR_SKIMS),
                key=ZARR_SKIMS,
                workspace=workspace,
            )
        zarr_path = artifact_to_path(zarr_value, workspace)
        if zarr_path and os.path.exists(zarr_path):
            extra_inputs.add_record(
                FileRecord(
                    file_path=zarr_path,
                    short_name=ZARR_SKIMS,
                    description="Current skims (Zarr) for BEAM",
                )
            )
            log_and_set_input(
                key=ZARR_SKIMS,
                path=zarr_path,
                description=(
                    f"Zarr skims input for BEAM year {state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
            )
        return {"extra_inputs": extra_inputs}

    def _log_outputs(
        outputs: BeamRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            log_output_only(
                key=short_name,
                path=str(path),
                description=description,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="beam",
        phase="run",
        outputs_class=BeamRunOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "beam", state, WorkflowState.Stage.traffic_assignment
        ),
        component_executor=_execute_beam_run,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "beam_run", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_beam_postprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM postprocess step function.

    This step merges BEAM outputs into updated skims and produces final
    skim artifacts for ActivitySim and UrbanSim inputs.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for BEAM postprocess.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.beam_run
        if upstream is None:
            raise RuntimeError("BEAM run must complete first")
        for short_name, path, description in upstream._iter_record_items():
            log_and_set_input(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
            )
        return {}

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="beam",
        phase="postprocess",
        outputs_class=BeamPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "beam", state, WorkflowState.Stage.traffic_assignment
        ),
        component_executor=_execute_beam_postprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "beam_postprocess", outputs
        ),
        input_logger=_log_inputs,
    )


def make_postprocessing_step() -> Callable[..., None]:
    """
    Build the postprocessing step function.

    This step runs optional post-run cleanup/export routines such as event
    processing and copying outputs to external destinations.

    Returns
    -------
    callable
        Step function for postprocessing.
    """

    @require_common_runtime()
    def _run_postprocessing_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
    ) -> None:
        if "postprocessing" in settings:
            from pilates.postprocessing.postprocessor import (
                copy_outputs_to_mep,
                process_event_file,
            )

            process_event_file(settings, state.forecast_year, state.current_inner_iter)
            copy_outputs_to_mep(
                settings,
                state.forecast_year,
                state.current_inner_iter,
                workspace,
            )

    return _run_postprocessing_step
