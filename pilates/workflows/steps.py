from __future__ import annotations

# Coupler IO map (manual reference, update when wiring changes).
#
# Step                           Coupler inputs (input_keys)                                 Coupler outputs (keys written)
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# initialization                 (none)                                                      UrbanSim init outputs:
#                                                                                              - usim_datastore_h5
#                                                                                              - omx_skims
#                                                                                              - hh_size
#                                                                                              - income_rates
#                                                                                              - relmap
#                                                                                              - schools
#                                                                                              - school_districts
#
#                                                                                              ActivitySim init outputs:
#                                                                                              - canonical_zones
#                                                                                              - clipped_geoms (if exists)
#                                                                                              - (configs tracked via ActivitySim config adapter)
#
#                                                                                              ATLAS init outputs:
#                                                                                              - one key per non-readme file copied from
#                                                                                                atlas.host_input_folder (or pilates/atlas/atlas_input)
#                                                                                                after scenario filtering. Key is sanitized relative path.
#
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# urbansim_preprocess             (none)                                                      Prepared inputs (from UrbansimPreprocessor._preprocess):
#                                                                                              - geoid_to_zone
#                                                                                              - usim_skims_input_updated (if BEAM skims copied)
#                                                                                              - plus pass-through of initialization inputs:
#                                                                                                usim_datastore_h5, omx_skims, hh_size, income_rates,
#                                                                                                relmap, schools, school_districts, usim_data_reference
#
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (if the input datastore exists)
#
# urbansim_run                    prepared_inputs keys + usim_datastore_h5 (if present)       Raw outputs:
#                                                                                              - usim_forecast_output
#
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (derived from usim_forecast_output)
#
# urbansim_postprocess            usim_datastore_h5                                            Processed outputs:
#                                                                                              - usim_input_archive_<year>
#                                                                                              - usim_input_merged_<year>
#
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (mapped from usim_input_merged_<year>)
#
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# atlas_preprocess                (none)                                                      (no coupler outputs logged in this step)
#
# atlas_run                       usim_datastore_h5                                            Raw outputs:
#                               + all static atlas input keys (if present)                    - one key per ATLAS CSV filename stem
#                               (input_keys)                                                    from expected_output_paths
#
#   Atlas static input keys (explicit; wildcards denote scenario/year variants):
#   Common (always eligible):
#   - accessbility2017
#   - accessbility_2015
#   - cpi
#   - modeaccessibility
#   - psid_names
#   - sfb_baseline
#   - taz_to_tract_sfbay
#   - vehicle_type_mapping_ESS_const_220_price (only if scenario=ess_cons)
#   - vehicle_type_mapping_baseline (only if scenario=baseline)
#   - vehicle_type_mapping_evMandForced2 (only if scenario=zev_mandate)
#
#   Scenario-specific (adopt/<scenario>/...):
#   - adopt_<scenario>_new_vehicle_annual_medians
#   - adopt_<scenario>_new_vehicle_representative_vehicle
#   - adopt_<scenario>_new_vehicles
#   - adopt_<scenario>_new_vehicles_biannual_values_<year>
#   - adopt_<scenario>_used_vehicles
#   - adopt_<scenario>_used_vehicles_<year>
#
# atlas_postprocess               atlas_run raw outputs (all keys above)                       Processed outputs:
#                                                                                              - usim_h5_updated
#                                                                                              - atlas_vehicles2_output
#
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (if updated H5 exists)
#
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# activitysim_preprocess          input_keys: usim_h5_updated (if present) OR usim_datastore_h5 Outputs:
#                               inputs (fallback): usim_datastore_h5 (path)                    - asim_land_use_in
#                                                + asim_mutable_configs_dir                     - asim_households_in
#                                                                                              - asim_persons_in
#                                                                                              - asim_omx_skims (if present)
#
# activitysim_compile             (none)                                                      Outputs:
#                                                                                              - zarr_skims
#
# activitysim_run                 activitysim_preprocess outputs + zarr_skims                 Raw outputs (parquet allowlist; keys as listed):
#                                                                                              - households
#                                                                                              - persons
#                                                                                              - land_use
#                                                                                              - tours
#                                                                                              - trips
#                                                                                              - joint_tour_participants
#                                                                                              - person_windows
#                                                                                              - disaggregate_accessibility
#                                                                                              - proto_households
#                                                                                              - proto_persons
#                                                                                              - proto_persons_merged
#                                                                                              - proto_tours
#                                                                                              - proto_disaggregate_accessibility
#                                                                                              - school_destination_size
#                                                                                              - school_modeled_size
#                                                                                              - school_shadow_prices
#                                                                                              - workplace_destination_size
#                                                                                              - workplace_location_accessibility
#                                                                                              - workplace_modeled_size
#                                                                                              - workplace_shadow_prices
#
# activitysim_postprocess         activitysim_run raw outputs (all keys above)                 Processed outputs:
#                                                                                              - same allowlist as activitysim_run
#                                                                                              Additionally logs:
#                                                                                              - usim_datastore_h5 (if updated H5 exists)
#
# ------------------------------------------------------------------------------------------------ -----------------------------------------------
# beam_preprocess                 (none)                                                      Prepared inputs (from BEAM preprocessor):
#                                                                                              - BEAM_PLANS_IN
#                                                                                              - BEAM_HOUSEHOLDS_IN
#                                                                                              - BEAM_PERSONS_IN
#                                                                                              - LINKSTATS_WARMSTART
#                                                                                              - ATLAS_VEHICLES2_INPUT (if present)
#                                                                                              - vehicles_beam_in (derived from ATLAS vehicles2)
#                                                                                              - plus any {file_stem}_beam_in created by
#                                                                                                preprocessor for other copied files
#
# beam_run                        beam_preprocess outputs (all keys above)                    Raw outputs (keys are base names below, with
#                                                                                              suffix _<year>_<iteration> and optional _sub<it>):
#                                                                                              Iteration-scoped outputs (files_to_get):
#                                                                                              - raw_od_skims
#                                                                                              - raw_od_skims_zarr
#                                                                                              - raw_origin_skims
#                                                                                              - linkstats
#                                                                                              - linkstats_unmodified
#                                                                                              - linkstats_parquet
#                                                                                              - linkstats_unmodified_parquet
#                                                                                              - beam_plans_out
#                                                                                              - beam_plans_xml
#                                                                                              - beam_experienced_plans_xml
#                                                                                              - beam_experienced_plans_scores
#                                                                                              - events
#                                                                                              - events_parquet
#                                                                                              - legs
#                                                                                              - route_history
#                                                                                              - final_vehicles
#                                                                                              - skims_taz
#                                                                                              - skims_taz_agg
#                                                                                              - skims_od
#                                                                                              - skims_od_agg
#                                                                                              - skims_od_vehicle_type
#                                                                                              - skims_od_vehicle_type_agg
#                                                                                              - skims_emissions
#                                                                                              - skims_emissions_agg
#                                                                                              - skims_ridehail_agg
#                                                                                              - skims_parking
#                                                                                              - skims_parking_agg
#                                                                                              - skims_transit_crowding
#                                                                                              - skims_transit_crowding_agg
#                                                                                              - skims_freight
#                                                                                              - skims_freight_agg
#                                                                                              - skims_travel_time_obs_sim
#                                                                                              - skims_travel_time_obs_sim_agg
#
#                                                                                              Top-level outputs (top_level_files):
#                                                                                              - beam_plans_final
#                                                                                              - beam_vehicles_final
#                                                                                              - beam_households_final
#                                                                                              - beam_persons_final
#                                                                                              - beam_population_final
#                                                                                              - beam_network_final
#                                                                                              - beam_output_plans_xml
#                                                                                              - beam_output_experienced_plans_xml
#                                                                                              - beam_output_vehicles_xml
#                                                                                              - beam_output_households_xml
#                                                                                              - beam_output_facilities_xml
#                                                                                              - beam_output_network_xml
#                                                                                              - beam_output_counts_xml
#
# beam_postprocess                beam_run raw outputs (all keys above)                        Outputs:
#                               + zarr_skims (if present)                                     - final_skims_omx OR zarr_skims
#                                                                                              - linkstats (promoted latest)
#                                                                                              - beam_plans_out (promoted latest)
#
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

from consist import define_step

from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.beam_warmstart import find_last_run_output_plans
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    log_and_set_input,
    log_and_set_output,
    log_input_only,
    log_output_only,
    record_store_to_outputs,
    resolve_artifact_from_value,
    update_coupler_from_beam_outputs,
)
from pilates.workflows.outputs_base import (
    deserialize_step_outputs,
    serialize_step_outputs,
)
from pilates.workflows.artifact_constants import (
    ASIM_OMX_SKIMS,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_PLANS_OUT,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_R5_OSM_FILE,
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_H5,
    USIM_H5_UPDATED,
    USIM_INPUT_ARCHIVE_PREFIX,
    ZARR_SKIMS,
    ASIM_HOUSEHOLDS_IN,
    ASIM_PERSONS_IN,
    ASIM_LAND_USE_IN,
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
from pilates.workflows.step_consist_meta import consist_step_meta
from workflow_state import WorkflowState

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.generic.records import RecordStore
    from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


def _warn_missing_coupler_inputs(
    coupler: Optional[CouplerProtocol],
    input_store: Optional[RecordStore],
    context: str,
) -> None:
    if coupler is None or input_store is None:
        return
    keys_attr = getattr(coupler, "keys", None)
    if not callable(keys_attr):
        return
    try:
        coupler_keys = set(keys_attr())
    except Exception:
        return
    missing = []
    for record in input_store.all_records():
        key = getattr(record, "short_name", None) or getattr(record, "unique_id", None)
        if key and key not in coupler_keys:
            missing.append(key)
    if missing:
        logger.warning(
            "[%s] Input RecordStore keys missing from coupler: %s",
            context,
            sorted(set(missing)),
        )

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


def _schema_outputs_from_class(outputs_class: Type[StepOutputsT]) -> Optional[list[str]]:
    record_keys = getattr(outputs_class, "record_keys", None) or {}
    values = [value for value in record_keys.values() if isinstance(value, str)]
    unique = sorted(set(values))
    return unique or None


def _decorate_step_with_consist(
    *,
    step_func: Callable[..., Any],
    step_model: str,
    description: str,
    schema_outputs: Optional[list[str]] = None,
    outputs: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> Callable[..., Any]:
    """
    Attach native Consist step metadata to a workflow step function.
    """
    if hasattr(step_func, "__consist_step__"):
        return step_func

    kwargs: Dict[str, Any] = {
        "model": step_model,
        "description": description,
        "name_template": "{func_name}__y{year}__i{iteration}__phase_{phase}",
        "tags": tags or [step_model],
        **consist_step_meta(step_model),
    }
    if schema_outputs:
        kwargs["schema_outputs"] = schema_outputs
    if outputs:
        kwargs["outputs"] = outputs
    return define_step(**kwargs)(step_func)


def _make_generic_step_function(
    *,
    coupler: CouplerProtocol,
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
            coupler=coupler,
            context=f"{model_name}_{phase}",
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

    step_model = f"{model_name}_{phase}"
    return _decorate_step_with_consist(
        step_func=_step_func,
        step_model=step_model,
        description=f"{model_name} {phase} workflow step",
        schema_outputs=_schema_outputs_from_class(outputs_class),
        tags=[model_name, phase],
    )


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
    coupler: Optional[CouplerProtocol] = None,
    context: str = "runner",
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
    _warn_missing_coupler_inputs(coupler, input_store, context)
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
    coupler: Optional[CouplerProtocol] = None,
    context: str = "beam_preprocess",
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
    _warn_missing_coupler_inputs(coupler, combined, context)
    return preprocessor.preprocess(workspace, combined)


def _execute_beam_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "beam_run",
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
    _warn_missing_coupler_inputs(coupler, input_store, context)
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
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "urbansim_run",
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
    _warn_missing_coupler_inputs(coupler, input_store, context)
    return run_runner(runner, input_store, workspace)


def _execute_urbansim_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "urbansim_postprocess",
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
    _warn_missing_coupler_inputs(coupler, raw_outputs, context)
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def _execute_atlas_run(
    runner: Runner,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "atlas_run",
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
    _warn_missing_coupler_inputs(coupler, input_store, context)
    return run_runner(runner, input_store, workspace)


def _execute_atlas_postprocess(
    postprocessor: Postprocessor,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    coupler: Optional[CouplerProtocol] = None,
    context: str = "atlas_postprocess",
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
    _warn_missing_coupler_inputs(coupler, raw_outputs, context)
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def make_urbansim_preprocess_step(
    *,
    coupler: CouplerProtocol,
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
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
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
    coupler: CouplerProtocol,
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
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
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
    coupler: CouplerProtocol,
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
        for short_name, path, description in outputs._iter_record_items():
            if short_name.startswith(USIM_INPUT_ARCHIVE_PREFIX):
                log_output_only(
                    key=short_name,
                    path=str(path),
                    description=description,
                    profile_file_schema=True,
                    h5_container=True,
                    hash_tables="if_unchanged",
                )
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    "UrbanSim datastore prepared for next iteration "
                    f"(year {state.forecast_year})"
                ),
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
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
    coupler: CouplerProtocol,
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
    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        usim_dir = workspace.get_usim_mutable_data_dir()
        usim_path = None
        if state.is_start_year():
            region_id = settings.urbansim.region_id
            if not region_id:
                region_map = settings.urbansim.region_mappings.get(
                    "region_to_region_id", {}
                )
                region_id = region_map.get(settings.run.region)
            if region_id:
                fname = settings.urbansim.input_file_template.format(region_id=region_id)
                usim_path = os.path.join(usim_dir, fname)
        else:
            fname = settings.urbansim.output_file_template.format(
                year=state.forecast_year
            )
            usim_path = os.path.join(usim_dir, fname)
        if usim_path and os.path.exists(usim_path):
            log_input_only(
                key=USIM_DATASTORE_H5,
                path=usim_path,
                description=(
                    f"UrbanSim datastore for ATLAS year {state.forecast_year}"
                ),
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
            )
        return {}

    def _log_outputs(
        outputs: AtlasPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            meta: Dict[str, Any] = {}
            if str(path).endswith((".csv", ".parquet")):
                meta["profile_file_schema"] = True
            log_output_only(
                key=short_name,
                path=str(path),
                description=description,
                **meta,
            )

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
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_atlas_run_step(
    *,
    coupler: CouplerProtocol,
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
    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.atlas_preprocess
        if upstream is None:
            raise RuntimeError("ATLAS preprocess must complete first")
        for short_name, path, description in upstream._iter_record_items():
            meta: Dict[str, Any] = {}
            if str(path).endswith((".csv", ".parquet")):
                meta["profile_file_schema"] = True
            log_input_only(
                key=short_name,
                path=str(path),
                description=description,
                **meta,
            )
        return {}

    def _log_outputs(
        outputs: AtlasRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            meta: Dict[str, Any] = {}
            if str(path).endswith((".csv", ".parquet")):
                meta["profile_file_schema"] = True
            log_output_only(
                key=short_name,
                path=str(path),
                description=description,
                **meta,
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
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_atlas_postprocess_step(
    *,
    coupler: CouplerProtocol,
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
            meta: Dict[str, Any] = {}
            if str(path).endswith((".csv", ".parquet")):
                meta["profile_file_schema"] = True
            log_output_only(
                key=short_name,
                path=str(path),
                description=description,
                **meta,
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
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
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
    coupler: CouplerProtocol,
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

    return _decorate_step_with_consist(
        step_func=_run_activitysim_compile_step,
        step_model="activitysim_compile",
        description="activitysim compile workflow step",
        outputs=[ZARR_SKIMS],
        schema_outputs=[ZARR_SKIMS],
        tags=["activitysim", "compile"],
    )


def make_activitysim_preprocess_step(
    *,
    coupler: CouplerProtocol,
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

    Notes
    -----
    This step focuses on preparing ActivitySim inputs. Config canonicalization
    is also performed here so config ingestion is tied to the preprocess phase.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        """
        Log ActivitySim preprocess inputs.

        This helper logs the UrbanSim datastore input if it exists in the coupler.

        Parameters
        ----------
        settings : PilatesConfig
            Simulation settings for config root resolution.
        state : WorkflowState
            Current workflow state (used for log metadata only).
        workspace : Workspace
            Workspace used to resolve mutable config paths.
        holder : StepOutputsHolder
            Outputs holder (unused for this helper).

        Returns
        -------
        dict
            Extra runtime kwargs for the step executor (empty for this helper).
        """
        tracker = cr.current_tracker()
        if tracker is not None:
            try:
                from pathlib import Path

                from consist.core.config_canonicalization import ConfigAdapterOptions
                from consist.integrations.activitysim import ActivitySimConfigAdapter
            except Exception:
                logger.debug(
                    "ActivitySim config adapter unavailable; skipping canonicalization."
                )
            else:
                config_root = (
                    Path(workspace.get_asim_mutable_configs_dir())
                    / settings.activitysim.main_configs_dir
                )
                if config_root.exists():
                    options = ConfigAdapterOptions(
                        strict=False,
                        bundle=True,
                        ingest=True,
                        allow_heuristic_refs=True,
                    )
                    current_run = cr.current_run()
                    run_id = getattr(current_run, "id", None) if current_run else None
                    try:
                        if run_id:
                            tracker.canonicalize_config(
                                ActivitySimConfigAdapter(),
                                [config_root],
                                run_id=run_id,
                                options=options,
                            )
                        else:
                            tracker.canonicalize_config(
                                ActivitySimConfigAdapter(),
                                [config_root],
                                options=options,
                            )
                    except Exception:
                        logger.warning(
                            "ActivitySim config canonicalization failed; "
                            "continuing without config ingestion.",
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        "ActivitySim config root not found for canonicalization: %s",
                        config_root,
                    )

        usim_input = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            updated_value = get_value(USIM_H5_UPDATED)
            selected_key = (
                USIM_H5_UPDATED if updated_value is not None else USIM_DATASTORE_H5
            )
            selected_value = (
                updated_value
                if updated_value is not None
                else get_value(USIM_DATASTORE_H5)
            )
            usim_input = resolve_artifact_from_value(
                selected_value,
                key=selected_key,
                workspace=workspace,
            )
        usim_path = artifact_to_path(usim_input, workspace)
        if usim_path and os.path.exists(usim_path):
            input_key = (
                USIM_H5_UPDATED
                if callable(get_value) and get_value(USIM_H5_UPDATED) is not None
                else USIM_DATASTORE_H5
            )
            input_desc = (
                f"UrbanSim datastore updated by ATLAS for ActivitySim year {state.year}"
                if input_key == USIM_H5_UPDATED
                else f"UrbanSim datastore for ActivitySim year {state.year}"
            )
            h5_tables_used = [
                "households",
                "persons",
                "jobs",
                "blocks",
            ]
            start_year = state.start_year
            if start_year is not None:
                h5_tables_used.extend(
                    [
                        f"/{start_year}/households",
                        f"/{start_year}/persons",
                        f"/{start_year}/jobs",
                        f"/{start_year}/blocks",
                    ]
                )
            log_and_set_input(
                key=input_key,
                path=usim_path,
                description=input_desc,
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                h5_tables_used=h5_tables_used,
            )
        return {}

    def _log_outputs(
        outputs: ActivitySimPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        """
        Log ActivitySim preprocess outputs and update the coupler.

        Parameters
        ----------
        outputs : ActivitySimPreprocessOutputs
            Typed outputs containing ActivitySim input tables and skims.
        settings : PilatesConfig
            Simulation settings (unused for output logging).
        state : WorkflowState
            Current workflow state (used for log metadata only).
        workspace : Workspace
            Workspace for path resolution.
        holder : StepOutputsHolder
            Outputs holder (unused for this helper).
        """
        profile_keys = {ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN}
        for short_name, path, description in outputs._iter_record_items():
            meta: Dict[str, Any] = {}
            if short_name in profile_keys:
                meta["profile_file_schema"] = True
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
                **meta,
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
    coupler: CouplerProtocol,
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

        profile_keys = {ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN}
        for short_name, path, description in upstream._iter_record_items():
            meta: Dict[str, Any] = {}
            if short_name in profile_keys:
                meta["profile_file_schema"] = True
            log_and_set_input(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
                **meta,
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
            artifact = cr.log_output(
                str(path),
                key=short_name,
                description=description,
            )
            if artifact is not None and getattr(artifact, "hash", None):
                outputs.raw_output_hashes[short_name] = artifact.hash

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
    coupler: CouplerProtocol,
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

    def _log_outputs(
        outputs: ActivitySimPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        profile_keys = {
            "persons_asim_out",
            "trips_asim_out",
            "tours_asim_out",
            "beam_plans_asim_out",
            "households_asim_out",
        }
        for short_name, path, description in outputs._iter_record_items():
            meta: Dict[str, Any] = {}
            content_hash = outputs.processed_output_hashes.get(short_name)
            if content_hash:
                meta["content_hash"] = content_hash
            if short_name in profile_keys:
                meta["profile_file_schema"] = True
            log_output_only(
                key=short_name,
                path=str(path),
                description=description,
                **meta,
            )
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim datastore updated by ActivitySim for year {state.forecast_year}"
                ),
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
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
        output_logger=_log_outputs,
    )


def make_beam_preprocess_step(
    *,
    coupler: CouplerProtocol,
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

    Notes
    -----
    This step focuses on generating BEAM inputs and canonicalizing BEAM config.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        tracker = cr.current_tracker()
        if tracker is not None:
            from pathlib import Path

            config_root = (
                Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
            )
            try:
                from consist.core.config_canonicalization import ConfigAdapterOptions
                from consist.integrations.beam import BeamConfigAdapter
            except Exception:
                logger.debug(
                    "BEAM config adapter unavailable; skipping canonicalization."
                )
            else:
                primary_config = config_root / settings.beam.config
                if primary_config.exists():
                    options = ConfigAdapterOptions(
                        strict=False,
                        bundle=False,
                        ingest=True,
                        allow_heuristic_refs=True,
                    )
                    current_run = cr.current_run()
                    run_id = getattr(current_run, "id", None) if current_run else None
                    beam_input_root = Path(workspace.get_beam_mutable_data_dir()).resolve()
                    beam_input_root = beam_input_root / settings.run.region
                    pwd_candidates = [
                        beam_input_root.parent,
                        beam_input_root,
                        beam_input_root.parent.parent,
                    ]
                    expected_suffix = Path("input") / settings.run.region
                    pwd_root = next(
                        (
                            root
                            for root in pwd_candidates
                            if (root / expected_suffix).exists()
                        ),
                        beam_input_root.parent,
                    )
                    env_overrides = {"PWD": str(pwd_root)}

                    try:
                        adapter = BeamConfigAdapter(
                            primary_config=primary_config,
                            env_overrides=env_overrides,
                        )
                        if run_id:
                            tracker.canonicalize_config(
                                adapter,
                                [config_root],
                                run_id=run_id,
                                options=options,
                            )
                        else:
                            tracker.canonicalize_config(
                                adapter,
                                [config_root],
                                options=options,
                            )
                    except Exception:
                        logger.warning(
                            "BEAM config canonicalization failed; "
                            "continuing without config ingestion.",
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        "BEAM primary config not found for canonicalization: %s",
                        primary_config,
                    )
        return {}

    def _log_outputs(
        outputs: BeamPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        """
        Log BEAM preprocess outputs and update the coupler.

        This helper logs prepared BEAM input artifacts into the coupler for
        downstream BEAM run and postprocess steps.

        Parameters
        ----------
        outputs : BeamPreprocessOutputs
            Typed outputs containing prepared BEAM inputs.
        settings : PilatesConfig
            Simulation settings for config root resolution.
        state : WorkflowState
            Current workflow state (used for log metadata only).
        workspace : Workspace
            Workspace used to resolve mutable BEAM config paths.
        holder : StepOutputsHolder
            Outputs holder (unused for this helper).
        """
        profile_schema_keys = {
            "households_beam_in",
            "persons_beam_in",
            "plans_beam_in",
        }

        for key, path in outputs.prepared_inputs.items():
            meta: Dict[str, Any] = {}
            if key in profile_schema_keys:
                meta["profile_file_schema"] = True
            log_and_set_output(
                key=key,
                path=str(path),
                description=(
                    f"BEAM prepared input {key} for year {state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
                **meta,
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
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_beam_run_step(
    *,
    coupler: CouplerProtocol,
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

        tracker = cr.current_tracker()
        if tracker is not None:
            from pathlib import Path

            config_root = (
                Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
            )

            # Log the BEAM R5 OSM file referenced in the resolved config if available.
            try:
                from sqlmodel import Session, select
                from consist.models.beam import BeamConfigCache, BeamConfigIngestRunLink
            except Exception:
                logger.debug("SQLModel/Consist beam models unavailable; skipping OSM logging.")
            else:
                current_run = cr.current_run()
                run_id = getattr(current_run, "id", None) if current_run else None
                if run_id and tracker.db is not None:
                    try:
                        with Session(tracker.db.engine) as session:
                            base_stmt = (
                                select(
                                    BeamConfigCache.value_str,
                                    BeamConfigCache.content_hash,
                                )
                                .join(
                                    BeamConfigIngestRunLink,
                                    BeamConfigCache.content_hash
                                    == BeamConfigIngestRunLink.content_hash,
                                )
                                .where(BeamConfigIngestRunLink.run_id == run_id)
                            )
                            config_name = settings.beam.config
                            if config_name:
                                base_stmt = base_stmt.where(
                                    BeamConfigIngestRunLink.config_name == config_name
                                )
                            osm_rows = session.exec(
                                base_stmt.where(
                                    BeamConfigCache.key == "beam.routing.r5.osmFile"
                                )
                            ).all()
                            if len(osm_rows) > 1:
                                logger.warning(
                                    "Multiple BEAM osmFile rows found for run_id=%s config=%s",
                                    run_id,
                                    config_name,
                                )
                            osm_value = osm_rows[0][0] if osm_rows else None
                            osm_hash = osm_rows[0][1] if osm_rows else None
                            logger.debug(
                                "BEAM config osmFile resolved value: %s (run_id=%s, db=%s, config=%s, hash=%s)",
                                osm_value,
                                run_id,
                                tracker.db.engine.url,
                                config_name,
                                osm_hash,
                            )
                            if osm_value == "/":
                                all_osm_rows = session.exec(
                                    select(
                                        BeamConfigIngestRunLink.config_name,
                                        BeamConfigCache.value_str,
                                        BeamConfigCache.content_hash,
                                    )
                                    .join(
                                        BeamConfigCache,
                                        BeamConfigCache.content_hash
                                        == BeamConfigIngestRunLink.content_hash,
                                    )
                                    .where(BeamConfigIngestRunLink.run_id == run_id)
                                    .where(BeamConfigCache.key == "beam.routing.r5.osmFile")
                                ).all()
                                logger.warning(
                                    "BEAM osmFile resolved to '/' for run_id=%s; rows=%s",
                                    run_id,
                                    all_osm_rows,
                                )
                            resolved_osm_path = None
                            if osm_value and "${" not in osm_value:
                                resolved_osm_path = osm_value
                                if not os.path.isabs(resolved_osm_path):
                                    resolved_osm_path = str(
                                        (config_root / resolved_osm_path).resolve()
                                    )
                                if not os.path.exists(resolved_osm_path):
                                    resolved_osm_path = None

                            if resolved_osm_path is None:
                                mapdb_row = session.exec(
                                    base_stmt.where(
                                        BeamConfigCache.key
                                        == "beam.routing.r5.osmMapdbFile"
                                    )
                                ).first()
                                mapdb_value = mapdb_row[0] if mapdb_row else None
                                logger.debug(
                                    "BEAM config osmMapdbFile resolved value: %s",
                                    mapdb_value,
                                )
                                if mapdb_value and "${" not in mapdb_value:
                                    resolved_osm_path = mapdb_value
                                    if not os.path.isabs(resolved_osm_path):
                                        resolved_osm_path = str(
                                            (config_root / resolved_osm_path).resolve()
                                        )
                                    if not os.path.exists(resolved_osm_path):
                                        resolved_osm_path = None

                            if resolved_osm_path:
                                cr.log_input(
                                    resolved_osm_path,
                                    key=BEAM_R5_OSM_FILE,
                                    description=(
                                        "BEAM R5 OSM input referenced by config"
                                    ),
                                )
                    except Exception:
                        logger.debug(
                            "Failed to resolve/log BEAM R5 OSM file from config.",
                            exc_info=True,
                        )
        for short_name, path, description in upstream._iter_record_items():
            log_and_set_input(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
            )

        from pathlib import Path

        output_root = Path(workspace.get_beam_output_dir()) / settings.run.region
        plans_path, experienced_path = find_last_run_output_plans(
            output_root, "year-"
        )
        if plans_path is not None and plans_path.exists():
            if plans_path.name == "output_plans.xml.gz":
                plans_key = BEAM_OUTPUT_PLANS_XML
            else:
                plans_key = BEAM_PLANS_OUT
            log_and_set_input(
                key=plans_key,
                path=str(plans_path),
                description=(
                    "BEAM warm-start plans (selected by BEAM from previous outputs)"
                ),
                coupler=coupler,
            )
        if experienced_path is not None and experienced_path.exists():
            if experienced_path.name == "output_experienced_plans.xml.gz":
                experienced_key = BEAM_OUTPUT_EXPERIENCED_PLANS_XML
            else:
                experienced_key = BEAM_EXPERIENCED_PLANS_XML
            log_and_set_input(
                key=experienced_key,
                path=str(experienced_path),
                description=(
                    "BEAM warm-start experienced plans (selected by BEAM from previous outputs)"
                ),
                coupler=coupler,
            )
        return {}

    def _log_outputs(
        outputs: BeamRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            meta: Dict[str, Any] = {}
            if short_name.startswith("beam_network_final"):
                meta["profile_file_schema"] = "if_changed"
                meta["reuse_if_unchanged"] = True
                meta["reuse_scope"] = "any_uri"
                try:
                    from pilates.database.schema.beam_schema import BeamNetworkFinal
                except Exception:
                    BeamNetworkFinal = None
                if BeamNetworkFinal is not None:
                    meta["schema"] = BeamNetworkFinal
            log_output_only(
                key=short_name,
                path=str(path),
                description=description,
                **meta,
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
    coupler: CouplerProtocol,
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

    def _log_outputs(
        outputs: BeamPostprocessOutputs,
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
        for short_name, path in outputs.split_events.items():
            log_output_only(
                key=short_name,
                path=str(path),
                description=f"BEAM events parquet split ({short_name})",
                profile_file_schema=True,
            )
        for short_name, path in outputs.split_event_links.items():
            log_output_only(
                key=short_name,
                path=str(path),
                description=f"BEAM events link table ({short_name})",
                profile_file_schema=True,
            )
        upstream = holder.beam_run
        if upstream is None:
            return
        combined_outputs = RecordStore()
        combined_outputs += upstream.to_record_store()
        combined_outputs += outputs.to_record_store()
        update_coupler_from_beam_outputs(combined_outputs, coupler, workspace)

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
        output_logger=_log_outputs,
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
