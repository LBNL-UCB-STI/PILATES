from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

from pilates.config.models import PilatesConfig
from pilates.runtime.context import (
    WorkflowRuntimeContext,
    ensure_workflow_runtime_context,
)
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.formatting import formatted_print
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    resolve_artifact_from_value,
)
from pilates.workflows.binding import (
    ArtifactBindingRule,
    build_binding_plan,
    build_key_only_binding_plan,
)
from pilates.workflows.orchestration import (
    ManifestConfig,
    StageRunner,
    StepRef,
    run_workflow,
)
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
)
from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.activitysim.outputs import (
    ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    ActivitySimPreprocessOutputs,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pilates.workflows.surface import EnabledWorkflowSurface

_ACTIVITYSIM_PILOT_H5_ROLE_KEYS = (
    USIM_POPULATION_SOURCE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_BASE_H5,
)


def _activitysim_population_year(state: WorkflowState) -> int:
    population_year = getattr(state, "forecast_year", None)
    if population_year is None:
        population_year = getattr(state, "year", None)
    if population_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year or WorkflowState.year must be set before "
            "ActivitySim population input resolution."
        )
    return int(population_year)


def _resolve_activitysim_postprocess_h5_role_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_usim_inputs: Mapping[str, Union[str, os.PathLike]],
    postprocess_required_keys: tuple[str, ...],
    postprocess_optional_keys: tuple[str, ...],
) -> Dict[str, Any]:
    if not state.is_enabled(WorkflowState.Stage.land_use):
        return {}

    explicit_inputs: Dict[str, Any] = {}
    population_value = resolved_usim_inputs.get(USIM_POPULATION_SOURCE_H5)
    if population_value is None:
        population_binding = build_binding_plan(
            step_name="activitysim_postprocess",
            coupler=None,
            artifact_rules=(
                ArtifactBindingRule(
                    semantic_key=USIM_POPULATION_SOURCE_H5,
                    required=USIM_POPULATION_SOURCE_H5 in postprocess_required_keys,
                    allow_coupler=False,
                    allow_fallback=True,
                    preferred_keys=(USIM_POPULATION_SOURCE_H5,),
                    fallback_provider="activitysim_population_source",
                ),
            ),
            restrict_to_inline_rules=True,
            required_keys=(
                (USIM_POPULATION_SOURCE_H5,)
                if USIM_POPULATION_SOURCE_H5 in postprocess_required_keys
                else ()
            ),
            optional_keys=(
                (USIM_POPULATION_SOURCE_H5,)
                if USIM_POPULATION_SOURCE_H5 in postprocess_optional_keys
                else ()
            ),
            settings=settings,
            state=state,
            workspace=workspace,
            year=state.forecast_year,
            surface=None,
        )
        population_inputs = (
            population_binding.inputs if population_binding.inputs is not None else {}
        )
        population_value = population_inputs.get(USIM_POPULATION_SOURCE_H5)
    if population_value is not None:
        explicit_inputs[USIM_POPULATION_SOURCE_H5] = population_value

    # The postprocessor needs two explicit H5 roles when land use is enabled:
    # the forecast-year population source used to build ActivitySim inputs and
    # the current datastore being updated for legacy postprocess semantics.
    current_value = resolved_usim_inputs.get(USIM_DATASTORE_CURRENT_H5)
    if current_value is None:
        current_binding = build_binding_plan(
            step_name="activitysim_postprocess",
            coupler=None,
            artifact_rules=(
                ArtifactBindingRule(
                    semantic_key=USIM_DATASTORE_CURRENT_H5,
                    required=USIM_DATASTORE_CURRENT_H5 in postprocess_required_keys,
                    allow_coupler=False,
                    allow_fallback=True,
                    preferred_keys=(USIM_DATASTORE_CURRENT_H5,),
                    fallback_provider="urbansim_inputs_for_year",
                ),
            ),
            restrict_to_inline_rules=True,
            required_keys=(
                (USIM_DATASTORE_CURRENT_H5,)
                if USIM_DATASTORE_CURRENT_H5 in postprocess_required_keys
                else ()
            ),
            optional_keys=(
                (USIM_DATASTORE_CURRENT_H5,)
                if USIM_DATASTORE_CURRENT_H5 in postprocess_optional_keys
                else ()
            ),
            settings=settings,
            state=state,
            workspace=workspace,
            year=state.year,
            surface=None,
        )
        current_inputs = (
            current_binding.inputs if current_binding.inputs is not None else {}
        )
        current_value = current_inputs.get(USIM_DATASTORE_CURRENT_H5)
    if current_value is not None:
        explicit_inputs[USIM_DATASTORE_CURRENT_H5] = current_value

    return explicit_inputs


def _activitysim_postprocess_role_binding_rules(
    *,
    postprocess_required_keys: tuple[str, ...],
    postprocess_optional_keys: tuple[str, ...],
    land_use_enabled: bool = False,
) -> tuple[ArtifactBindingRule, ...]:
    role_keys = set(postprocess_required_keys) | set(postprocess_optional_keys)
    rules: list[ArtifactBindingRule] = []
    if land_use_enabled:
        for semantic_key in (USIM_POPULATION_SOURCE_H5, USIM_DATASTORE_CURRENT_H5):
            rules.append(
                ArtifactBindingRule(
                    semantic_key=semantic_key,
                    required=semantic_key in postprocess_required_keys,
                    allow_coupler=False,
                    allow_fallback=False,
                    preferred_keys=(semantic_key,),
                )
            )
    if USIM_DATASTORE_BASE_H5 in role_keys:
        rules.append(
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_BASE_H5,
                required=USIM_DATASTORE_BASE_H5 in postprocess_required_keys,
                allow_coupler=False,
                allow_fallback=True,
                preferred_keys=(USIM_DATASTORE_BASE_H5,),
                fallback_provider="activitysim_input_datastore",
            )
        )
    return tuple(rules)


@dataclass
class ActivityDemandPhaseInputs:
    """
    Inputs for one ActivitySim (activity-demand) iteration.

    Parameters
    ----------
    year : int
        Forecast year being simulated.
    iteration : int
        Supply-demand iteration index for the year.
    usim_inputs : Mapping[str, Union[str, os.PathLike]]
        Pre-resolved UrbanSim datastore inputs, if land use already ran or
        fallback inputs were provided.
    """

    year: int
    iteration: int
    usim_inputs: Mapping[str, Union[str, os.PathLike]]


@dataclass
class ActivityDemandPhaseOutputs:
    """
    Outputs from one ActivitySim (activity-demand) iteration.

    Parameters
    ----------
    activity_demand_outputs : Optional[dict[str, Any]]
        Mapping containing ActivitySim outputs needed downstream
        (e.g., households, persons, plans). None if not produced.
    """

    activity_demand_outputs: Optional[Dict[str, Any]]


def _activitysim_exact_rewind_restore(
    workspace: Workspace,
    *,
    year: int,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    metadata = getattr(workspace, "_activitysim_exact_rewind_restore", None)
    if not isinstance(metadata, dict):
        return None
    if metadata.get("year") != year or metadata.get("iteration") != iteration:
        return None
    return metadata


def _seed_exact_rewind_activitysim_preprocess_outputs(
    *,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
) -> ActivitySimPreprocessOutputs:
    input_dir = workspace.get_asim_mutable_data_dir()
    preprocess_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=Path(input_dir),
        land_use_table=Path(input_dir) / "land_use.csv",
        households_table=Path(input_dir) / "households.csv",
        persons_table=Path(input_dir) / "persons.csv",
        omx_skims=(
            (Path(input_dir) / "skims.omx")
            if (Path(input_dir) / "skims.omx").exists()
            else None
        ),
    )
    outputs_holder.activitysim_preprocess = preprocess_outputs
    return preprocess_outputs


def _surface_restart_missing_explicit_roles(
    *,
    coupler: CouplerProtocol,
    resolved_usim_inputs: Mapping[str, Union[str, os.PathLike]],
    surface: Optional["EnabledWorkflowSurface"],
) -> tuple[str, ...]:
    if surface is None:
        return ()
    runtime_surface = surface.step_surface("activitysim_preprocess")
    if runtime_surface is None or not runtime_surface.enabled:
        return ()
    get_value = getattr(coupler, "get", None)
    return tuple(
        key
        for key in (
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
        )
        if bool(
            getattr(
                runtime_surface.input_role_policies.get(key),
                "restart_requires_explicit_before_execution",
                False,
            )
        )
        and key not in resolved_usim_inputs
        and not (callable(get_value) and get_value(key) is not None)
    )


def _seed_postprocess_role_fallbacks_from_coupler(
    *,
    coupler: CouplerProtocol,
    resolved_usim_inputs: Dict[str, Union[str, os.PathLike]],
) -> None:
    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return
    coupler_population = get_value(USIM_POPULATION_SOURCE_H5)
    coupler_current = get_value(USIM_DATASTORE_CURRENT_H5)
    if (
        USIM_DATASTORE_CURRENT_H5 not in resolved_usim_inputs
        and coupler_current is not None
    ):
        resolved_usim_inputs[USIM_DATASTORE_CURRENT_H5] = coupler_current
    elif (
        USIM_DATASTORE_CURRENT_H5 not in resolved_usim_inputs
        and coupler_population is not None
    ):
        resolved_usim_inputs[USIM_DATASTORE_CURRENT_H5] = coupler_population
    if (
        USIM_POPULATION_SOURCE_H5 not in resolved_usim_inputs
        and coupler_population is not None
    ):
        resolved_usim_inputs[USIM_POPULATION_SOURCE_H5] = coupler_population
    elif (
        USIM_POPULATION_SOURCE_H5 not in resolved_usim_inputs
        and coupler_current is not None
    ):
        resolved_usim_inputs[USIM_POPULATION_SOURCE_H5] = coupler_current


def _run_activity_demand_phase(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    inputs: ActivityDemandPhaseInputs,
    outputs_holder: StepOutputsHolder,
    manifest_config: ManifestConfig,
    context: Optional[WorkflowRuntimeContext] = None,
    state: Optional[WorkflowState] = None,
    settings: Optional[PilatesConfig] = None,
    workspace: Optional[Workspace] = None,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> ActivityDemandPhaseOutputs:
    """
    Run ActivitySim for a single supply-demand iteration.

    This executes the ActivitySim preprocess, compile (once per year),
    and run/postprocess steps. It also assembles the required inputs
    from UrbanSim outputs or fallbacks and ensures skims are available
    when resuming after compilation.
    """
    runtime_context = ensure_workflow_runtime_context(
        context=context,
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    settings = runtime_context.settings
    state = runtime_context.state
    workspace = runtime_context.workspace
    population_year = _activitysim_population_year(state)

    formatted_print("ACTIVITY DEMAND MODEL")
    stage_runner = StageRunner(
        stage_name="activity_demand",
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=f"{inputs.year}_iter{inputs.iteration}",
        iteration=inputs.iteration,
        manifest_config=manifest_config,
        run_workflow_fn=run_workflow,
    )
    exact_rewind_restore = _activitysim_exact_rewind_restore(
        workspace,
        year=inputs.year,
        iteration=inputs.iteration,
    )
    runtime_surface = runtime_context.surface
    profile = runtime_surface.profile
    resolved_usim_inputs = dict(inputs.usim_inputs)
    missing_restart_roles = _surface_restart_missing_explicit_roles(
        coupler=coupler,
        resolved_usim_inputs=resolved_usim_inputs,
        surface=runtime_surface,
    )
    if (
        bool(getattr(state, "is_restart_run", False))
        and profile.land_use_enabled
        and missing_restart_roles
    ):
        from .supply_demand_resume import (
            _restore_supply_demand_usim_inputs_for_resume,
        )

        resolved_usim_inputs.update(
            _restore_supply_demand_usim_inputs_for_resume(
                coupler=coupler,
                workspace=workspace,
                state=state,
                settings=settings,
            )
        )
        missing_restart_roles = _surface_restart_missing_explicit_roles(
            coupler=coupler,
            resolved_usim_inputs=resolved_usim_inputs,
            surface=runtime_surface,
        )
    if missing_restart_roles:
        raise RuntimeError(
            "Restart metadata is missing required post-land-use UrbanSim H5 roles "
            f"for ActivitySim: {', '.join(missing_restart_roles)}. "
            "This restart likely predates the explicit population-source H5 role split."
        )

    # ActivitySim runs in three manifest-checkpointed semantic phases:
    # preprocess, primary run, and postprocess. The primary run performs its
    # private local Numba warmup only when it is actually executed.
    if exact_rewind_restore is not None:
        _seed_exact_rewind_activitysim_preprocess_outputs(
            workspace=workspace,
            outputs_holder=outputs_holder,
        )
    else:
        preprocess_explicit_inputs: Optional[Dict[str, Union[str, os.PathLike]]] = None
        if not profile.land_use_enabled:
            population_source = resolved_usim_inputs.get(
                USIM_DATASTORE_BASE_H5
            ) or resolved_usim_inputs.get(USIM_DATASTORE_CURRENT_H5)
            if population_source is not None:
                preprocess_explicit_inputs = {
                    USIM_POPULATION_SOURCE_H5: population_source,
                }

        preprocess_binding = build_binding_plan(
            step_name="activitysim_preprocess",
            coupler=coupler,
            explicit_inputs=preprocess_explicit_inputs,
            fallback_inputs=resolved_usim_inputs,
            settings=settings,
            state=state,
            workspace=workspace,
            year=population_year,
            surface=runtime_surface,
        )

        if preprocess_binding.missing_required:
            raise RuntimeError(
                "ActivitySim preprocess requires a resolved population-source UrbanSim datastore "
                "(population_source, forecast output, current, or base), but none were available."
            )

        stage_runner.run_step(
            stage_name="activity_demand_preprocess",
            step=StepRef(
                name="activitysim_preprocess",
                step_func=make_activitysim_preprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder,
                    surface=runtime_surface,
                ),
                binding=preprocess_binding,
                year=state.forecast_year,
            ),
        )

    def _resolved_existing_zarr_skims_path() -> Optional[str]:
        """Resolve only a published durable/coupler Zarr artifact.

        A local runtime store is an implementation detail and must not change
        the semantic primary request from OMX mode to Zarr mode.
        """
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            zarr_path = artifact_to_existing_path(
                resolve_artifact_from_value(
                    get_value(ZARR_SKIMS),
                    key=ZARR_SKIMS,
                    workspace=workspace,
                ),
                workspace=workspace,
            )
            if zarr_path:
                return zarr_path
        return None

    upstream_preprocess = outputs_holder.activitysim_preprocess
    if upstream_preprocess is None:
        raise RuntimeError("ActivitySim preprocess must complete first")
    asim_run_input_keys = [
        short_name for short_name, _, _ in upstream_preprocess._iter_record_items()
    ]
    asim_run_input_keys = [key for key in asim_run_input_keys if key != ASIM_OMX_SKIMS]
    zarr_path = _resolved_existing_zarr_skims_path()
    skim_mode = "zarr" if zarr_path else "omx"
    if zarr_path:
        logger.info("ActivitySim skim source: reusable Zarr (%s).", zarr_path)
    else:
        logger.info("ActivitySim skim source: OMX conversion; Zarr will be produced.")
    if skim_mode == "zarr":
        asim_run_input_keys.append(ZARR_SKIMS)
    else:
        asim_run_input_keys.append(ASIM_OMX_SKIMS)

    stage_runner.run_step(
        stage_name="activity_demand_run",
        step=StepRef(
            name="activitysim_run",
            step_func=make_activitysim_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
                skim_mode=skim_mode,
            ),
            binding=build_key_only_binding_plan(
                step_name="activitysim_run",
                input_keys=asim_run_input_keys,
                coupler=coupler,
                settings=settings,
                state=state,
                workspace=workspace,
                year=population_year,
                surface=runtime_surface,
            ),
            declared_outputs=[
                *ASIM_REQUIRED_RUN_OUTPUT_KEYS,
                *([ZARR_SKIMS] if skim_mode == "omx" else []),
            ],
            cache_hydration=("outputs-requested" if skim_mode == "omx" else "metadata"),
            validate_cached_outputs="eager",
            year=state.forecast_year,
        ),
        runtime_kwargs_extra=(
            {"skip_numba_warmup": True} if exact_rewind_restore is not None else None
        ),
    )

    if outputs_holder.activitysim_run is None:
        raise RuntimeError("ActivitySim run must complete first")

    postprocess_required_keys: tuple[str, ...] = ()
    postprocess_optional_keys: tuple[str, ...] = ()
    if runtime_surface is not None:
        postprocess_surface = runtime_surface.step_surface("activitysim_postprocess")
        if postprocess_surface is not None:
            postprocess_required_keys = tuple(
                key
                for key in postprocess_surface.required_input_keys
                if key in _ACTIVITYSIM_PILOT_H5_ROLE_KEYS
            )
            postprocess_optional_keys = tuple(
                key
                for key in postprocess_surface.optional_input_keys
                if key in _ACTIVITYSIM_PILOT_H5_ROLE_KEYS
            )
    if profile.land_use_enabled:
        _seed_postprocess_role_fallbacks_from_coupler(
            coupler=coupler,
            resolved_usim_inputs=resolved_usim_inputs,
        )
    postprocess_explicit_inputs = _resolve_activitysim_postprocess_h5_role_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_usim_inputs=resolved_usim_inputs,
        postprocess_required_keys=postprocess_required_keys,
        postprocess_optional_keys=postprocess_optional_keys,
    )
    postprocess_fallback_inputs = {
        key: value
        for key, value in resolved_usim_inputs.items()
        if key not in {USIM_POPULATION_SOURCE_H5, USIM_DATASTORE_CURRENT_H5}
    }
    activitysim_postprocess_binding = build_binding_plan(
        step_name="activitysim_postprocess",
        coupler=coupler,
        explicit_inputs=postprocess_explicit_inputs or None,
        fallback_inputs=postprocess_fallback_inputs or None,
        artifact_rules=_activitysim_postprocess_role_binding_rules(
            postprocess_required_keys=postprocess_required_keys,
            postprocess_optional_keys=postprocess_optional_keys,
            land_use_enabled=profile.land_use_enabled,
        ),
        restrict_to_inline_rules=True,
        required_keys=postprocess_required_keys,
        optional_keys=postprocess_optional_keys,
        settings=settings,
        state=state,
        workspace=workspace,
        year=state.forecast_year,
        surface=runtime_surface,
    )
    if activitysim_postprocess_binding.missing_required:
        get_value = getattr(coupler, "get", None)
        coupler_current = (
            get_value(USIM_DATASTORE_CURRENT_H5) if callable(get_value) else None
        )
        coupler_population = (
            get_value(USIM_POPULATION_SOURCE_H5) if callable(get_value) else None
        )
        raise RuntimeError(
            "ActivitySim postprocess could not resolve its required UrbanSim H5 roles: "
            f"{', '.join(activitysim_postprocess_binding.missing_required)}; "
            f"resolved_usim_inputs_keys={sorted(resolved_usim_inputs.keys())}; "
            f"resolved_current={resolved_usim_inputs.get(USIM_DATASTORE_CURRENT_H5)!r}; "
            f"resolved_population={resolved_usim_inputs.get(USIM_POPULATION_SOURCE_H5)!r}; "
            f"coupler_current={coupler_current!r}; "
            f"coupler_population={coupler_population!r}"
        )

    stage_runner.run_step(
        stage_name="activity_demand_postprocess",
        step=StepRef(
            name="activitysim_postprocess",
            step_func=make_activitysim_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
                surface=runtime_surface,
            ),
            binding=activitysim_postprocess_binding,
            year=state.forecast_year,
        ),
    )

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.activity_demand,
    )

    postprocess_outputs = outputs_holder.activitysim_postprocess
    activity_demand_outputs = (
        step_output_handoff_mapping(postprocess_outputs, coupler=coupler)
        if postprocess_outputs is not None
        else None
    )
    if exact_rewind_restore is not None:
        try:
            delattr(workspace, "_activitysim_exact_rewind_restore")
        except Exception:
            logger.debug(
                "Failed clearing ActivitySim exact rewind metadata after use",
                exc_info=True,
            )

    return ActivityDemandPhaseOutputs(activity_demand_outputs=activity_demand_outputs)
