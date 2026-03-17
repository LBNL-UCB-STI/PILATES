from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Type, TypeVar

from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.utils.coupler_helpers import artifact_to_existing_path
from pilates.workflows.artifact_key_migrations import resolve_artifact_key
from pilates.workflows.artifact_keys import (
    BEAM_CONFIG_FILE,
    BEAM_NETWORK_FINAL,
    LINKSTATS,
    LINKSTATS_WARMSTART,
)
from pilates.workflows.outputs_base import StepOutputsBase, ValidationContext
from pilates.workspace import Workspace

# Model-specific step factories for BEAM.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PLANS_OUT,
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
    CouplerProtocol,
    StepOutputsHolder,
    WorkflowState,
    _beam_log_facet_meta,
    _beam_postprocess_split_facet_meta,
    _decorate_step_with_consist,
    _log_beam_r5_osm_input,
    _log_step_records,
    _schema_outputs_from_class,
    _upstream_outputs_view,
    cr,
    find_last_run_output_plans,
    log_and_set_input,
    log_and_set_output,
    log_input_only,
    log_output_only,
)

StepOutputsT = TypeVar("StepOutputsT", bound=StepOutputsBase)


def _primary_beam_config_path(
    settings: PilatesConfig,
    workspace: Workspace,
) -> Path:
    return (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.beam.config
    )


def _require_primary_beam_config(
    settings: PilatesConfig,
    workspace: Workspace,
) -> Path:
    config_path = _primary_beam_config_path(settings, workspace)
    if not config_path.exists():
        raise FileNotFoundError(
            "BEAM primary config file is missing: "
            f"{config_path}. Expected from settings.beam.config="
            f"{settings.beam.config!r} under the mutable BEAM input dir for "
            f"region {settings.run.region!r}."
        )
    return config_path


def _is_beam_sub_iteration_key(short_name: Optional[str]) -> bool:
    return bool(
        short_name and ("_sub" in short_name or "__beam_sub_iter" in short_name)
    )


def _beam_linkstats_publication_meta(
    short_name: Optional[str],
    *,
    family: str,
) -> Dict[str, Any]:
    if not short_name:
        return {}
    for prefix in ("linkstats_parquet_", "linkstats_"):
        if not short_name.startswith(prefix):
            continue
        tail = short_name[len(prefix) :]
        parts = tail.split("_")
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            iteration = int(parts[1])
        except ValueError:
            continue
        facet: Dict[str, Any] = {
            "artifact_family": family,
            "year": year,
            "iteration": iteration,
        }
        if len(parts) > 2 and parts[2].startswith("sub"):
            try:
                facet["beam_sub_iteration"] = int(parts[2][3:])
            except ValueError:
                continue
        return {
            "facet": facet,
            "facet_schema_version": "v1",
            "facet_index": True,
        }
    return {}


def _publish_beam_run_outputs(
    *,
    outputs: BeamRunOutputs,
    coupler: CouplerProtocol,
) -> None:
    promoted_linkstats = outputs.promoted_linkstats_for_publication()
    if promoted_linkstats is not None:
        source_key, path = promoted_linkstats
        linkstats_meta = _beam_linkstats_publication_meta(
            source_key,
            family="linkstats",
        )
        log_and_set_output(
            key=LINKSTATS,
            path=str(path),
            description="BEAM linkstats output for downstream runs",
            coupler=coupler,
            profile_file_schema=True,
            **linkstats_meta,
        )
        log_and_set_output(
            key=LINKSTATS_WARMSTART,
            path=str(path),
            description="BEAM warm-start linkstats for downstream runs",
            coupler=coupler,
            profile_file_schema=True,
            **linkstats_meta,
        )

    for short_name, path in outputs.iter_linkstats_parquet_outputs():
        linkstats_meta = _beam_log_facet_meta(short_name)
        if _is_beam_sub_iteration_key(short_name):
            log_output_only(
                key=short_name,
                path=str(path),
                description="BEAM linkstats parquet output for downstream runs",
                profile_file_schema=True,
                **linkstats_meta,
            )
            continue
        log_and_set_output(
            key=short_name,
            path=str(path),
            description="BEAM linkstats parquet output for downstream runs",
            coupler=coupler,
            profile_file_schema=True,
            **linkstats_meta,
        )

    for short_name, path in outputs.iter_unmodified_phys_sim_outputs():
        record_meta = _beam_log_facet_meta(short_name)
        if _is_beam_sub_iteration_key(short_name):
            log_output_only(
                key=short_name,
                path=str(path),
                description=(
                    "BEAM unmodified linkstats parquet output for phys sim "
                    "sub-iteration"
                ),
                profile_file_schema=True,
                **record_meta,
            )
            continue
        log_and_set_output(
            key=short_name,
            path=str(path),
            description=(
                "BEAM unmodified linkstats parquet output for phys sim sub-iteration"
            ),
            coupler=coupler,
            profile_file_schema=True,
            **record_meta,
        )

    promoted_plans = outputs.promoted_plans_for_publication()
    if promoted_plans is not None:
        _, path = promoted_plans
        log_and_set_output(
            key=BEAM_PLANS_OUT,
            path=str(path),
            description="BEAM plans output for downstream runs",
            coupler=coupler,
            profile_file_schema=True,
        )


def _execute_beam_preprocess(
    preprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    activity_demand_outputs: Optional[Dict[str, Any]] = None,
    previous_beam_outputs: Optional[Dict[str, Any]] = None,
    beam_preprocess_inputs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> BeamPreprocessOutputs:
    return preprocessor.preprocess(
        workspace,
        activity_demand_outputs=activity_demand_outputs,
        previous_beam_outputs=previous_beam_outputs,
        beam_preprocess_inputs=beam_preprocess_inputs,
    )


def _execute_beam_run(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    extra_inputs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> BeamRunOutputs:
    upstream = outputs_holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")
    if not isinstance(upstream, BeamPreprocessOutputs):
        raise TypeError("beam_run requires BeamPreprocessOutputs from beam_preprocess")
    return runner.run(
        upstream,
        workspace,
        extra_inputs=extra_inputs,
    )


def _execute_beam_postprocess(
    postprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **_: Any,
) -> BeamPostprocessOutputs:
    upstream = outputs_holder.beam_run
    if upstream is None:
        raise RuntimeError("BEAM run must complete first")
    if not isinstance(upstream, BeamRunOutputs):
        raise TypeError("beam_postprocess requires BeamRunOutputs from beam_run")
    return postprocessor.postprocess(upstream, workspace)


def _execute_beam_full_skim(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    previous_beam_outputs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> BeamFullSkimOutputs:
    upstream = outputs_holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")
    if not isinstance(upstream, BeamPreprocessOutputs):
        raise TypeError(
            "beam_full_skim requires BeamPreprocessOutputs from beam_preprocess"
        )
    return runner.run(
        upstream,
        workspace,
        previous_beam_outputs=previous_beam_outputs,
    )


def _make_beam_step_function(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    model_name: str,
    phase: str,
    outputs_class: Type[StepOutputsT],
    component_getter: Callable[[ModelFactory, WorkflowState], Any],
    component_executor: Callable[..., StepOutputsT],
    outputs_holder_setter: Callable[[StepOutputsHolder, StepOutputsT], None],
    input_logger: Optional[Callable[..., Dict[str, Any]]] = None,
    output_logger: Optional[Callable[..., None]] = None,
    output_recoverer: Optional[Callable[..., Optional[StepOutputsT]]] = None,
) -> Callable[..., None]:
    @cr.require_runtime_kwargs("settings", "state", "workspace")
    def _step_func(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        **kwargs: Any,
    ) -> None:
        factory = ModelFactory()
        component = component_getter(factory, state)

        extra_kwargs = (
            dict(input_logger(settings, state, workspace, outputs_holder) or {})
            if input_logger is not None
            else {}
        )
        step_outputs = component_executor(
            component,
            workspace,
            outputs_holder,
            coupler=coupler,
            context=f"{model_name}_{phase}",
            **extra_kwargs,
            **kwargs,
        )
        if not isinstance(step_outputs, outputs_class):
            raise TypeError(
                f"{model_name}_{phase} must return {outputs_class.__name__}, "
                f"got {type(step_outputs).__name__}"
            )
        step_outputs.validate(
            context=ValidationContext(
                settings=settings,
                state=state,
                workspace=workspace,
                step_name=f"{model_name}_{phase}",
                upstream_outputs=_upstream_outputs_view(
                    outputs_holder,
                    current_step_name=f"{model_name}_{phase}",
                ),
            )
        )
        outputs_holder_setter(outputs_holder, step_outputs)

        if output_logger is not None:
            output_logger(step_outputs, settings, state, workspace, outputs_holder)

    if output_logger is not None:
        setattr(
            _step_func,
            "__pilates_output_replayer__",
            lambda outputs, settings, state, workspace, holder: output_logger(
                outputs, settings, state, workspace, holder
            ),
        )
    if output_recoverer is not None:
        setattr(_step_func, "__pilates_output_recoverer__", output_recoverer)

    step_model = f"{model_name}_{phase}"
    return _decorate_step_with_consist(
        step_func=_step_func,
        step_model=step_model,
        description=f"{step_model} workflow step",
        schema_outputs=_schema_outputs_from_class(outputs_class),
        outputs=list(outputs_class.declared_output_keys()) or None,
        tags=[model_name, phase],
    )


def _resolve_cached_run_outputs(run_id: Optional[str]) -> Dict[str, Any]:
    if not run_id:
        return {}
    tracker = cr.current_tracker()
    if tracker is None:
        return {}
    get_run_outputs = getattr(tracker, "get_run_outputs", None)
    if not callable(get_run_outputs):
        return {}
    try:
        run_outputs = get_run_outputs(run_id) or {}
    except Exception:
        return {}
    resolved: Dict[str, Any] = {}
    for raw_key, value in run_outputs.items():
        if value is None:
            continue
        raw_key_str = str(raw_key)
        local_key = raw_key_str.split("/", 1)[-1]
        resolved[resolve_artifact_key(local_key)] = value
    return resolved


def _recovered_cached_paths(
    *,
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
    workspace: Workspace,
) -> Dict[str, Path]:
    merged: Dict[str, Any] = {}
    if cached_outputs:
        for raw_key, value in cached_outputs.items():
            if value is None:
                continue
            raw_key_str = str(raw_key)
            local_key = raw_key_str.split("/", 1)[-1]
            merged[resolve_artifact_key(local_key)] = value
    for key, value in _resolve_cached_run_outputs(run_id).items():
        merged[key] = value
    recovered: Dict[str, Path] = {}
    for key, value in merged.items():
        path = artifact_to_existing_path(
            value,
            workspace=workspace,
            materialize_from_archive=True,
        )
        if path is not None:
            recovered[key] = Path(path)
    return recovered


def _recover_beam_run_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[BeamRunOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    if not recovered_paths:
        return None
    return BeamRunOutputs(
        beam_output_dir=Path(workspace.get_beam_output_dir()),
        raw_outputs=recovered_paths,
    )


def _recover_beam_postprocess_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[BeamPostprocessOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    if not recovered_paths:
        return None
    return BeamPostprocessOutputs(
        zarr_skims=recovered_paths.get("zarr_skims"),
        final_skims_omx=recovered_paths.get("final_skims_omx"),
        split_events={
            key: path
            for key, path in recovered_paths.items()
            if key.startswith("events_parquet_") and "_type_" in key
        },
        split_event_links={
            key: path
            for key, path in recovered_paths.items()
            if key.startswith("path_traversal_links_")
        },
    )


def _recover_beam_full_skim_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[BeamFullSkimOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    full_skims = recovered_paths.get("beam_full_skims")
    if full_skims is None:
        return None
    return BeamFullSkimOutputs(full_skims=full_skims)


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
        _require_primary_beam_config(settings, workspace)
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
        _log_step_records(
            record_items=(
                (
                    key,
                    path,
                    f"BEAM prepared input {key} for year {state.year}, iter {state.iteration}",
                )
                for key, path in outputs.prepared_inputs.items()
            ),
            log_fn=lambda key, path, description, **meta: log_and_set_output(
                key=key,
                path=path,
                description=description,
                coupler=coupler,
                **meta,
            ),
            profile_schema_keys={
                "plans_beam_in",
                "vehicles_beam_in",
            },
        )

    return _make_beam_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="beam",
        phase="preprocess",
        outputs_class=BeamPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor("beam", state),
        component_executor=lambda component, workspace, outputs_holder, **kwargs: (
            _execute_beam_preprocess(
                component,
                workspace,
                outputs_holder,
                **kwargs,
            )
        ),
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

        config_path = _require_primary_beam_config(settings, workspace)
        log_input_only(
            key=BEAM_CONFIG_FILE,
            path=str(config_path),
            description="BEAM config file consumed by the BEAM run",
        )

        tracker = cr.current_tracker()
        if tracker is not None:
            _log_beam_r5_osm_input(
                tracker=tracker,
                settings=settings,
                workspace=workspace,
            )
        for short_name, path, description in upstream._iter_record_items():
            log_input_only(
                key=short_name,
                path=str(path),
                description=description,
            )

        output_root = Path(workspace.get_beam_output_dir()) / settings.run.region
        plans_path, experienced_path = find_last_run_output_plans(output_root, "year-")
        if plans_path is not None and plans_path.exists():
            if plans_path.name == "output_plans.xml.gz":
                plans_key = BEAM_OUTPUT_PLANS_XML
            else:
                plans_key = BEAM_PLANS_OUT
            log_input_only(
                key=plans_key,
                path=str(plans_path),
                description=(
                    "BEAM warm-start plans (selected by BEAM from previous outputs)"
                ),
            )
        if experienced_path is not None and experienced_path.exists():
            if experienced_path.name == "output_experienced_plans.xml.gz":
                experienced_key = BEAM_OUTPUT_EXPERIENCED_PLANS_XML
            else:
                experienced_key = BEAM_EXPERIENCED_PLANS_XML
            log_input_only(
                key=experienced_key,
                path=str(experienced_path),
                description=(
                    "BEAM warm-start experienced plans (selected by BEAM from previous outputs)"
                ),
            )
        return {}

    def _log_outputs(
        outputs: BeamRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        def _beam_run_extra_meta(
            short_name: str,
            _path: str,
            _description: str,
        ) -> Dict[str, Any]:
            meta: Dict[str, Any] = {}
            facet_meta = _beam_log_facet_meta(short_name)
            if facet_meta:
                meta.update(facet_meta)
            if short_name == BEAM_NETWORK_FINAL:
                meta.update(
                    {
                        "profile_file_schema": "if_changed",
                        "reuse_if_unchanged": True,
                        "reuse_scope": "any_uri",
                    }
                )
                beam_network_schema: Any = None
                try:
                    from pilates.database.schema.beam_schema import BeamNetworkFinal
                except Exception:
                    beam_network_schema = None
                else:
                    beam_network_schema = BeamNetworkFinal
                if beam_network_schema is not None:
                    meta["schema"] = beam_network_schema
            return meta

        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=log_output_only,
            extra_meta_fn=_beam_run_extra_meta,
        )

    return _make_beam_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="beam",
        phase="run",
        outputs_class=BeamRunOutputs,
        component_getter=lambda factory, state: factory.get_runner("beam", state),
        component_executor=lambda component, workspace, outputs_holder, **kwargs: (
            _execute_beam_run(
                component,
                workspace,
                outputs_holder,
                **kwargs,
            )
        ),
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "beam_run", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
        output_recoverer=_recover_beam_run_outputs,
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
            facet_meta = _beam_postprocess_split_facet_meta(short_name)
            log_output_only(
                key=short_name,
                path=str(path),
                description=f"BEAM events parquet split ({short_name})",
                profile_file_schema=True,
                **facet_meta,
            )
        for short_name, path in outputs.split_event_links.items():
            facet_meta = _beam_postprocess_split_facet_meta(short_name)
            log_output_only(
                key=short_name,
                path=str(path),
                description=f"BEAM events link table ({short_name})",
                profile_file_schema=True,
                **facet_meta,
            )
        upstream = holder.beam_run
        if upstream is None:
            return
        _publish_beam_run_outputs(outputs=upstream, coupler=coupler)

    return _make_beam_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="beam",
        phase="postprocess",
        outputs_class=BeamPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "beam", state
        ),
        component_executor=lambda component, workspace, outputs_holder, **kwargs: (
            _execute_beam_postprocess(
                component,
                workspace,
                outputs_holder,
                **kwargs,
            )
        ),
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "beam_postprocess", outputs
        ),
        output_logger=_log_outputs,
        output_recoverer=_recover_beam_postprocess_outputs,
    )


def make_beam_full_skim_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM full-skim step function.

    This step runs BEAM's FullSkimsCreatorApp to produce background skims
    from prepared BEAM inputs and optional warm-start linkstats.
    """

    def _log_outputs(
        outputs: BeamFullSkimOutputs,
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

    return _make_beam_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="beam_full",
        phase="skim",
        outputs_class=BeamFullSkimOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "beam_full_skim", state
        ),
        component_executor=lambda component, workspace, outputs_holder, **kwargs: (
            _execute_beam_full_skim(
                component,
                workspace,
                outputs_holder,
                **kwargs,
            )
        ),
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "beam_full_skim", outputs
        ),
        output_logger=_log_outputs,
        output_recoverer=_recover_beam_full_skim_outputs,
    )
