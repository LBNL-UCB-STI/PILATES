from __future__ import annotations

from typing import Any, Callable, Dict

from pilates.config.models import PilatesConfig
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
    RecordStore,
    StepOutputsHolder,
    WorkflowState,
    _beam_log_facet_meta,
    _beam_postprocess_split_facet_meta,
    _execute_beam_full_skim,
    _execute_beam_postprocess,
    _execute_beam_preprocess,
    _execute_beam_run,
    _log_beam_r5_osm_input,
    _log_step_records,
    _make_generic_step_function,
    cr,
    find_last_run_output_plans,
    log_and_set_input,
    log_and_set_output,
    log_output_only,
    update_coupler_from_beam_outputs,
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
                "households_beam_in",
                "persons_beam_in",
                "plans_beam_in",
            },
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
            _log_beam_r5_osm_input(
                tracker=tracker,
                settings=settings,
                workspace=workspace,
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
        def _beam_run_extra_meta(
            short_name: str,
            _path: str,
            _description: str,
        ) -> Dict[str, Any]:
            meta: Dict[str, Any] = {}
            facet_meta = _beam_log_facet_meta(short_name)
            if facet_meta:
                meta.update(facet_meta)
            if short_name.startswith("beam_network_final"):
                meta.update(
                    {
                        "profile_file_schema": "if_changed",
                        "reuse_if_unchanged": True,
                        "reuse_scope": "any_uri",
                    }
                )
                try:
                    from pilates.database.schema.beam_schema import BeamNetworkFinal
                except Exception:
                    BeamNetworkFinal = None
                if BeamNetworkFinal is not None:
                    meta["schema"] = BeamNetworkFinal
            return meta

        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=log_output_only,
            extra_meta_fn=_beam_run_extra_meta,
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

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="beam_full",
        phase="skim",
        outputs_class=BeamFullSkimOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "beam_full_skim", state, WorkflowState.Stage.traffic_assignment
        ),
        component_executor=_execute_beam_full_skim,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "beam_full_skim", outputs
        ),
        output_logger=_log_outputs,
    )
