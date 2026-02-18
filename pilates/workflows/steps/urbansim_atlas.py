from __future__ import annotations

from typing import Any, Callable, Dict

from pilates.config.models import PilatesConfig
from pilates.workspace import Workspace

# Model-specific step factories for UrbanSim and ATLAS.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_H5,
    USIM_INPUT_ARCHIVE_PREFIX,
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
    CouplerProtocol,
    StepOutputsHolder,
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
    WorkflowState,
    _atlas_artifact_facet_meta,
    _execute_atlas_postprocess,
    _execute_atlas_run,
    _execute_preprocess,
    _execute_urbansim_postprocess,
    _execute_urbansim_run,
    _log_step_records,
    _make_generic_step_function,
    _urbansim_output_facet_meta,
    log_and_set_output,
    log_input_only,
    log_output_only,
    logger,
    warm_start_activities,
)

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
                **_urbansim_output_facet_meta(
                    short_name, forecast_year=state.forecast_year
                ),
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
                key=USIM_DATASTORE_BASE_H5,
                path=str(usim_input_path),
                description="UrbanSim base datastore for preprocessing",
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_BASE_H5, forecast_year=state.forecast_year
                ),
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
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=state.forecast_year
                ),
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
                    **_urbansim_output_facet_meta(
                        short_name, forecast_year=state.forecast_year
                    ),
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
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=state.forecast_year
                ),
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
    def _log_outputs(
        outputs: AtlasPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        run_scenario = getattr(getattr(settings, "atlas", None), "scenario", None)
        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=log_output_only,
            profile_schema_suffixes=(".csv", ".parquet"),
            extra_meta_fn=lambda key, _path, _description: _atlas_artifact_facet_meta(
                key,
                run_scenario=run_scenario,
                forecast_year=state.forecast_year,
                artifact_family="atlas_preprocess_output",
            ),
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
        run_scenario = getattr(getattr(settings, "atlas", None), "scenario", None)
        _log_step_records(
            record_items=upstream._iter_record_items(),
            log_fn=log_input_only,
            profile_schema_suffixes=(".csv", ".parquet"),
            extra_meta_fn=lambda key, _path, _description: _atlas_artifact_facet_meta(
                key,
                run_scenario=run_scenario,
                forecast_year=state.forecast_year,
                artifact_family="atlas_run_input",
            ),
        )
        return {}

    def _log_outputs(
        outputs: AtlasRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=log_output_only,
            profile_schema_suffixes=(".csv", ".parquet"),
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
        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=log_output_only,
            profile_schema_suffixes=(".csv", ".parquet"),
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
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=state.forecast_year
                ),
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
