from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from consist import (
    ExecutionOptions,
    StepIdentity,
    define_step,
)

from pilates.config.models import PilatesConfig
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs as NativeUrbanSimPostprocessOutputs,
)
from pilates.urbansim.outputs import UrbanSimRunOutputs as NativeUrbanSimRunOutputs
from pilates.urbansim.postprocessor import UrbansimPostprocessor
from pilates.urbansim.preprocessor import stage_urbansim_run_workspace
from pilates.urbansim.runner import UrbanSimLaunchContext, UrbansimRunner
from pilates.utils.consist_runtime import require_runtime_kwargs
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.usim_h5 import create_usim_population_source_snapshot
from pilates.workflows.artifact_keys import (
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_INPUT_MERGED_PREFIX,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.input_authority import requires_prior_beam_skim_handoff
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_consist_meta import consist_step_meta
from pilates.workflows.step_definition import (
    ConfigContract,
    InputContract,
    StepDefinition,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState

from .shared import (
    _execution_options_with_launch_context,
    _native_contract_output_paths,
    _native_execution_options,
    _persisted_output_path,
    _resolve_native_inputs,
    _strict_requested_output_cache_options,
    _validation_context,
)


# Native Consist step definitions.


def _urbansim_run_execution_options(
    *,
    resolved_inputs: ResolvedStepInputs | None = None,
    **_: Any,
) -> ExecutionOptions:
    """Pass the resolver-owned UrbanSim launch context through runtime kwargs."""

    return _execution_options_with_launch_context(
        resolved_inputs=resolved_inputs,
        step_name="urbansim_run",
        context_key="urbansim_launch_context",
        expected_type=UrbanSimLaunchContext,
    )


def _urbansim_run_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    return UrbansimRunner.expected_outputs(settings, state, workspace)


def _urbansim_postprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    paths = dict(UrbansimPostprocessor.expected_outputs(settings, state, workspace))
    forecast_year = state.forecast_year
    mutable_data_dir = Path(workspace.get_usim_mutable_data_dir())
    input_name = settings.urbansim.input_file_template.format(
        region_id=settings.urbansim.region_mappings["region_to_region_id"][
            settings.run.region
        ]
    )
    merged_path = mutable_data_dir / input_name
    paths[f"{USIM_INPUT_MERGED_PREFIX}{forecast_year}"] = merged_path
    paths[f"{USIM_INPUT_ARCHIVE_PREFIX}{forecast_year}"] = mutable_data_dir / (
        f"input_data_for_{forecast_year}_outputs.h5"
    )
    paths[USIM_POPULATION_SOURCE_H5] = _urbansim_population_source_path(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return paths


def _urbansim_population_source_path(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> Path:
    """Return the declared immutable snapshot destination for this forecast year."""

    forecast_output = Path(workspace.get_usim_mutable_data_dir()) / (
        settings.urbansim.output_file_template.format(year=state.forecast_year)
    )
    return forecast_output.with_name(
        f"{forecast_output.stem}_population_source{forecast_output.suffix}"
    )


def _resolve_urbansim_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    requires_beam_skim = requires_prior_beam_skim_handoff(
        settings=settings,
        state=state,
    )
    region = settings.run.region
    region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
    mutable_data_dir = Path(workspace.get_usim_mutable_data_dir())
    output_datastore = Path(
        _urbansim_run_native_output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
        )[USIM_DATASTORE_H5]
    )
    return _resolve_native_inputs(
        step_name="urbansim_run",
        function=_native_urbansim_run,
        required_roles=(
            USIM_DATASTORE_H5,
            *((FINAL_SKIMS_OMX,) if requires_beam_skim else ()),
        ),
        optional_roles=(() if requires_beam_skim else (FINAL_SKIMS_OMX,)),
        logical_destinations={
            USIM_DATASTORE_H5: mutable_data_dir
            / settings.urbansim.input_file_template.format(region_id=region_id),
            FINAL_SKIMS_OMX: mutable_data_dir / "final_skims.omx",
        },
        coupler=coupler,
        metadata={
            "urbansim_launch_context": UrbanSimLaunchContext(
                mutable_data_dir=mutable_data_dir,
                output_datastore=output_datastore,
            )
        },
    )


def _resolve_urbansim_postprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    step_identity: StepIdentity | None = None,
) -> ResolvedStepInputs:
    return _resolve_native_inputs(
        step_name="urbansim_postprocess",
        function=_native_urbansim_postprocess,
        required_roles=(USIM_DATASTORE_H5,),
        optional_roles=(),
        logical_destinations={
            USIM_DATASTORE_H5: Path(
                _urbansim_run_native_output_paths(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                )[USIM_DATASTORE_H5]
            )
        },
        coupler=coupler,
        step_identity=step_identity,
    )


_URBANSIM_RUN_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="static model data is identity-only rather than resolver-bound",
    config_contract=ConfigContract.payload(),
)
_URBANSIM_POSTPROCESS_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="previous mutable UrbanSim input H5 remains a model-visible read",
    config_contract=ConfigContract.payload(),
)


@define_step(
    model="urbansim_run",
    name_template="urbansim_run__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_DATASTORE_H5: None},
    optional_input_keys=(FINAL_SKIMS_OMX,),
    schema_outputs=[USIM_DATASTORE_H5, USIM_FORECAST_OUTPUT],
    output_paths=_native_contract_output_paths(_urbansim_run_native_output_paths),
    input_binding="paths",
    **consist_step_meta("urbansim_run", input_contract=_URBANSIM_RUN_INPUT_CONTRACT),
)
@require_runtime_kwargs("settings", "state", "workspace", "urbansim_launch_context")
def _native_urbansim_run(
    usim_datastore_h5: Path,
    final_skims_omx: Path | None = None,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    urbansim_launch_context: UrbanSimLaunchContext,
) -> None:
    inputs = stage_urbansim_run_workspace(
        settings=settings,
        state=state,
        workspace=workspace,
        mutable_data_dir=urbansim_launch_context.mutable_data_dir,
        usim_datastore_h5=usim_datastore_h5,
        final_skims_omx=final_skims_omx,
    )
    UrbansimRunner("urbansim", state).run(
        inputs,
        launch_context=urbansim_launch_context,
    )


@define_step(
    model="urbansim_postprocess",
    name_template="urbansim_postprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_DATASTORE_H5: None},
    schema_outputs=[USIM_DATASTORE_H5, USIM_POPULATION_SOURCE_H5],
    output_paths=_native_contract_output_paths(
        _urbansim_postprocess_native_output_paths
    ),
    input_binding="paths",
    **consist_step_meta(
        "urbansim_postprocess", input_contract=_URBANSIM_POSTPROCESS_INPUT_CONTRACT
    ),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_urbansim_postprocess(
    usim_datastore_h5: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    UrbansimPostprocessor("urbansim", state).postprocess(
        NativeUrbanSimRunOutputs(usim_datastore_h5=usim_datastore_h5),
        workspace,
    )
    create_usim_population_source_snapshot(
        source_h5=usim_datastore_h5,
        destination_h5=_urbansim_population_source_path(
            settings=settings,
            state=state,
            workspace=workspace,
        ),
        year=state.forecast_year,
    )


def _project_urbansim_run(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> NativeUrbanSimRunOutputs:
    declared_outputs = _urbansim_run_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    datastore = _persisted_output_path(
        outputs,
        step_name="urbansim_run",
        key=USIM_DATASTORE_H5,
        declared_outputs=declared_outputs,
        workspace=workspace,
    )
    projected = NativeUrbanSimRunOutputs(
        usim_datastore_h5=datastore,
        raw_outputs={USIM_FORECAST_OUTPUT: datastore},
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="urbansim_run",
        )
    )
    return projected


def _project_urbansim_postprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> NativeUrbanSimPostprocessOutputs:
    declared_outputs = _urbansim_postprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    datastore = _persisted_output_path(
        outputs,
        step_name="urbansim_postprocess",
        key=USIM_DATASTORE_H5,
        declared_outputs=declared_outputs,
        workspace=workspace,
    )
    processed_outputs = {
        key: _persisted_output_path(
            outputs,
            step_name="urbansim_postprocess",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in outputs
        if key != USIM_DATASTORE_H5
    }
    projected = NativeUrbanSimPostprocessOutputs(
        usim_datastore_h5=datastore,
        processed_outputs=processed_outputs,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="urbansim_postprocess",
        )
    )
    return projected


URBANSIM_RUN = StepDefinition(
    name="urbansim_run",
    function=_native_urbansim_run,
    resolve_inputs=_resolve_urbansim_run_inputs,
    project_outputs=_project_urbansim_run,
    input_contract=_URBANSIM_RUN_INPUT_CONTRACT,
    output_paths=_urbansim_run_native_output_paths,
    execution_options=_urbansim_run_execution_options,
    cache_options=_strict_requested_output_cache_options,
    optional_output_keys=(USIM_FORECAST_OUTPUT,),
)

URBANSIM_POSTPROCESS = StepDefinition(
    name="urbansim_postprocess",
    function=_native_urbansim_postprocess,
    resolve_inputs=_resolve_urbansim_postprocess_inputs,
    project_outputs=_project_urbansim_postprocess,
    input_contract=_URBANSIM_POSTPROCESS_INPUT_CONTRACT,
    output_paths=_urbansim_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
    preflight_identity=True,
)

# Canonical registry members use the workflow step names. Uppercase aliases keep
# the definitions convenient to import in focused model tests.
urbansim_run = URBANSIM_RUN
urbansim_postprocess = URBANSIM_POSTPROCESS
