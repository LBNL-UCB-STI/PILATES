from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from consist import (
    BindingResult,
    ExecutionOptions,
    OutputSet,
    define_step,
)

from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
)
from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasLaunchContext, AtlasRunner
from pilates.config.models import PilatesConfig
from pilates.urbansim.postprocessor import get_usim_datastore_fname
from pilates.utils.consist_runtime import require_runtime_kwargs
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.usim_h5 import create_usim_population_source_snapshot
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    FINAL_SKIMS_OMX,
    OMX_SKIMS,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_H5_UPDATED,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.binding import ArtifactBindingRule
from pilates.workflows.coupler_namespace import resolve_coupler_value
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
from .urbansim import (
    _urbansim_population_source_path,
    _urbansim_run_native_output_paths,
)


# Native Consist step definitions.

_ATLAS_HOUSEHOLDV_CSV_INPUT = "atlas_householdv_csv"
_ATLAS_VEHICLES_CSV_INPUT = "atlas_vehicles_csv"


def _atlas_run_execution_options(
    *,
    resolved_inputs: ResolvedStepInputs | None = None,
    **_: Any,
) -> ExecutionOptions:
    """Pass the resolver-owned ATLAS launch context through runtime kwargs."""

    return _execution_options_with_launch_context(
        resolved_inputs=resolved_inputs,
        step_name="atlas_run",
        context_key="atlas_launch_context",
        expected_type=AtlasLaunchContext,
    )


def _atlas_preprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    del resolved_inputs
    return dict(
        _atlas_preprocess_prepared_input_paths(
            settings=settings,
            state=state,
            workspace=workspace,
        )
    )


def _atlas_preprocess_prepared_input_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> Mapping[str, Path]:
    """Project the known prepared CSV members from the declared year root."""
    root = Path(workspace.get_atlas_mutable_input_dir()) / f"year{state.forecast_year}"
    paths: dict[str, Path] = {
        "atlas_households_csv": root / "households.csv",
        "atlas_blocks_csv": root / "blocks.csv",
        "atlas_persons_csv": root / "persons.csv",
        "atlas_residential_csv": root / "residential.csv",
        "atlas_jobs_csv": root / "jobs.csv",
    }
    if state.year > state.start_year:
        paths["atlas_grave_csv"] = root / "grave.csv"
    if settings.atlas.beamac > 0:
        paths["atlas_accessibility_csv"] = (
            root / f"accessibility_{state.forecast_year}_tract.csv"
        )
    return paths


def _atlas_run_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    output_dir = Path(workspace.get_atlas_output_dir())
    forecast_year = state.forecast_year
    return {
        f"householdv_{forecast_year}": output_dir / f"householdv_{forecast_year}.csv",
        f"vehicles_{forecast_year}": output_dir / f"vehicles_{forecast_year}.csv",
    }


def _atlas_run_output_sets(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Mapping[str, OutputSet]:
    """Declare the ATLAS continuation surface exactly."""

    del settings, resolved_inputs
    forecast_year = state.forecast_year
    continuation_members = ("vehicles_output.RData", "households_output.RData")
    return {
        f"atlas_continuation_{forecast_year}": OutputSet(
            root=Path(workspace.get_atlas_mutable_input_dir()) / f"year{forecast_year}",
            include=continuation_members,
            expected_members=continuation_members,
        ),
    }


def _atlas_postprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    outputs = AtlasPostprocessor.expected_outputs(settings, state, workspace)
    # ATLAS mutates the exact datastore supplied to the callable. Its published
    # ActivitySim source is a distinct post-mutation snapshot, so it can carry
    # exact-year HDF5 aliases without changing the mutable UrbanSim handoff.
    outputs[USIM_POPULATION_SOURCE_H5] = _urbansim_population_source_path(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return outputs


def _resolve_atlas_preprocess_inputs(
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
    usim_input_fname = get_usim_datastore_fname(
        settings,
        io="input" if state.is_start_year() else "output",
        year=state.forecast_year,
    )
    atlas_input = Path(workspace.get_usim_mutable_data_dir()) / usim_input_fname
    if not atlas_input.exists():
        atlas_input = Path(
            _urbansim_run_native_output_paths(
                settings=settings, state=state, workspace=workspace
            )[USIM_DATASTORE_H5]
        )
    return _resolve_native_inputs(
        step_name="atlas_preprocess",
        function=_native_atlas_preprocess,
        required_roles=(
            USIM_DATASTORE_H5,
            *((FINAL_SKIMS_OMX,) if requires_beam_skim else ()),
        ),
        optional_roles=(() if requires_beam_skim else (FINAL_SKIMS_OMX,)),
        logical_destinations={
            USIM_DATASTORE_H5: Path(atlas_input),
            FINAL_SKIMS_OMX: Path(workspace.get_atlas_mutable_input_dir())
            / "final_skims.omx",
        },
        coupler=coupler,
    )


def _resolve_atlas_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    prepared_inputs = _atlas_preprocess_prepared_input_paths(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    launch_context = AtlasLaunchContext(
        input_root=Path(workspace.get_atlas_mutable_input_dir()),
        output_root=Path(workspace.get_atlas_output_dir()),
    )
    return _resolve_native_inputs(
        step_name="atlas_run",
        function=_native_atlas_run,
        required_roles=tuple(prepared_inputs),
        optional_roles=(),
        logical_destinations=prepared_inputs,
        coupler=coupler,
        metadata={"atlas_launch_context": launch_context},
    )


def _resolve_atlas_postprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    forecast_year = state.forecast_year
    semantic_roles = {
        _ATLAS_HOUSEHOLDV_CSV_INPUT: f"householdv_{forecast_year}",
        _ATLAS_VEHICLES_CSV_INPUT: f"vehicles_{forecast_year}",
        USIM_DATASTORE_H5: USIM_DATASTORE_H5,
    }
    logical_destinations = {
        _ATLAS_HOUSEHOLDV_CSV_INPUT: Path(workspace.get_atlas_output_dir())
        / f"householdv_{forecast_year}.csv",
        _ATLAS_VEHICLES_CSV_INPUT: Path(workspace.get_atlas_output_dir())
        / f"vehicles_{forecast_year}.csv",
        USIM_DATASTORE_H5: Path(
            _urbansim_run_native_output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
            )[USIM_DATASTORE_H5]
        ),
    }
    selected_inputs: dict[str, Any] = {}
    source_by_role: dict[str, str] = {}
    selected_key_by_role: dict[str, str] = {}
    for parameter, semantic_role in semantic_roles.items():
        resolved = resolve_coupler_value(coupler, semantic_role)
        source_by_role[parameter] = resolved.source
        if resolved.storage_key is not None:
            selected_key_by_role[parameter] = resolved.storage_key
        if resolved.value is not None:
            selected_inputs[parameter] = resolved.value

    return ResolvedStepInputs(
        step_name="atlas_postprocess",
        binding=BindingResult(inputs=selected_inputs),
        required_roles=tuple(semantic_roles),
        source_by_role=source_by_role,
        selected_key_by_role=selected_key_by_role,
        logical_destinations=logical_destinations,
        metadata={"semantic_role_by_parameter": semantic_roles},
    )


_ATLAS_PREPROCESS_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="conditional workspace skim fallback remains available",
    config_contract=ConfigContract.payload(),
)
_ATLAS_RUN_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="static inputs are identity-only model-root reads",
    config_contract=ConfigContract.payload(),
)
_ATLAS_POSTPROCESS_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="vehicle mapping is discovered in the mutable ATLAS root",
    config_contract=ConfigContract.payload(),
)


@define_step(
    model="atlas_preprocess",
    name_template="atlas_preprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_DATASTORE_H5: None},
    optional_input_keys=(FINAL_SKIMS_OMX,),
    schema_outputs=[
        "atlas_households_csv",
        "atlas_blocks_csv",
        "atlas_persons_csv",
        "atlas_residential_csv",
        "atlas_jobs_csv",
        "atlas_grave_csv",
        "atlas_accessibility_csv",
    ],
    output_paths=_native_contract_output_paths(_atlas_preprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta(
        "atlas_preprocess", input_contract=_ATLAS_PREPROCESS_INPUT_CONTRACT
    ),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_atlas_preprocess(
    usim_datastore_h5: Path,
    final_skims_omx: Path | None = None,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    AtlasPreprocessor("atlas", state).preprocess(
        workspace,
        usim_datastore_h5=usim_datastore_h5,
        final_skims_omx=final_skims_omx,
        allow_workspace_skim_fallback=not requires_prior_beam_skim_handoff(
            settings=settings,
            state=state,
        ),
    )


@define_step(
    model="atlas_run",
    name_template="atlas_run__y{year}__i{iteration}__phase_{phase}",
    inputs={
        "atlas_households_csv": None,
        "atlas_blocks_csv": None,
        "atlas_persons_csv": None,
        "atlas_residential_csv": None,
        "atlas_jobs_csv": None,
        "atlas_grave_csv": None,
        "atlas_accessibility_csv": None,
    },
    schema_outputs=[],
    output_paths=_native_contract_output_paths(_atlas_run_native_output_paths),
    input_binding="paths",
    **consist_step_meta("atlas_run", input_contract=_ATLAS_RUN_INPUT_CONTRACT),
)
@require_runtime_kwargs("settings", "state", "workspace", "atlas_launch_context")
def _native_atlas_run(
    atlas_households_csv: Path,
    atlas_blocks_csv: Path,
    atlas_persons_csv: Path,
    atlas_residential_csv: Path,
    atlas_jobs_csv: Path,
    atlas_grave_csv: Path | None = None,
    atlas_accessibility_csv: Path | None = None,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    atlas_launch_context: AtlasLaunchContext,
) -> None:
    received_inputs = {
        "atlas_households_csv": atlas_households_csv,
        "atlas_blocks_csv": atlas_blocks_csv,
        "atlas_persons_csv": atlas_persons_csv,
        "atlas_residential_csv": atlas_residential_csv,
        "atlas_jobs_csv": atlas_jobs_csv,
        **({"atlas_grave_csv": atlas_grave_csv} if atlas_grave_csv is not None else {}),
        **(
            {"atlas_accessibility_csv": atlas_accessibility_csv}
            if atlas_accessibility_csv is not None
            else {}
        ),
    }
    expected_inputs = _atlas_preprocess_prepared_input_paths(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    if received_inputs != expected_inputs:
        raise RuntimeError(
            "ATLAS run received prepared inputs at unexpected destinations: "
            f"expected {expected_inputs}, got {received_inputs}"
        )
    AtlasRunner("atlas", state).run(
        AtlasPreprocessOutputs(
            atlas_mutable_input_dir=atlas_launch_context.input_root,
            prepared_inputs=received_inputs,
        ),
        atlas_launch_context,
    )


@define_step(
    model="atlas_postprocess",
    name_template="atlas_postprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={
        _ATLAS_HOUSEHOLDV_CSV_INPUT: None,
        _ATLAS_VEHICLES_CSV_INPUT: None,
        USIM_DATASTORE_H5: None,
    },
    schema_outputs=[USIM_POPULATION_SOURCE_H5, ATLAS_VEHICLES2_OUTPUT],
    output_paths=_native_contract_output_paths(_atlas_postprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta(
        "atlas_postprocess", input_contract=_ATLAS_POSTPROCESS_INPUT_CONTRACT
    ),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_atlas_postprocess(
    atlas_householdv_csv: Path,
    atlas_vehicles_csv: Path,
    usim_datastore_h5: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    forecast_year = state.forecast_year
    atlas_output_dir = Path(workspace.get_atlas_output_dir())
    expected_raw_outputs = {
        f"householdv_{forecast_year}": atlas_output_dir
        / f"householdv_{forecast_year}.csv",
        f"vehicles_{forecast_year}": atlas_output_dir / f"vehicles_{forecast_year}.csv",
    }
    received_raw_outputs = {
        f"householdv_{forecast_year}": atlas_householdv_csv,
        f"vehicles_{forecast_year}": atlas_vehicles_csv,
    }
    for key, expected_path in expected_raw_outputs.items():
        received_path = received_raw_outputs[key]
        if received_path != expected_path:
            raise RuntimeError(
                "ATLAS postprocess received raw output at an unexpected destination: "
                f"expected {expected_path}, got {received_path}"
            )
    AtlasPostprocessor("atlas", state).postprocess(
        AtlasRunOutputs(
            atlas_output_dir=atlas_output_dir,
            raw_outputs=received_raw_outputs,
        ),
        workspace,
        usim_datastore_h5=usim_datastore_h5,
    )
    destination_h5 = Path(
        _atlas_postprocess_native_output_paths(
            settings=settings, state=state, workspace=workspace
        )[USIM_POPULATION_SOURCE_H5]
    )
    create_usim_population_source_snapshot(
        source_h5=usim_datastore_h5,
        destination_h5=destination_h5,
        year=state.forecast_year,
    )


def _project_atlas_preprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> AtlasPreprocessOutputs:
    declared_outputs = _atlas_preprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    prepared_inputs = {
        key: _persisted_output_path(
            outputs,
            step_name="atlas_preprocess",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in declared_outputs
    }
    projected = AtlasPreprocessOutputs(
        atlas_mutable_input_dir=Path(workspace.get_atlas_mutable_input_dir()),
        prepared_inputs=prepared_inputs,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="atlas_preprocess",
        )
    )
    return projected


def _project_atlas_run(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> AtlasRunOutputs:
    declared_outputs = _atlas_run_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    projected = AtlasRunOutputs(
        atlas_output_dir=Path(workspace.get_atlas_output_dir()),
        raw_outputs={
            key: _persisted_output_path(
                outputs,
                step_name="atlas_run",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in declared_outputs
        },
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="atlas_run",
        )
    )
    return projected


def _project_atlas_postprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> AtlasPostprocessOutputs:
    declared_outputs = _atlas_postprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    projected = AtlasPostprocessOutputs(
        atlas_output_dir=Path(workspace.get_atlas_output_dir()),
        usim_datastore_h5=_persisted_output_path(
            outputs,
            step_name="atlas_postprocess",
            key=USIM_POPULATION_SOURCE_H5,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        processed_outputs={
            ATLAS_VEHICLES2_OUTPUT: _persisted_output_path(
                outputs,
                step_name="atlas_postprocess",
                key=ATLAS_VEHICLES2_OUTPUT,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
        },
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="atlas_postprocess",
        )
    )
    return projected


ATLAS_PREPROCESS = StepDefinition(
    name="atlas_preprocess",
    function=_native_atlas_preprocess,
    resolve_inputs=_resolve_atlas_preprocess_inputs,
    project_outputs=_project_atlas_preprocess,
    input_contract=_ATLAS_PREPROCESS_INPUT_CONTRACT,
    output_paths=_atlas_preprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
    binding_rules=(
        ArtifactBindingRule(
            semantic_key=USIM_DATASTORE_CURRENT_H5,
            required=True,
            allow_fallback=True,
            preferred_keys=(USIM_DATASTORE_CURRENT_H5, USIM_H5_UPDATED),
        ),
        ArtifactBindingRule(
            semantic_key=USIM_DATASTORE_BASE_H5,
            required=True,
            allow_fallback=True,
            preferred_keys=(
                USIM_DATASTORE_BASE_H5,
                USIM_DATASTORE_CURRENT_H5,
                USIM_H5_UPDATED,
            ),
        ),
        ArtifactBindingRule(
            semantic_key=OMX_SKIMS,
            required=False,
            allow_fallback=True,
            preferred_keys=(FINAL_SKIMS_OMX, OMX_SKIMS),
        ),
    ),
    optional_output_keys=("atlas_accessibility_csv",),
)

ATLAS_RUN = StepDefinition(
    name="atlas_run",
    function=_native_atlas_run,
    resolve_inputs=_resolve_atlas_run_inputs,
    project_outputs=_project_atlas_run,
    input_contract=_ATLAS_RUN_INPUT_CONTRACT,
    output_paths=_atlas_run_native_output_paths,
    output_sets=_atlas_run_output_sets,
    execution_options=_atlas_run_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

ATLAS_POSTPROCESS = StepDefinition(
    name="atlas_postprocess",
    function=_native_atlas_postprocess,
    resolve_inputs=_resolve_atlas_postprocess_inputs,
    project_outputs=_project_atlas_postprocess,
    input_contract=_ATLAS_POSTPROCESS_INPUT_CONTRACT,
    output_paths=_atlas_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

# Canonical registry members use the workflow step names. Uppercase aliases keep
# the definitions convenient to import in focused model tests.
atlas_preprocess = ATLAS_PREPROCESS
atlas_run = ATLAS_RUN
atlas_postprocess = ATLAS_POSTPROCESS
