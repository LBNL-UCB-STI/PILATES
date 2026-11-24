from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from consist import BindingResult, CacheOptions, ExecutionOptions, define_step

from pilates.config.models import PilatesConfig
from pilates.impacts.outputs import ImpactsPostsimOutputs, ImpactsPresimOutputs
from pilates.impacts.postsim import ImpactsPostsimRunner
from pilates.impacts.presim import ImpactsPresimRunner
from pilates.utils.consist_runtime import require_runtime_kwargs
from pilates.workflows.artifact_keys import (
    IMPACTS_EXPOSURE_TABLE,
    IMPACTS_PIPELINE_MANIFEST,
    IMPACTS_POSTPROCESS_MANIFEST,
    IMPACTS_POSTSIM_SETTINGS,
    IMPACTS_PRESIM_SETTINGS,
)
from pilates.workflows.consist_metadata import consist_step_meta
from pilates.workflows.contracts import ConfigContract, InputContract, StepDefinition
from pilates.workflows.output_projection import require_output
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.state import WorkflowState
from pilates.workflows.steps.shared import _persisted_output_path
from pilates.workspace import Workspace


_IMPACTS_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="external IMPACTS execution consumes files staged in the run workspace",
    config_contract=ConfigContract.payload(),
)


def _impacts_output_dir(*, settings: PilatesConfig, state: WorkflowState, workspace: Workspace, stage: str) -> Path:
    year = state.forecast_year or settings.run.start_year
    output_run_name = f"{settings.run.output_run_name}_y{year}"
    prefix = "impacts_presim" if stage == "presim" else "impacts-postsim"
    return Path(workspace.get_impacts_output_dir()) / f"{prefix}--{settings.run.region}--{output_run_name}"


def _presim_paths(*, settings: PilatesConfig, state: WorkflowState, workspace: Workspace, **_: Any) -> Mapping[str, Path]:
    if getattr(settings, "impacts", None) is None:
        return {}
    del state
    return {IMPACTS_PRESIM_SETTINGS: Path(workspace.get_impacts_settings_path("presim"))}


def _postsim_paths(*, settings: PilatesConfig, state: WorkflowState, workspace: Workspace, **_: Any) -> Mapping[str, Path]:
    if getattr(settings, "impacts", None) is None:
        return {}
    output_dir = _impacts_output_dir(settings=settings, state=state, workspace=workspace, stage="postsim")
    return {
        IMPACTS_POSTSIM_SETTINGS: Path(workspace.get_impacts_settings_path("postsim")),
        IMPACTS_PIPELINE_MANIFEST: output_dir / "pipeline_manifest.yaml",
        IMPACTS_POSTPROCESS_MANIFEST: output_dir / "postprocess_manifest.yaml",
        IMPACTS_EXPOSURE_TABLE: output_dir / "exposure" / "beam_concentration_distribution.parquet",
    }


@define_step(
    model="impacts_presim",
    name_template="impacts_presim__y{year}__i{iteration}__phase_{phase}",
    schema_outputs=[IMPACTS_PRESIM_SETTINGS],
    output_paths=lambda context: _presim_paths(
        settings=context.get_runtime("settings"),
        state=context.get_runtime("state"),
        workspace=context.get_runtime("workspace"),
    ),
    input_binding="paths",
    **consist_step_meta("impacts_presim", input_contract=_IMPACTS_INPUT_CONTRACT),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_impacts_presim(*, settings: PilatesConfig, state: WorkflowState, workspace: Workspace) -> None:
    if settings.impacts is None:
        raise ValueError("Impacts config is missing")
    ImpactsPresimRunner("impacts_presim", state).run(None, workspace)


@define_step(
    model="impacts_postsim",
    name_template="impacts_postsim__y{year}__i{iteration}__phase_{phase}",
    schema_outputs=[
        IMPACTS_POSTSIM_SETTINGS,
        IMPACTS_PIPELINE_MANIFEST,
        IMPACTS_POSTPROCESS_MANIFEST,
        IMPACTS_EXPOSURE_TABLE,
    ],
    output_paths=lambda context: _postsim_paths(
        settings=context.get_runtime("settings"),
        state=context.get_runtime("state"),
        workspace=context.get_runtime("workspace"),
    ),
    input_binding="paths",
    **consist_step_meta("impacts_postsim", input_contract=_IMPACTS_INPUT_CONTRACT),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_impacts_postsim(*, settings: PilatesConfig, state: WorkflowState, workspace: Workspace) -> None:
    if settings.impacts is None:
        raise ValueError("Impacts config is missing")
    ImpactsPostsimRunner("impacts_postsim", state).run(None, workspace)


def _resolve_impacts_inputs(*, step_name: str, **_: Any) -> ResolvedStepInputs:
    return ResolvedStepInputs(step_name=step_name, binding=BindingResult(inputs={}))


def _project_presim_outputs(outputs: Mapping[str, Any], *, settings: PilatesConfig, state: WorkflowState, workspace: Workspace) -> ImpactsPresimOutputs:
    paths = _presim_paths(settings=settings, state=state, workspace=workspace)
    return ImpactsPresimOutputs(
        output_dir=Path(workspace.get_impacts_output_dir()),
        settings_path=_persisted_output_path(outputs, step_name="impacts_presim", key=IMPACTS_PRESIM_SETTINGS, declared_outputs=paths, workspace=workspace),
        bridge_settings_path=Path(workspace.get_impacts_bridge_settings_path("presim")),
    )


def _project_postsim_outputs(outputs: Mapping[str, Any], *, settings: PilatesConfig, state: WorkflowState, workspace: Workspace) -> ImpactsPostsimOutputs:
    paths = _postsim_paths(settings=settings, state=state, workspace=workspace)
    for key in paths:
        require_output(outputs, step_name="impacts_postsim", key=key)
    return ImpactsPostsimOutputs(
        output_dir=_impacts_output_dir(settings=settings, state=state, workspace=workspace, stage="postsim"),
        settings_path=_persisted_output_path(outputs, step_name="impacts_postsim", key=IMPACTS_POSTSIM_SETTINGS, declared_outputs=paths, workspace=workspace),
        bridge_settings_path=Path(workspace.get_impacts_bridge_settings_path("postsim")),
        pipeline_manifest=_persisted_output_path(outputs, step_name="impacts_postsim", key=IMPACTS_PIPELINE_MANIFEST, declared_outputs=paths, workspace=workspace),
        postprocess_manifest=_persisted_output_path(outputs, step_name="impacts_postsim", key=IMPACTS_POSTPROCESS_MANIFEST, declared_outputs=paths, workspace=workspace),
        exposure_table=_persisted_output_path(outputs, step_name="impacts_postsim", key=IMPACTS_EXPOSURE_TABLE, declared_outputs=paths, workspace=workspace),
    )


def _execution_options(**_: Any) -> ExecutionOptions:
    return ExecutionOptions(input_binding="paths")


def _generated_fleet_cache(*, settings: PilatesConfig, **_: Any) -> CacheOptions:
    if getattr(getattr(settings, "impacts", None), "use_generated_fleet", False):
        # shortcut: rerun IMPACTS until the Consist contract tracks the fleet handoff.
        return CacheOptions(cache_mode="overwrite")
    return CacheOptions()


impacts_presim = StepDefinition(
    name="impacts_presim",
    function=_native_impacts_presim,
    resolve_inputs=lambda **kwargs: _resolve_impacts_inputs(step_name="impacts_presim", **kwargs),
    project_outputs=_project_presim_outputs,
    input_contract=_IMPACTS_INPUT_CONTRACT,
    output_paths=_presim_paths,
    execution_options=_execution_options,
    cache_options=_generated_fleet_cache,
)
impacts_postsim = StepDefinition(
    name="impacts_postsim",
    function=_native_impacts_postsim,
    resolve_inputs=lambda **kwargs: _resolve_impacts_inputs(step_name="impacts_postsim", **kwargs),
    project_outputs=_project_postsim_outputs,
    input_contract=_IMPACTS_INPUT_CONTRACT,
    output_paths=_postsim_paths,
    execution_options=_execution_options,
    cache_options=_generated_fleet_cache,
)
