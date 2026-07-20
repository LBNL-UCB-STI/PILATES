from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from consist import (
    Artifact,
    BindingResult,
    CacheOptions,
    ExecutionOptions,
    ResolvedBinding,
    StepIdentity,
    define_step,
)

from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasRunner
from pilates.config.models import PilatesConfig
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs as NativeUrbanSimPostprocessOutputs,
)
from pilates.urbansim.outputs import (
    UrbanSimPreprocessOutputs as NativeUrbanSimPreprocessOutputs,
)
from pilates.urbansim.outputs import UrbanSimRunOutputs as NativeUrbanSimRunOutputs
from pilates.urbansim.postprocessor import UrbansimPostprocessor
from pilates.urbansim.runner import UrbansimRunner
from pilates.urbansim.preprocessor import UrbansimPreprocessor
from pilates.workflows.artifact_keys import (
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    FINAL_SKIMS_OMX,
    USIM_POPULATION_SOURCE_H5,
    USIM_MUTABLE_DATA_DIR,
)
from pilates.workflows.binding import build_resolved_binding
from pilates.workflows.output_projection import require_output
from pilates.workflows.outputs_base import ValidationContext
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_definition import StepDefinition
from pilates.workflows.coupler_namespace import resolve_coupler_value
from pilates.workflows.step_consist_meta import consist_step_meta
from pilates.utils.consist_runtime import require_runtime_kwargs
from pilates.workspace import Workspace

# Model-specific step factories for UrbanSim and ATLAS.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_INPUT_MERGED_PREFIX,
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
    CouplerProtocol,
    WorkflowState,
)


# Native Consist step definitions.


def _persisted_output_path(
    outputs: Mapping[str, Any],
    *,
    step_name: str,
    key: str,
    declared_outputs: Mapping[str, Any],
    workspace: Workspace,
) -> Path:
    """Return one output only when it exists at its declared destination.

    A projection represents the current step execution. It must not fall back
    to an artifact's mounted source, archive location, or historical workspace
    merely because that artifact remains available after a cache hit.
    """

    require_output(outputs, step_name=step_name, key=key)
    try:
        output_spec = declared_outputs[key]
    except KeyError as exc:
        raise RuntimeError(
            f"{step_name} output {key!r} has no declared destination"
        ) from exc

    if isinstance(output_spec, os.PathLike):
        path = Path(output_spec)
    elif isinstance(output_spec, str):
        if output_spec.startswith("workspace://"):
            path = Path(workspace.full_path) / output_spec[
                len("workspace://") :
            ].lstrip("/")
        elif "://" in output_spec:
            raise RuntimeError(
                f"{step_name} output {key!r} has a non-local declared destination"
            )
        else:
            path = Path(output_spec)
            if not path.is_absolute():
                path = Path(workspace.full_path) / path
    else:
        raise RuntimeError(
            f"{step_name} output {key!r} has an invalid declared destination"
        )

    if not path.exists():
        raise RuntimeError(
            f"{step_name} output {key!r} is missing at declared destination {path}"
        )
    return path


def _validation_context(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_name: str,
) -> ValidationContext:
    return ValidationContext(
        settings=settings,
        state=state,
        workspace=workspace,
        step_name=step_name,
    )


def _resolve_native_inputs(
    *,
    step_name: str,
    function: Callable[..., Any],
    required_roles: Sequence[str],
    optional_roles: Sequence[str],
    logical_destinations: Mapping[str, Path],
    coupler: CouplerProtocol,
    step_identity: StepIdentity | None = None,
) -> ResolvedStepInputs:
    """Select fixed semantic roles once and retain their Consist artifacts."""

    selected_inputs: Dict[str, Any] = {}
    source_by_role: Dict[str, str] = {}
    selected_key_by_role: Dict[str, str] = {}
    selected_destinations: Dict[str, Path] = {}

    for role in (*required_roles, *optional_roles):
        resolved = resolve_coupler_value(coupler, role)
        source_by_role[role] = resolved.source
        if resolved.storage_key is not None:
            selected_key_by_role[role] = resolved.storage_key
        if resolved.value is None:
            continue
        selected_inputs[role] = resolved.value
        destination = logical_destinations.get(role)
        if destination is not None:
            selected_destinations[role] = destination

    if step_identity is None:
        binding: BindingResult | ResolvedBinding = BindingResult(inputs=selected_inputs)
    else:
        selected_artifacts: dict[str, Artifact] = {}
        for role, value in selected_inputs.items():
            if not isinstance(value, Artifact):
                raise TypeError(
                    f"{step_name} strict binding requires a tracked Artifact for "
                    f"{role!r}, got {type(value).__name__}"
                )
            selected_artifacts[role] = value
        binding = build_resolved_binding(
            step_name=step_name,
            function=function,
            selected_artifacts=selected_artifacts,
            logical_destinations={
                role: Path("inputs") / role for role in selected_artifacts
            },
            selection_diagnostics={
                "source_by_role": source_by_role,
                "selected_key_by_role": selected_key_by_role,
            },
            source_by_parameter={
                role: source_by_role[role] for role in selected_artifacts
            },
            step_identity=step_identity,
        )

    return ResolvedStepInputs(
        step_name=step_name,
        binding=binding,
        required_roles=tuple(required_roles),
        optional_roles=tuple(optional_roles),
        source_by_role=source_by_role,
        selected_key_by_role=selected_key_by_role,
        logical_destinations=selected_destinations,
    )


def _native_execution_options(
    *,
    resolved_inputs: ResolvedStepInputs | None = None,
    **_: Any,
) -> ExecutionOptions:
    """Request Consist staging at the deterministic model-owned destinations."""

    return ExecutionOptions(
        input_binding="paths",
        input_paths=resolved_inputs.logical_destinations,
        input_materialization="requested",
        input_materialization_mode="copy",
    )


def _strict_requested_output_cache_options(**_: Any) -> CacheOptions:
    """Admit a cached model step only after all requested output paths hydrate."""

    return CacheOptions(
        cache_hydration="outputs-requested",
        cache_hydration_failure="miss",
    )


def _native_contract_output_paths(
    provider: Callable[..., Mapping[str, Any]],
) -> Callable[[Any], Mapping[str, Any]]:
    """Expose a StepDefinition path provider to direct Consist resolution."""

    def resolve(context: Any) -> Mapping[str, Any]:
        settings = context.get_runtime("settings", default=None)
        state = context.get_runtime("state", default=None)
        workspace = context.get_runtime("workspace", default=None)
        if settings is None or state is None or workspace is None:
            return {}
        return provider(settings=settings, state=state, workspace=workspace)

    return resolve


def _urbansim_preprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    return UrbansimPreprocessor.expected_outputs(settings, state, workspace)


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
    return paths


def _atlas_preprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    year_input_dir = atlas_input_dir / f"year{state.forecast_year}"
    paths: Dict[str, Any] = {
        "atlas_mutable_input_dir": atlas_input_dir,
        "atlas_households_csv": year_input_dir / "households.csv",
        "atlas_blocks_csv": year_input_dir / "blocks.csv",
        "atlas_persons_csv": year_input_dir / "persons.csv",
        "atlas_residential_csv": year_input_dir / "residential_units.csv",
        "atlas_jobs_csv": year_input_dir / "jobs.csv",
    }
    if state.year > state.start_year:
        paths["atlas_grave_csv"] = year_input_dir / "grave.csv"
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
        ATLAS_OUTPUT_DIR: output_dir,
        f"householdv_{forecast_year}": output_dir / f"householdv_{forecast_year}.csv",
        f"vehicles_{forecast_year}": output_dir / f"vehicles_{forecast_year}.csv",
    }


def _atlas_postprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    outputs = AtlasPostprocessor.expected_outputs(settings, state, workspace)
    # ATLAS mutates the exact datastore supplied to the callable.  Native
    # resolution stages that role at UrbanSim's forecast-output destination,
    # including in the start year; output projection must name that same file.
    outputs[USIM_POPULATION_SOURCE_H5] = _urbansim_run_native_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
    )[USIM_DATASTORE_H5]
    return outputs


def _resolve_urbansim_preprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    output_paths = _urbansim_preprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    destinations = {
        USIM_DATASTORE_H5: Path(output_paths[USIM_DATASTORE_H5]),
        FINAL_SKIMS_OMX: Path(workspace.get_usim_mutable_data_dir())
        / "final_skims.omx",
    }
    return _resolve_native_inputs(
        step_name="urbansim_preprocess",
        function=_native_urbansim_preprocess,
        required_roles=(USIM_DATASTORE_H5,),
        optional_roles=(FINAL_SKIMS_OMX,),
        logical_destinations=destinations,
        coupler=coupler,
    )


def _resolve_urbansim_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    del settings, state
    return _resolve_native_inputs(
        step_name="urbansim_run",
        function=_native_urbansim_run,
        required_roles=(USIM_MUTABLE_DATA_DIR,),
        optional_roles=(),
        logical_destinations={
            USIM_MUTABLE_DATA_DIR: Path(workspace.get_usim_mutable_data_dir())
        },
        coupler=coupler,
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


def _resolve_atlas_preprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    atlas_input = AtlasPreprocessor.expected_inputs(settings, state, workspace)[
        USIM_DATASTORE_H5
    ]
    if atlas_input is None:
        atlas_input = _urbansim_run_native_output_paths(
            settings=settings, state=state, workspace=workspace
        )[USIM_DATASTORE_H5]
    return _resolve_native_inputs(
        step_name="atlas_preprocess",
        function=_native_atlas_preprocess,
        required_roles=(USIM_DATASTORE_H5,),
        optional_roles=(),
        logical_destinations={USIM_DATASTORE_H5: Path(atlas_input)},
        coupler=coupler,
    )


def _resolve_atlas_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    del settings, state
    return _resolve_native_inputs(
        step_name="atlas_run",
        function=_native_atlas_run,
        required_roles=("atlas_mutable_input_dir",),
        optional_roles=(),
        logical_destinations={
            "atlas_mutable_input_dir": Path(workspace.get_atlas_mutable_input_dir())
        },
        coupler=coupler,
    )


def _resolve_atlas_postprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    return _resolve_native_inputs(
        step_name="atlas_postprocess",
        function=_native_atlas_postprocess,
        required_roles=(ATLAS_OUTPUT_DIR, USIM_DATASTORE_H5),
        optional_roles=(),
        logical_destinations={
            ATLAS_OUTPUT_DIR: Path(workspace.get_atlas_output_dir()),
            USIM_DATASTORE_H5: Path(
                _urbansim_run_native_output_paths(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                )[USIM_DATASTORE_H5]
            ),
        },
        coupler=coupler,
    )


@define_step(
    model="urbansim_preprocess",
    name_template="urbansim_preprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_DATASTORE_H5: None},
    optional_input_keys=(FINAL_SKIMS_OMX,),
    schema_outputs=[
        USIM_DATASTORE_H5,
        "omx_skims",
        "hh_size",
        "income_rates",
        "relmap",
        "geoid_to_zone",
        "schools",
        "school_districts",
    ],
    output_paths=_native_contract_output_paths(
        _urbansim_preprocess_native_output_paths
    ),
    input_binding="paths",
    **consist_step_meta("urbansim_preprocess"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_urbansim_preprocess(
    usim_datastore_h5: Path,
    final_skims_omx: Path | None = None,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    UrbansimPreprocessor("urbansim", state).preprocess(
        workspace,
        usim_datastore_h5=usim_datastore_h5,
        final_skims_omx=final_skims_omx,
    )


@define_step(
    model="urbansim_run",
    name_template="urbansim_run__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_MUTABLE_DATA_DIR: None},
    schema_outputs=[USIM_DATASTORE_H5, USIM_FORECAST_OUTPUT],
    output_paths=_native_contract_output_paths(_urbansim_run_native_output_paths),
    input_binding="paths",
    **consist_step_meta("urbansim_run"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_urbansim_run(
    usim_mutable_data_dir: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    UrbansimRunner("urbansim", state).run(
        NativeUrbanSimPreprocessOutputs(usim_mutable_data_dir=usim_mutable_data_dir),
        workspace,
    )


@define_step(
    model="urbansim_postprocess",
    name_template="urbansim_postprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_DATASTORE_H5: None},
    schema_outputs=[USIM_DATASTORE_H5],
    output_paths=_native_contract_output_paths(
        _urbansim_postprocess_native_output_paths
    ),
    input_binding="paths",
    **consist_step_meta("urbansim_postprocess"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_urbansim_postprocess(
    usim_datastore_h5: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    UrbansimPostprocessor("urbansim", state).postprocess(
        NativeUrbanSimRunOutputs(usim_datastore_h5=usim_datastore_h5),
        workspace,
    )


@define_step(
    model="atlas_preprocess",
    name_template="atlas_preprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_DATASTORE_H5: None},
    schema_outputs=[
        "atlas_mutable_input_dir",
        "atlas_households_csv",
        "atlas_blocks_csv",
        "atlas_persons_csv",
        "atlas_residential_csv",
        "atlas_jobs_csv",
        "atlas_grave_csv",
    ],
    output_paths=_native_contract_output_paths(_atlas_preprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta("atlas_preprocess"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_atlas_preprocess(
    usim_datastore_h5: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    AtlasPreprocessor("atlas", state).preprocess(
        workspace,
        usim_datastore_h5=usim_datastore_h5,
    )


@define_step(
    model="atlas_run",
    name_template="atlas_run__y{year}__i{iteration}__phase_{phase}",
    inputs={"atlas_mutable_input_dir": None},
    schema_outputs=[ATLAS_OUTPUT_DIR],
    output_paths=_native_contract_output_paths(_atlas_run_native_output_paths),
    input_binding="paths",
    **consist_step_meta("atlas_run"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_atlas_run(
    atlas_mutable_input_dir: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    AtlasRunner("atlas", state).run(
        AtlasPreprocessOutputs(atlas_mutable_input_dir=atlas_mutable_input_dir),
        workspace,
    )


@define_step(
    model="atlas_postprocess",
    name_template="atlas_postprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={ATLAS_OUTPUT_DIR: None, USIM_DATASTORE_H5: None},
    schema_outputs=[USIM_POPULATION_SOURCE_H5, ATLAS_VEHICLES2_OUTPUT],
    output_paths=_native_contract_output_paths(_atlas_postprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta("atlas_postprocess"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_atlas_postprocess(
    atlas_output_dir: Path,
    usim_datastore_h5: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    forecast_year = state.forecast_year
    AtlasPostprocessor("atlas", state).postprocess(
        AtlasRunOutputs(
            atlas_output_dir=atlas_output_dir,
            raw_outputs={
                f"householdv_{forecast_year}": atlas_output_dir
                / f"householdv_{forecast_year}.csv",
                f"vehicles_{forecast_year}": atlas_output_dir
                / f"vehicles_{forecast_year}.csv",
            },
        ),
        workspace,
        usim_datastore_h5=usim_datastore_h5,
    )


def _project_urbansim_preprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> NativeUrbanSimPreprocessOutputs:
    declared_outputs = _urbansim_preprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    prepared_inputs = {
        key: _persisted_output_path(
            outputs,
            step_name="urbansim_preprocess",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in outputs
        if key != USIM_MUTABLE_DATA_DIR
    }
    projected = NativeUrbanSimPreprocessOutputs(
        usim_mutable_data_dir=_persisted_output_path(
            outputs,
            step_name="urbansim_preprocess",
            key=USIM_MUTABLE_DATA_DIR,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        prepared_inputs=prepared_inputs,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="urbansim_preprocess",
        )
    )
    return projected


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
        for key in outputs
        if key != "atlas_mutable_input_dir"
    }
    projected = AtlasPreprocessOutputs(
        atlas_mutable_input_dir=_persisted_output_path(
            outputs,
            step_name="atlas_preprocess",
            key="atlas_mutable_input_dir",
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
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
        atlas_output_dir=_persisted_output_path(
            outputs,
            step_name="atlas_run",
            key=ATLAS_OUTPUT_DIR,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        raw_outputs={
            key: _persisted_output_path(
                outputs,
                step_name="atlas_run",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in outputs
            if key != ATLAS_OUTPUT_DIR
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
        atlas_output_dir=_persisted_output_path(
            outputs,
            step_name="atlas_postprocess",
            key=ATLAS_OUTPUT_DIR,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
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


URBANSIM_PREPROCESS = StepDefinition(
    name="urbansim_preprocess",
    function=_native_urbansim_preprocess,
    resolve_inputs=_resolve_urbansim_preprocess_inputs,
    project_outputs=_project_urbansim_preprocess,
    output_paths=_urbansim_preprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

URBANSIM_RUN = StepDefinition(
    name="urbansim_run",
    function=_native_urbansim_run,
    resolve_inputs=_resolve_urbansim_run_inputs,
    project_outputs=_project_urbansim_run,
    output_paths=_urbansim_run_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

URBANSIM_POSTPROCESS = StepDefinition(
    name="urbansim_postprocess",
    function=_native_urbansim_postprocess,
    resolve_inputs=_resolve_urbansim_postprocess_inputs,
    project_outputs=_project_urbansim_postprocess,
    output_paths=_urbansim_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
    preflight_identity=True,
)

ATLAS_PREPROCESS = StepDefinition(
    name="atlas_preprocess",
    function=_native_atlas_preprocess,
    resolve_inputs=_resolve_atlas_preprocess_inputs,
    project_outputs=_project_atlas_preprocess,
    output_paths=_atlas_preprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

ATLAS_RUN = StepDefinition(
    name="atlas_run",
    function=_native_atlas_run,
    resolve_inputs=_resolve_atlas_run_inputs,
    project_outputs=_project_atlas_run,
    output_paths=_atlas_run_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

ATLAS_POSTPROCESS = StepDefinition(
    name="atlas_postprocess",
    function=_native_atlas_postprocess,
    resolve_inputs=_resolve_atlas_postprocess_inputs,
    project_outputs=_project_atlas_postprocess,
    output_paths=_atlas_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

# Canonical registry members use the workflow step names.  Uppercase aliases keep
# the six definitions convenient to import in focused model tests.
urbansim_preprocess = URBANSIM_PREPROCESS
urbansim_run = URBANSIM_RUN
urbansim_postprocess = URBANSIM_POSTPROCESS
atlas_preprocess = ATLAS_PREPROCESS
atlas_run = ATLAS_RUN
atlas_postprocess = ATLAS_POSTPROCESS
