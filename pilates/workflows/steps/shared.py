"""Shared helpers, facet builders, and contract validation for workflow steps.

Step input/output contracts and semantic artifact roles are defined in
``pilates/workflows/catalog.py`` and the native ``StepDefinition`` modules
under ``pilates/workflows/steps/``.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Sequence,
)

from consist import (
    Artifact,
    BindingResult,
    CacheOptions,
    ExecutionOptions,
    ResolvedBinding,
    StepIdentity,
)

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ZARR_SKIMS,
)
from pilates.workflows.binding import build_resolved_binding
from pilates.workflows.coupler_namespace import resolve_coupler_value
from pilates.workflows.output_projection import require_output
from pilates.workflows.outputs_base import ValidationContext
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workspace import Workspace
from workflow_state import WorkflowState


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
    metadata: Mapping[str, Any] | None = None,
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
        metadata=metadata or {},
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


def _execution_options_with_launch_context(
    *,
    resolved_inputs: ResolvedStepInputs | None,
    step_name: str,
    context_key: str,
    expected_type: type[Any],
) -> ExecutionOptions:
    """Pass a resolver-owned launch context through ExecutionOptions.runtime_kwargs."""

    if resolved_inputs is None:
        raise RuntimeError(f"{step_name} requires resolved inputs before execution")
    launch_context = resolved_inputs.metadata.get(context_key)
    if not isinstance(launch_context, expected_type):
        raise RuntimeError(f"{step_name} is missing its resolved launch context")
    options = _native_execution_options(resolved_inputs=resolved_inputs)
    return replace(
        options,
        runtime_kwargs={context_key: launch_context},
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


def _activitysim_output_facet_meta(
    short_name: str,
    *,
    year: int,
    iteration: int,
) -> Dict[str, Any]:
    family = None
    snapshot_meta: Dict[str, Any] = {}
    if short_name.endswith("_asim_out"):
        family = short_name[: -len("_asim_out")]
    elif short_name.startswith("asim_input_") and short_name.endswith("_archived"):
        family = "asim_input_archived"
        input_name = short_name.removeprefix("asim_input_").removesuffix("_archived")
        source_role_map = {
            "households_csv": ASIM_HOUSEHOLDS_IN,
            "persons_csv": ASIM_PERSONS_IN,
            "land_use_csv": ASIM_LAND_USE_IN,
            "skims_omx": ASIM_OMX_SKIMS,
            "skims_zarr": ZARR_SKIMS,
        }
        snapshot_meta = {
            "source_role": source_role_map.get(input_name, input_name),
            "snapshot_role": f"asim_input_{input_name}",
            "snapshot_reason": "exact_rewind",
            "storage_event": "snapshot_copy",
        }
    elif short_name == ZARR_SKIMS:
        family = "zarr_skims"
    if family is None:
        return {}
    result: Dict[str, Any] = {
        "facet": {
            "artifact_family": family,
            **snapshot_meta,
            "year": year,
            "iteration": iteration,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }
    if short_name == ZARR_SKIMS:
        result["driver"] = "zarr"
        result["meta"] = {"directory_artifact": True}
    return result
