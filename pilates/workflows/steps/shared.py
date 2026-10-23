"""Shared helpers, facet builders, and contract validation for workflow steps.

Step input/output contracts and semantic artifact roles are defined in
``pilates/workflows/catalog.py`` and the native ``StepDefinition`` modules
under ``pilates/workflows/steps/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
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
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_H5,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_INPUT_MERGED_PREFIX,
    USIM_FORECAST_OUTPUT,
    ZARR_SKIMS,
)
from pilates.workflows.binding import build_resolved_binding
from pilates.workflows.catalog import (
    WORKFLOW_STEP_SPECS,
    workflow_step_spec_for_step_name,
)
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
    return ExecutionOptions(
        load_inputs=options.load_inputs,
        input_binding=options.input_binding,
        input_paths=options.input_paths,
        input_materialization=options.input_materialization,
        input_materialization_mode=options.input_materialization_mode,
        executor=options.executor,
        container=options.container,
        runtime_kwargs={context_key: launch_context},
        inject_context=options.inject_context,
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


def _parse_prefixed_iteration_key(
    short_name: str, prefix: str
) -> Optional[Dict[str, Any]]:
    marker = f"{prefix}_"
    if not short_name.startswith(marker):
        return None
    tail = short_name[len(marker) :]
    parts = tail.split("_")
    if len(parts) < 2:
        return None
    try:
        year = int(parts[0])
        iteration = int(parts[1])
    except ValueError:
        return None
    payload: Dict[str, Any] = {
        "year": year,
        "iteration": iteration,
    }
    if len(parts) > 2 and parts[2].startswith("sub"):
        try:
            payload["beam_sub_iteration"] = int(parts[2][3:])
        except ValueError:
            pass
    return payload


def _beam_artifact_facets(short_name: str) -> Optional[Dict[str, Any]]:
    for prefix, family in (
        ("events_parquet", "events_parquet"),
        ("raw_od_skims", "raw_od_skims"),
        ("raw_od_skims_zarr", "raw_od_skims_zarr"),
        ("linkstats_parquet", "linkstats_parquet"),
    ):
        parsed = _parse_prefixed_iteration_key(short_name, prefix)
        if parsed is not None:
            return {"artifact_family": family, **parsed}

    if short_name.startswith("linkstats_unmodified_parquet__"):
        tokens = short_name.split("__")
        payload: Dict[str, Any] = {
            "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet"
        }
        for token in tokens[1:]:
            if token.startswith("y"):
                try:
                    payload["year"] = int(token[1:])
                except ValueError:
                    return None
            elif token.startswith("i"):
                try:
                    payload["iteration"] = int(token[1:])
                except ValueError:
                    return None
            elif token.startswith("phys_sim_iter"):
                try:
                    payload["phys_sim_iteration"] = int(token[len("phys_sim_iter") :])
                except ValueError:
                    return None
            elif token.startswith("beam_sub_iter"):
                try:
                    payload["beam_sub_iteration"] = int(token[len("beam_sub_iter") :])
                except ValueError:
                    return None
        if {"year", "iteration", "phys_sim_iteration"} <= set(payload.keys()):
            return payload
        return None

    return None


def _beam_log_facet_meta(short_name: str) -> Dict[str, Any]:
    facet = _beam_artifact_facets(short_name)
    if not facet:
        return {}
    return {
        "facet": facet,
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _beam_postprocess_split_facet_meta(short_name: str) -> Dict[str, Any]:
    if short_name.startswith("events_parquet_") and "_type_" in short_name:
        head, event_type = short_name.split("_type_", 1)
        parsed = _parse_prefixed_iteration_key(head, "events_parquet")
        if parsed:
            return {
                "facet": {
                    "artifact_family": "events_parquet_split",
                    "event_type": event_type,
                    **parsed,
                },
                "facet_schema_version": "v1",
                "facet_index": True,
            }
    if short_name.startswith("path_traversal_links_"):
        parsed = _parse_prefixed_iteration_key(short_name, "path_traversal_links")
        if parsed:
            return {
                "facet": {
                    "artifact_family": "path_traversal_links",
                    **parsed,
                },
                "facet_schema_version": "v1",
                "facet_index": True,
            }
    return {}


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
    return {
        "facet": {
            "artifact_family": family,
            **snapshot_meta,
            "year": year,
            "iteration": iteration,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _urbansim_output_facet_meta(
    short_name: str,
    *,
    forecast_year: int,
) -> Dict[str, Any]:
    snapshot_meta: Dict[str, Any] = {}
    if short_name.startswith(USIM_INPUT_ARCHIVE_PREFIX):
        family = "usim_input_archive"
        snapshot_meta = {
            "source_role": USIM_DATASTORE_H5,
            "snapshot_role": "usim_input_archive",
            "snapshot_reason": "pre_merge_input",
            "storage_event": "snapshot_move",
        }
    elif short_name.startswith(USIM_INPUT_MERGED_PREFIX):
        family = "usim_input_merged"
        snapshot_meta = {
            "source_role": "usim_input_archive",
            "snapshot_role": "usim_input_merged",
            "snapshot_reason": "post_merge_handoff",
            "storage_event": "merged_h5_output",
        }
    elif short_name == USIM_FORECAST_OUTPUT:
        family = "usim_forecast_output"
    elif short_name == USIM_DATASTORE_H5:
        family = "usim_datastore_h5"
    elif short_name == USIM_DATASTORE_BASE_H5:
        family = "usim_datastore_base_h5"
    else:
        return {}
    return {
        "facet": {
            "artifact_family": family,
            **snapshot_meta,
            "year": forecast_year,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _atlas_artifact_facet_meta(
    short_name: str,
    *,
    run_scenario: Optional[str],
    forecast_year: int,
    artifact_family: str = "atlas_input",
) -> Dict[str, Any]:
    key = short_name.replace("\\", "/")
    key_compact = key.replace("/", "_")

    input_group = "global"
    parsed_scenario = None
    input_year = None

    if key.startswith("adopt/") or key_compact.startswith("adopt_"):
        input_group = "adopt"
        if key.startswith("adopt/"):
            parts = key.split("/")
            if len(parts) >= 2:
                parsed_scenario = parts[1]
        else:
            parts = key_compact.split("_")
            if len(parts) >= 2:
                parsed_scenario = parts[1]
    elif key_compact.startswith("vehicle_type_mapping_"):
        input_group = "vehicle_type_mapping"
        if "baseline" in key_compact:
            parsed_scenario = "baseline"
        elif "evMandForced2" in key_compact:
            parsed_scenario = "zev_mandate"
        elif "ESS_const_220_price" in key_compact:
            parsed_scenario = "ess_cons"
    elif key_compact.startswith("atlas_vehicles2"):
        input_group = "vehicles2"
    elif key_compact.startswith("usim_"):
        input_group = "usim"

    tail = key_compact.rsplit("_", 1)
    if len(tail) == 2 and len(tail[1]) == 4 and tail[1].isdigit():
        input_year = int(tail[1])

    facet: Dict[str, Any] = {
        "artifact_family": artifact_family,
        "input_group": input_group,
        "forecast_year": forecast_year,
    }
    scenario_value = parsed_scenario or run_scenario
    if scenario_value:
        facet["scenario"] = str(scenario_value)
    if input_year is not None:
        facet["input_year"] = input_year

    return {
        "facet": facet,
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _declared_step_model(step_func: Callable[..., Any]) -> Optional[str]:
    meta = getattr(step_func, "__consist_step__", None)
    model = getattr(meta, "model", None)
    if isinstance(model, str) and model:
        return model
    return None


def _declared_step_name(step_func: Callable[..., Any]) -> Optional[str]:
    """Return the registered native identity for a decorated step callable."""
    for spec in WORKFLOW_STEP_SPECS:
        if spec.definition.function is step_func:
            return spec.step_name
    return None


def validate_workflow_step_contracts(
    *,
    declared_steps: Optional[Iterable[Callable[..., Any]]] = None,
    allow_untracked_declared: Optional[Set[str]] = None,
    require_all_tracked_declared: bool = True,
) -> None:
    """Validate native decorated steps against the catalog contract."""
    if declared_steps is None:
        return

    errors: list[str] = []
    declared_by_name: Dict[str, Callable[..., Any]] = {}
    for step in declared_steps:
        model = _declared_step_model(step)
        if model is None:
            errors.append(
                "Declared step callable is missing __consist_step__.model metadata"
            )
            continue
        name = _declared_step_name(step) or model
        if name in declared_by_name:
            errors.append(f"Duplicate declared step identity: {name}")
            continue
        declared_by_name[name] = step

    expected_names = {spec.step_name for spec in WORKFLOW_STEP_SPECS}
    declared_names = set(declared_by_name)
    if require_all_tracked_declared:
        missing = expected_names - declared_names
        if missing:
            errors.append(
                "Native definitions missing from declared steps: "
                + ", ".join(sorted(missing))
            )
    allowed = {"postprocessing"} | set(allow_untracked_declared or ())
    unexpected = declared_names - expected_names - allowed
    if unexpected:
        errors.append("Undeclared native step models: " + ", ".join(sorted(unexpected)))

    for name, step in declared_by_name.items():
        spec = workflow_step_spec_for_step_name(name)
        if spec is None:
            continue
        step_meta = getattr(step, "__consist_step__", None)
        schema_outputs = (
            getattr(step_meta, "schema_outputs", ()) if step_meta is not None else ()
        )
        if schema_outputs is not None and not set(spec.output_keys).issubset(
            set(schema_outputs)
        ):
            errors.append(
                f"Step '{name}' metadata schema outputs omit catalog outputs "
                + ", ".join(sorted(set(spec.output_keys) - set(schema_outputs)))
            )

    if errors:
        raise RuntimeError(
            "Workflow step contract validation failed:\n- " + "\n- ".join(errors)
        )
