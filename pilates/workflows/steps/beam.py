"""Native Consist definitions and resolvers for the BEAM workflow steps.

Each BEAM step resolves the semantic roles it needs once, executes through
Consist, and projects ``RunResult.outputs`` into typed outputs.  The module
publishes current-role outputs and output-only diagnostics without expanding
the cross-model handoff surface.  The sole mid-stage restart boundary is
``beam_run_completed -> beam_postprocess``; archive roots remain storage
metadata, not a second execution path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from consist import (
    BindingResult,
    CacheOptions,
    ExecutionOptions,
    define_step,
)

from pilates.beam.config_hocon import (
    beam_primary_config_path,
)
from pilates.beam.launch_paths import (
    validate_r5_execution_reference,
    validate_staged_linkstats_reference,
)
from pilates.beam.launch_config import BeamLaunchConfig
from pilates.beam.runner import BeamRunner
from pilates.config.models import PilatesConfig
from pilates.utils.coupler_helpers import (
    artifact_to_path,
)
from pilates.workflows.coupler_namespace import (
    coupler_storage_keys,
    coupler_storage_value,
)
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workflows.binding import (
    artifact_rules_for_step_name,
    resolve_artifact_roles,
)
from pilates.workflows.state_helpers import resolve_forecast_year
from pilates.workflows.output_projection import require_output
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.surface import build_enabled_workflow_surface
from pilates.workflows.step_consist_meta import consist_step_meta
from pilates.workflows.step_definition import StepDefinition
from pilates.workflows.outputs_base import ValidationContext
from pilates.workspace import Workspace

# Model-specific native step definitions for BEAM.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    BEAM_PLANS_OUT,
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)

logger = logging.getLogger(__name__)

_BEAM_INCLUDE_RE = re.compile(r'^\s*include\s+(?:"([^"]+)"|file\("([^"]+)"\))')
_BEAM_CONFIG_REFERENCE_MANIFEST = "__archive_manifest.json"
_ATLAS_VEHICLES2_BASENAME_RE = re.compile(r"^vehicles2_\d{4}\.csv(?:\.gz)?$")


def _primary_beam_config_path(
    settings: PilatesConfig,
    workspace: Workspace,
) -> Path:
    return beam_primary_config_path(settings, workspace=workspace)


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


# Native Consist step definitions -------------------------------------------------

#
# These values deliberately sit beside the legacy factories until the coordinated
# stage cutover.  They do not capture a holder or coupler: all semantic selection is
# completed by their resolver and all declared outputs are persisted by Consist.


def _path_from_output(
    *,
    outputs: Mapping[str, Any],
    step_name: str,
    key: str,
    declared_outputs: Mapping[str, Any],
    workspace: Any,
) -> Path:
    """Return one output from the current invocation's declared path map.

    Cache-hit artifacts retain their original URI by design.  The typed
    projector must therefore validate the deterministic current destination
    requested for this invocation rather than consulting artifact metadata.
    """

    require_output(outputs, step_name=step_name, key=key)
    try:
        destination = declared_outputs[key]
    except KeyError as exc:
        raise RuntimeError(
            f"{step_name} output {key!r} has no declared current destination."
        ) from exc
    if isinstance(destination, os.PathLike):
        path = Path(destination)
    elif isinstance(destination, str) and destination.startswith("workspace://"):
        workspace_root = getattr(workspace, "full_path", None)
        if workspace_root is None:
            raise RuntimeError(
                f"{step_name} output {key!r} cannot resolve its workspace destination."
            )
        path = Path(workspace_root) / destination[len("workspace://") :].lstrip("/")
    elif isinstance(destination, str) and "://" in destination:
        raise RuntimeError(
            f"{step_name} output {key!r} has a non-local declared destination."
        )
    elif isinstance(destination, str):
        path = Path(destination)
        if not path.is_absolute():
            workspace_root = getattr(workspace, "full_path", None)
            if workspace_root is None:
                raise RuntimeError(
                    f"{step_name} output {key!r} cannot resolve its relative destination."
                )
            path = Path(workspace_root) / path
    else:
        raise RuntimeError(
            f"{step_name} output {key!r} has an invalid declared destination."
        )

    if not path.exists():
        raise RuntimeError(
            f"{step_name} output {key!r} is missing at declared destination {path}."
        )
    return path


def _native_output_destination(
    *, root: Path, step_name: str, key: str, suffix: str
) -> Path:
    """Return the individually keyed current destination for one BEAM output."""

    return root / ".pilates-consist-outputs" / step_name / f"{key}{suffix}"


def _beam_preprocess_native_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    del settings, state
    selected = (
        tuple(resolved_inputs.metadata.get("native_output_keys", ()))
        if resolved_inputs is not None
        else (
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            LINKSTATS_WARMSTART,
            "vehicles_beam_in",
        )
    )
    suffixes = {
        BEAM_PLANS_IN: ".csv",
        BEAM_HOUSEHOLDS_IN: ".csv",
        BEAM_PERSONS_IN: ".csv",
        LINKSTATS_WARMSTART: ".csv.gz",
        "vehicles_beam_in": ".csv",
    }
    root = Path(workspace.get_beam_mutable_data_dir())
    return {
        key: _native_output_destination(
            root=root,
            step_name="beam_preprocess",
            key=key,
            suffix=suffixes.get(key, ""),
        )
        for key in selected
    }


def _beam_run_native_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    del settings, resolved_inputs
    year = resolve_forecast_year(state)
    if year is None:
        raise RuntimeError("beam_run requires a resolved forecast year.")
    iteration = int(state.iteration)
    keys_and_suffixes = {
        LINKSTATS: ".csv.gz",
        BEAM_PLANS_OUT: ".csv.gz",
        f"raw_od_skims_{year}_{iteration}": ".omx",
        f"raw_od_skims_zarr_{year}_{iteration}": ".zarr",
        f"events_parquet_{year}_{iteration}": ".parquet",
    }
    root = Path(workspace.get_beam_output_dir())
    return {
        key: _native_output_destination(
            root=root,
            step_name="beam_run",
            key=key,
            suffix=suffix,
        )
        for key, suffix in keys_and_suffixes.items()
    }


def _beam_postprocess_native_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    from pilates.beam.postprocessor import BeamPostprocessor

    static_outputs = {
        key: Path(path)
        for key, path in BeamPostprocessor.expected_outputs(
            settings, state, workspace
        ).items()
        if path is not None
    }
    if resolved_inputs is None:
        return static_outputs
    dynamic_outputs = resolved_inputs.metadata.get("beam_postprocess_output_paths", {})
    if not isinstance(dynamic_outputs, Mapping):
        raise RuntimeError(
            "beam_postprocess resolved output paths must be a key-to-path mapping."
        )
    duplicate_keys = set(static_outputs).intersection(dynamic_outputs)
    if duplicate_keys:
        raise RuntimeError(
            "beam_postprocess resolved output keys overlap static outputs: "
            + ", ".join(sorted(duplicate_keys))
        )
    resolved_output_paths = {
        **static_outputs,
        **{key: Path(path) for key, path in dynamic_outputs.items()},
    }
    if len(set(resolved_output_paths.values())) != len(resolved_output_paths):
        raise RuntimeError("beam_postprocess resolved output paths are not injective.")
    return resolved_output_paths


def _beam_full_skim_native_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    del resolved_inputs
    from pilates.beam.runner import BeamFullSkimRunner

    return {
        key: Path(path)
        for key, path in BeamFullSkimRunner.expected_outputs(
            settings, state, workspace
        ).items()
        if path is not None
    }


def _materialize_native_outputs(
    *,
    source_paths: Mapping[str, Path],
    declared_outputs: Mapping[str, Path],
) -> None:
    """Copy selected semantic outputs to their declared individual paths."""

    for key, destination in declared_outputs.items():
        source = source_paths.get(key)
        if source is None:
            continue
        source = Path(source)
        if not source.exists():
            raise RuntimeError(
                f"BEAM native output {key!r} is missing before declared output logging: "
                f"{source}."
            )
        if source.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        elif source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def _beam_run_native_sources(produced: BeamRunOutputs) -> dict[str, Path]:
    """Select the closed BEAM run semantic surface from raw runner outputs."""

    sources = dict(produced.raw_outputs)
    latest_linkstats = produced._latest_raw_output_for_prefix(LINKSTATS)
    if latest_linkstats is None:
        latest_linkstats = produced._latest_raw_output_for_prefix("linkstats_parquet")
    if latest_linkstats is not None:
        sources[LINKSTATS] = latest_linkstats[1]
    latest_plans = produced._latest_raw_output_for_prefix(BEAM_PLANS_OUT)
    if latest_plans is not None:
        sources[BEAM_PLANS_OUT] = latest_plans[1]
    return sources


def _beam_postprocess_native_sources(
    produced: BeamPostprocessOutputs,
) -> dict[str, Path]:
    """Return every typed BEAM postprocess output by its semantic key."""

    sources = {
        **produced.split_events,
        **produced.split_event_links,
    }
    if produced.zarr_skims is not None:
        sources[ZARR_SKIMS] = produced.zarr_skims
    if produced.final_skims_omx is not None:
        sources["final_skims_omx"] = produced.final_skims_omx
    return sources


def _validate_native_outputs(
    outputs: Any,
    *,
    step_name: str,
    settings: Any,
    state: Any,
    workspace: Any,
) -> None:
    outputs.validate(
        context=ValidationContext(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name=step_name,
        )
    )


def _project_beam_preprocess_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> BeamPreprocessOutputs:
    declared_outputs = _beam_preprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    prepared = {
        key: _path_from_output(
            outputs=outputs,
            step_name="beam_preprocess",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in (
            *BeamPreprocessOutputs.required_output_keys(),
            "vehicles_beam_in",
            LINKSTATS_WARMSTART,
        )
        if key in outputs and key in declared_outputs
    }
    projected = BeamPreprocessOutputs(
        beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
        prepared_inputs=prepared,
    )
    _validate_native_outputs(
        projected,
        step_name="beam_preprocess",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return projected


def _project_beam_run_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> BeamRunOutputs:
    declared_outputs = _beam_run_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    raw_outputs = {
        key: _path_from_output(
            outputs=outputs,
            step_name="beam_run",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in outputs
        if key in declared_outputs
    }
    projected = BeamRunOutputs(
        beam_output_dir=Path(workspace.get_beam_output_dir()),
        raw_outputs=raw_outputs,
    )
    _validate_native_outputs(
        projected,
        step_name="beam_run",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return projected


def _project_beam_postprocess_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> BeamPostprocessOutputs:
    dynamic_output_keys = {
        key
        for key in outputs
        if (
            key.startswith("events_parquet_")
            and "_type_" in key
            or key.startswith("path_traversal_links_")
        )
    }
    resolved_dynamic_outputs = resolved_inputs.metadata.get(
        "beam_postprocess_output_paths", {}
    )
    if not isinstance(resolved_dynamic_outputs, Mapping):
        raise RuntimeError(
            "beam_postprocess resolved output paths must be a key-to-path mapping."
        )
    if dynamic_output_keys != set(resolved_dynamic_outputs):
        raise RuntimeError(
            "beam_postprocess persisted typed output keys differ from its resolved "
            "closed output map: expected "
            f"{sorted(resolved_dynamic_outputs)}, got {sorted(dynamic_output_keys)}."
        )
    declared_outputs = _beam_postprocess_native_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved_inputs,
    )
    projected = BeamPostprocessOutputs(
        zarr_skims=(
            _path_from_output(
                outputs=outputs,
                step_name="beam_postprocess",
                key=ZARR_SKIMS,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            if ZARR_SKIMS in outputs and ZARR_SKIMS in declared_outputs
            else None
        ),
        final_skims_omx=(
            _path_from_output(
                outputs=outputs,
                step_name="beam_postprocess",
                key="final_skims_omx",
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            if "final_skims_omx" in outputs and "final_skims_omx" in declared_outputs
            else None
        ),
        split_events={
            key: _path_from_output(
                outputs=outputs,
                step_name="beam_postprocess",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in outputs
            if key in declared_outputs
            and key.startswith("events_parquet_")
            and "_type_" in key
        },
        split_event_links={
            key: _path_from_output(
                outputs=outputs,
                step_name="beam_postprocess",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in outputs
            if key in declared_outputs and key.startswith("path_traversal_links_")
        },
    )
    _validate_native_outputs(
        projected,
        step_name="beam_postprocess",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return projected


def _project_beam_full_skim_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> BeamFullSkimOutputs:
    declared_outputs = _beam_full_skim_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    projected = BeamFullSkimOutputs(
        full_skims=_path_from_output(
            outputs=outputs,
            step_name="beam_full_skim",
            key="beam_full_skims",
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
    )
    _validate_native_outputs(
        projected,
        step_name="beam_full_skim",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return projected


def _input_destination(*, workspace: Any, key: str, source: Any) -> Path:
    source_path = artifact_to_path(source, workspace=workspace)
    if key == ATLAS_VEHICLES2_OUTPUT:
        source_name = Path(source_path).name if source_path is not None else ""
        if not _ATLAS_VEHICLES2_BASENAME_RE.fullmatch(source_name):
            raise ValueError(
                "BEAM preprocess requires a year-qualified ATLAS vehicles2 "
                f"source filename, got {source_path!r}."
            )
        destination_name = source_name
    else:
        suffixes = "".join(Path(source_path).suffixes) if source_path else ""
        destination_name = f"{key}{suffixes}"
    return (
        Path(workspace.get_beam_mutable_data_dir())
        / ".consist-inputs"
        / destination_name
    )


def _resolved_beam_inputs(
    *,
    step_name: str,
    coupler: Any,
    workspace: Any,
    required_roles: Iterable[str],
    optional_roles: Iterable[str] = (),
    explicit_inputs: Mapping[str, Any] | None = None,
    logical_destinations: Mapping[str, Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ResolvedStepInputs:
    inputs: dict[str, Any] = dict(explicit_inputs or {})
    source_by_role = {key: "explicit" for key in inputs}
    selected_key_by_role = {key: key for key in inputs}
    destinations = dict(logical_destinations or {})
    for key in (*required_roles, *optional_roles):
        if key in inputs:
            continue
        value = coupler_storage_value(coupler, key)
        if value is None:
            source_by_role[key] = "missing"
            continue
        inputs[key] = value
        source_by_role[key] = "coupler"
        selected_key_by_role[key] = key
        destinations.setdefault(
            key,
            _input_destination(workspace=workspace, key=key, source=value),
        )
    return ResolvedStepInputs(
        step_name=step_name,
        binding=BindingResult(inputs=inputs),
        required_roles=tuple(required_roles),
        optional_roles=tuple(optional_roles),
        source_by_role=source_by_role,
        selected_key_by_role=selected_key_by_role,
        logical_destinations=destinations,
        metadata=metadata or {},
    )


def _native_execution_options(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> ExecutionOptions:
    del settings, state, workspace
    runtime_kwargs: dict[str, Any] = {}
    if "beam_postprocess_dynamic_paths" in resolved_inputs.metadata:
        runtime_kwargs["beam_run_dynamic_paths"] = dict(
            resolved_inputs.metadata["beam_postprocess_dynamic_paths"]
        )
    if "beam_postprocess_output_paths" in resolved_inputs.metadata:
        runtime_kwargs["beam_postprocess_output_paths"] = dict(
            resolved_inputs.metadata["beam_postprocess_output_paths"]
        )
    return ExecutionOptions(
        input_binding="paths",
        input_materialization="requested",
        input_paths=resolved_inputs.logical_destinations,
        runtime_kwargs=runtime_kwargs,
        inject_context="_consist_ctx",
    )


def _native_contract_output_paths(
    provider: Callable[..., Mapping[str, Any]],
) -> Callable[[Any], Mapping[str, Any]]:
    def resolve(context: Any) -> Mapping[str, Any]:
        settings = context.get_runtime("settings", default=None)
        state = context.get_runtime("state", default=None)
        workspace = context.get_runtime("workspace", default=None)
        if settings is None or state is None or workspace is None:
            return {}
        return provider(settings=settings, state=state, workspace=workspace)

    return resolve


def _strict_requested_output_cache(
    *, settings: Any, state: Any, workspace: Any
) -> CacheOptions:
    del settings, state, workspace
    return CacheOptions(
        cache_hydration="outputs-requested",
        cache_hydration_failure="miss",
    )


def _resolve_beam_preprocess_inputs(
    *, settings: Any, state: Any, workspace: Any, coupler: Any
) -> ResolvedStepInputs:
    config_path = _require_primary_beam_config(settings, workspace)
    surface = build_enabled_workflow_surface(settings)
    requires_atlas_vehicles = (
        surface.profile.vehicle_ownership_model_enabled
        and getattr(state, "current_inner_iter", 0) == 0
    )
    required_roles = (
        BEAM_CONFIG_FILE,
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
        *((ATLAS_VEHICLES2_OUTPUT,) if requires_atlas_vehicles else ()),
    )
    optional_roles = (
        LINKSTATS_WARMSTART,
        *((ATLAS_VEHICLES2_OUTPUT,) if not requires_atlas_vehicles else ()),
    )
    resolved = resolve_artifact_roles(
        step_name="beam_preprocess",
        coupler=coupler,
        settings=settings,
        state=state,
        workspace=workspace,
        required_roles=required_roles,
        optional_roles=optional_roles,
        artifact_rules=artifact_rules_for_step_name(
            "beam_preprocess", settings=settings
        ),
        explicit_inputs={BEAM_CONFIG_FILE: config_path},
        logical_destinations={BEAM_CONFIG_FILE: config_path},
        year=resolve_forecast_year(state),
        surface=surface,
    )
    logical_destinations = dict(resolved.logical_destinations)
    for key, source in (resolved.binding.inputs or {}).items():
        logical_destinations.setdefault(
            key,
            _input_destination(workspace=workspace, key=key, source=source),
        )
    return ResolvedStepInputs(
        step_name=resolved.step_name,
        binding=resolved.binding,
        required_roles=resolved.required_roles,
        optional_roles=resolved.optional_roles,
        source_by_role=resolved.source_by_role,
        selected_key_by_role=resolved.selected_key_by_role,
        logical_destinations=logical_destinations,
        metadata={
            "native_output_keys": tuple(
                key
                for key in (
                    BEAM_PLANS_IN,
                    BEAM_HOUSEHOLDS_IN,
                    BEAM_PERSONS_IN,
                    LINKSTATS_WARMSTART,
                    "vehicles_beam_in",
                )
                if key in (resolved.binding.inputs or {})
            )
        },
    )


def _resolve_beam_run_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: Any,
    launch_config: BeamLaunchConfig | None = None,
) -> ResolvedStepInputs:
    config_path = (
        launch_config.primary_config
        if launch_config is not None
        else _require_primary_beam_config(settings, workspace)
    )
    return _resolved_beam_inputs(
        step_name="beam_run",
        coupler=coupler,
        workspace=workspace,
        required_roles=(
            BEAM_CONFIG_FILE,
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ),
        optional_roles=(LINKSTATS_WARMSTART, ZARR_SKIMS),
        explicit_inputs={BEAM_CONFIG_FILE: config_path},
        logical_destinations={BEAM_CONFIG_FILE: config_path},
    )


def _postprocess_dynamic_keys(
    *, storage_keys: Iterable[str], year: int, iteration: int
) -> tuple[str, ...]:
    keys = tuple(storage_keys)
    selected = [
        key
        for key in keys
        if key.startswith(
            (f"events_parquet_{year}_{iteration}", f"raw_od_skims_{year}_{iteration}")
        )
    ]
    if not any(key.startswith("events_parquet_") for key in selected):
        selected.extend(key for key in keys if key.startswith("events_parquet_"))
    if not any(key.startswith("raw_od_skims") for key in selected):
        selected.extend(key for key in keys if key.startswith("raw_od_skims"))
    return tuple(dict.fromkeys(selected))


def _selected_postprocess_events_key(
    *, dynamic_keys: Iterable[str], year: int, iteration: int
) -> str | None:
    """Choose the exact events input consumed by ``BeamPostprocessor``.

    This mirrors the postprocessor's identity-bearing event selection: the
    canonical iteration key wins, otherwise its highest numeric sub-iteration
    is selected.  Older/fallback event keys remain valid inputs for other
    postprocess work but must not expand this invocation's output contract.
    """

    target = f"events_parquet_{year}_{iteration}"
    keys = tuple(dynamic_keys)
    if target in keys:
        return target
    selected: str | None = None
    selected_sub_iteration = -1
    for key in keys:
        if not key.startswith(f"{target}_sub"):
            continue
        suffix = key[len(f"{target}_sub") :]
        try:
            sub_iteration = int(suffix)
        except ValueError:
            continue
        if sub_iteration > selected_sub_iteration:
            selected = key
            selected_sub_iteration = sub_iteration
    return selected


def _beam_postprocess_split_output_paths(
    *,
    selected_events_key: str | None,
    inputs: Mapping[str, Any],
    year: int,
    iteration: int,
    workspace: Any,
) -> dict[str, Path]:
    """Close typed split outputs from the selected, locally readable event input.

    Event types are data-dependent, so the one semantic event input selected
    for this invocation is the only authority for the exact keyed output map.
    Refuse to run when that source cannot be inspected; accepting a partial
    output map would make a cache hit semantically different from a fresh run.
    """

    if selected_events_key is None:
        return {}
    source = inputs.get(selected_events_key)
    source_path = artifact_to_path(source, workspace=workspace)
    if source_path is None or "://" in source_path or not Path(source_path).is_file():
        raise RuntimeError(
            "beam_postprocess cannot inspect selected events input "
            f"{selected_events_key!r} to close typed outputs."
        )
    try:
        import pandas as pd

        event_types = pd.read_parquet(source_path, columns=["type"])["type"]
    except Exception as exc:
        raise RuntimeError(
            "beam_postprocess cannot inspect event types for selected input "
            f"{selected_events_key!r}: {source_path}."
        ) from exc
    if event_types.empty:
        return {}
    event_types = sorted({str(value) for value in event_types.dropna()})
    root = Path(workspace.get_beam_output_dir())
    event_keys = tuple(
        f"events_parquet_{year}_{iteration}_type_{_sanitize_beam_event_type(event_type)}"
        for event_type in event_types
    )
    if len(set(event_keys)) != len(event_keys):
        raise RuntimeError(
            "beam_postprocess selected event types do not map to injective "
            f"semantic output keys: {event_types}."
        )
    output_paths = {
        key: _native_output_destination(
            root=root,
            step_name="beam_postprocess",
            key=f"events_parquet_{year}_{iteration}_type_{_sanitize_beam_event_type(event_type)}",
            suffix=".parquet",
        )
        for key, event_type in zip(event_keys, event_types, strict=True)
    }
    if "PathTraversal" in event_types:
        key = f"path_traversal_links_{year}_{iteration}"
        output_paths[key] = _native_output_destination(
            root=root,
            step_name="beam_postprocess",
            key=key,
            suffix=".parquet",
        )
    if len(set(output_paths.values())) != len(output_paths):
        raise RuntimeError(
            "beam_postprocess resolved typed output paths are not injective."
        )
    return output_paths


def _sanitize_beam_event_type(event_type: str) -> str:
    """Match the postprocessor's semantic event-type key normalization."""

    safe = re.sub(r"[^A-Za-z0-9]+", "_", event_type).strip("_")
    return safe or "unknown"


def _postprocess_destination(*, key: str, workspace: Any, iteration: int) -> Path:
    output_dir = Path(workspace.get_beam_output_dir())
    if key.startswith("events_parquet_"):
        return output_dir / ".pilates-consist-inputs" / f"{key}.parquet"
    if key.startswith("raw_od_skims_zarr"):
        return output_dir / ".pilates-consist-inputs" / f"{key}.zarr"
    if key.startswith("raw_od_skims"):
        return output_dir / ".pilates-consist-inputs" / f"{key}.omx"
    raise RuntimeError(
        f"beam_postprocess has no deterministic destination for {key!r}."
    )


def _resolve_beam_postprocess_inputs(
    *, settings: Any, state: Any, workspace: Any, coupler: Any
) -> ResolvedStepInputs:
    year = resolve_forecast_year(state)
    if year is None:
        raise RuntimeError("beam_postprocess requires a resolved forecast year.")
    iteration = state.iteration
    storage_keys = coupler_storage_keys(coupler)
    dynamic_keys = _postprocess_dynamic_keys(
        storage_keys=storage_keys,
        year=int(year),
        iteration=int(iteration),
    )
    destinations = {
        key: _postprocess_destination(
            key=key, workspace=workspace, iteration=int(iteration)
        )
        for key in dynamic_keys
    }
    if (
        settings.activitysim is not None
        and coupler_storage_value(coupler, ZARR_SKIMS) is not None
    ):
        destinations[ZARR_SKIMS] = (
            Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
        )
        optional_roles = (ZARR_SKIMS,)
    else:
        optional_roles = ()
    dynamic_paths = {key: destinations[key] for key in dynamic_keys}
    resolved = _resolved_beam_inputs(
        step_name="beam_postprocess",
        coupler=coupler,
        workspace=workspace,
        required_roles=dynamic_keys,
        optional_roles=optional_roles,
        logical_destinations=destinations,
        metadata={"beam_postprocess_dynamic_paths": dynamic_paths},
    )
    selected_events_key = _selected_postprocess_events_key(
        dynamic_keys=dynamic_keys,
        year=int(year),
        iteration=int(iteration),
    )
    split_outputs = _beam_postprocess_split_output_paths(
        selected_events_key=selected_events_key,
        inputs=resolved.binding.inputs or {},
        year=int(year),
        iteration=int(iteration),
        workspace=workspace,
    )
    return ResolvedStepInputs(
        step_name=resolved.step_name,
        binding=resolved.binding,
        required_roles=resolved.required_roles,
        optional_roles=resolved.optional_roles,
        source_by_role=resolved.source_by_role,
        selected_key_by_role=resolved.selected_key_by_role,
        logical_destinations=resolved.logical_destinations,
        metadata={
            **resolved.metadata,
            "beam_postprocess_output_paths": MappingProxyType(split_outputs),
        },
    )


def _resolve_beam_full_skim_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: Any,
    launch_config: BeamLaunchConfig | None = None,
) -> ResolvedStepInputs:
    config_path = (
        launch_config.primary_config
        if launch_config is not None
        else _require_primary_beam_config(settings, workspace)
    )
    return _resolved_beam_inputs(
        step_name="beam_full_skim",
        coupler=coupler,
        workspace=workspace,
        required_roles=(
            BEAM_CONFIG_FILE,
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ),
        optional_roles=(LINKSTATS_WARMSTART,),
        explicit_inputs={BEAM_CONFIG_FILE: config_path},
        logical_destinations={BEAM_CONFIG_FILE: config_path},
    )


def _log_native_output_records(*, outputs: Any, context: Any) -> None:
    logged_keys: set[str] = set()
    for key, path, _description in outputs._iter_record_items():
        if key in logged_keys:
            continue
        logged_keys.add(key)
        context.log_output(
            path,
            key=key,
            artifact_kind="directory" if Path(path).is_dir() else "file",
        )


@define_step(
    model="beam_preprocess",
    name_template="beam_preprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={
        BEAM_CONFIG_FILE: None,
        BEAM_PLANS_IN: None,
        BEAM_HOUSEHOLDS_IN: None,
        BEAM_PERSONS_IN: None,
    },
    optional_input_keys=(LINKSTATS_WARMSTART, ATLAS_VEHICLES2_OUTPUT),
    schema_outputs=[
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
        LINKSTATS_WARMSTART,
        "vehicles_beam_in",
    ],
    output_paths=_native_contract_output_paths(_beam_preprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta("beam_preprocess"),
)
def _native_beam_preprocess(
    beam_config_file: Path,
    plans_beam_in: Path,
    households_beam_in: Path,
    persons_beam_in: Path,
    linkstats_warmstart: Path | None = None,
    atlas_vehicles2_output: Path | None = None,
    *,
    settings: Any,
    state: Any,
    workspace: Workspace,
    _consist_ctx: Any,
) -> None:
    if not beam_config_file.exists():
        raise FileNotFoundError(
            f"beam_preprocess config is missing: {beam_config_file}"
        )
    from pilates.beam.preprocessor import BeamPreprocessor

    inputs = {
        BEAM_PLANS_IN: plans_beam_in,
        BEAM_HOUSEHOLDS_IN: households_beam_in,
        BEAM_PERSONS_IN: persons_beam_in,
    }
    if linkstats_warmstart is not None:
        inputs[LINKSTATS_WARMSTART] = linkstats_warmstart
    if atlas_vehicles2_output is not None:
        inputs[ATLAS_VEHICLES2_OUTPUT] = atlas_vehicles2_output
    produced = BeamPreprocessor("beam_preprocess", state).preprocess(
        workspace,
        beam_preprocess_inputs=inputs,
    )
    _validate_native_outputs(
        produced,
        step_name="beam_preprocess",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    _materialize_native_outputs(
        source_paths=produced.prepared_inputs,
        declared_outputs=_beam_preprocess_native_output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
            resolved_inputs=ResolvedStepInputs(
                step_name="beam_preprocess",
                binding=BindingResult(inputs=inputs),
                metadata={"native_output_keys": tuple(produced.prepared_inputs)},
            ),
        ),
    )
    del _consist_ctx


@define_step(
    model="beam_run",
    name_template="beam_run__y{year}__i{iteration}__phase_{phase}",
    inputs={
        BEAM_CONFIG_FILE: None,
        BEAM_PLANS_IN: None,
        BEAM_HOUSEHOLDS_IN: None,
        BEAM_PERSONS_IN: None,
    },
    optional_input_keys=(LINKSTATS_WARMSTART, ZARR_SKIMS),
    schema_outputs=[LINKSTATS, BEAM_PLANS_OUT],
    output_paths=_native_contract_output_paths(_beam_run_native_output_paths),
    input_binding="paths",
    **consist_step_meta("beam_run"),
)
def _native_beam_run(
    beam_config_file: Path,
    plans_beam_in: Path,
    households_beam_in: Path,
    persons_beam_in: Path,
    linkstats_warmstart: Path | None = None,
    zarr_skims: Path | None = None,
    *,
    settings: Any,
    state: Any,
    workspace: Workspace,
    beam_launch_config: BeamLaunchConfig,
    _consist_ctx: Any,
) -> None:
    if not beam_config_file.exists():
        raise FileNotFoundError(f"beam_run config is missing: {beam_config_file}")
    if beam_config_file != beam_launch_config.primary_config:
        raise RuntimeError(
            "beam_run binding config must be the same derived config mounted by "
            "the runner."
        )
    validate_staged_linkstats_reference(
        settings=settings,
        workspace=workspace,
        run_context=_consist_ctx,
        config_root=beam_launch_config.root,
    )
    validate_r5_execution_reference(
        settings=settings,
        workspace=workspace,
        run_context=_consist_ctx,
        config_root=beam_launch_config.root,
    )
    prepared = {
        BEAM_PLANS_IN: plans_beam_in,
        BEAM_HOUSEHOLDS_IN: households_beam_in,
        BEAM_PERSONS_IN: persons_beam_in,
    }
    if linkstats_warmstart is not None:
        prepared[LINKSTATS_WARMSTART] = linkstats_warmstart
    produced = BeamRunner("beam_run", state).run(
        BeamPreprocessOutputs(
            beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
            prepared_inputs=prepared,
        ),
        workspace,
        launch_config=beam_launch_config,
        extra_inputs={ZARR_SKIMS: zarr_skims} if zarr_skims is not None else None,
    )
    _validate_native_outputs(
        produced,
        step_name="beam_run",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    _materialize_native_outputs(
        source_paths=_beam_run_native_sources(produced),
        declared_outputs=_beam_run_native_output_paths(
            settings=settings, state=state, workspace=workspace
        ),
    )
    del _consist_ctx


@define_step(
    model="beam_postprocess",
    name_template="beam_postprocess__y{year}__i{iteration}__phase_{phase}",
    optional_input_keys=(ZARR_SKIMS,),
    schema_outputs=[ZARR_SKIMS, "final_skims_omx"],
    output_paths=_native_contract_output_paths(_beam_postprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta("beam_postprocess"),
)
def _native_beam_postprocess(
    zarr_skims: Path | None = None,
    *,
    beam_run_dynamic_paths: Mapping[str, Path],
    beam_postprocess_output_paths: Mapping[str, Path],
    settings: Any,
    state: Any,
    workspace: Workspace,
    _consist_ctx: Any,
) -> None:
    from pilates.beam.postprocessor import BeamPostprocessor

    produced = BeamPostprocessor("beam_postprocess", state).postprocess(
        BeamRunOutputs(
            beam_output_dir=Path(workspace.get_beam_output_dir()),
            raw_outputs=dict(beam_run_dynamic_paths),
        ),
        workspace,
        zarr_skims=zarr_skims,
    )
    _validate_native_outputs(
        produced,
        step_name="beam_postprocess",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    declared_outputs = _beam_postprocess_native_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    overlap = set(declared_outputs).intersection(beam_postprocess_output_paths)
    if overlap:
        raise RuntimeError(
            "beam_postprocess resolved output keys overlap static outputs: "
            + ", ".join(sorted(overlap))
        )
    declared_outputs.update(
        {key: Path(path) for key, path in beam_postprocess_output_paths.items()}
    )
    produced_sources = _beam_postprocess_native_sources(produced)
    expected_dynamic_keys = set(beam_postprocess_output_paths)
    produced_dynamic_keys = {
        key
        for key in produced_sources
        if (
            key.startswith("events_parquet_")
            and "_type_" in key
            or key.startswith("path_traversal_links_")
        )
    }
    if produced_dynamic_keys != expected_dynamic_keys:
        raise RuntimeError(
            "beam_postprocess produced typed output keys differ from its resolved "
            "closed output map: expected "
            f"{sorted(expected_dynamic_keys)}, got {sorted(produced_dynamic_keys)}."
        )
    _materialize_native_outputs(
        source_paths=produced_sources,
        declared_outputs=declared_outputs,
    )
    del _consist_ctx


@define_step(
    model="beam_full_skim",
    name_template="beam_full_skim__y{year}__i{iteration}__phase_{phase}",
    inputs={
        BEAM_CONFIG_FILE: None,
        BEAM_PLANS_IN: None,
        BEAM_HOUSEHOLDS_IN: None,
        BEAM_PERSONS_IN: None,
    },
    optional_input_keys=(LINKSTATS_WARMSTART,),
    schema_outputs=["beam_full_skims"],
    output_paths=_native_contract_output_paths(_beam_full_skim_native_output_paths),
    input_binding="paths",
    **consist_step_meta("beam_full_skim"),
)
def _native_beam_full_skim(
    beam_config_file: Path,
    plans_beam_in: Path,
    households_beam_in: Path,
    persons_beam_in: Path,
    linkstats_warmstart: Path | None = None,
    *,
    settings: Any,
    state: Any,
    workspace: Workspace,
    beam_launch_config: BeamLaunchConfig,
    _consist_ctx: Any,
) -> None:
    if beam_config_file != beam_launch_config.primary_config:
        raise RuntimeError(
            "beam_full_skim binding config must be the same derived config mounted "
            "by the runner."
        )
    prepared = {
        BEAM_PLANS_IN: plans_beam_in,
        BEAM_HOUSEHOLDS_IN: households_beam_in,
        BEAM_PERSONS_IN: persons_beam_in,
    }
    if linkstats_warmstart is not None:
        prepared[LINKSTATS_WARMSTART] = linkstats_warmstart
    from pilates.beam.runner import BeamFullSkimRunner

    produced = BeamFullSkimRunner("beam_full_skim", state).run(
        BeamPreprocessOutputs(
            beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
            prepared_inputs=prepared,
        ),
        workspace,
        launch_config=beam_launch_config,
    )
    _validate_native_outputs(
        produced,
        step_name="beam_full_skim",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    del _consist_ctx


beam_preprocess = StepDefinition(
    name="beam_preprocess",
    function=_native_beam_preprocess,
    resolve_inputs=_resolve_beam_preprocess_inputs,
    project_outputs=_project_beam_preprocess_outputs,
    output_paths=_beam_preprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache,
)
beam_run = StepDefinition(
    name="beam_run",
    function=_native_beam_run,
    resolve_inputs=_resolve_beam_run_inputs,
    project_outputs=_project_beam_run_outputs,
    output_paths=_beam_run_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache,
)
beam_postprocess = StepDefinition(
    name="beam_postprocess",
    function=_native_beam_postprocess,
    resolve_inputs=_resolve_beam_postprocess_inputs,
    project_outputs=_project_beam_postprocess_outputs,
    output_paths=_beam_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache,
)
beam_full_skim = StepDefinition(
    name="beam_full_skim",
    function=_native_beam_full_skim,
    resolve_inputs=_resolve_beam_full_skim_inputs,
    project_outputs=_project_beam_full_skim_outputs,
    output_paths=_beam_full_skim_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache,
)
