from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, TYPE_CHECKING


from pilates.config.models import PilatesConfig
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils.consist_types import (
    CouplerProtocol,
    ScenarioWithCoupler,
)
from pilates.utils.coupler_helpers import (
    _emit_artifact_lifecycle_event,
    artifact_to_existing_path,
    set_coupler_from_artifact,
)
from pilates.utils.formatting import formatted_print
from pilates.beam.launch_config import BeamLaunchConfig, build_beam_launch_config
from pilates.beam.config_hocon import beam_config_root
from pilates.beam.launch_paths import prepare_r5_raw_rebuild
from pilates.workflows.coupler_namespace import coupler_storage_value
from consist import Artifact, RunResult, Tracker
from consist.api import archive_run_output_files
from consist.protocols import ArtifactRecordLike
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.steps import (
    beam_full_skim,
    beam_postprocess,
    beam_preprocess,
    beam_run,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.artifact_keys import (
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


_FAIL_AFTER_BEAM_RUN_ENV = "PILATES_FAIL_AFTER_BEAM_RUN"
_BEAM_VEHICLES_IN = "vehicles_beam_in"


@dataclass
class TrafficAssignmentPhaseInputs:
    """
    Inputs for one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    year : int
        Forecast year being simulated.
    iteration : int
        Supply-demand iteration index for the year.
    activity_demand_outputs : Optional[dict[str, Any]]
        ActivitySim outputs used to seed BEAM inputs for this iteration.
    previous_beam_outputs : Optional[dict[str, Any]]
        Prior BEAM outputs (e.g., linkstats) used for warm-starting.
    """

    year: int
    iteration: int


@dataclass
class TrafficAssignmentPhaseOutputs:
    """
    Outputs from one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    previous_beam_outputs : Optional[dict[str, Any]]
        BEAM postprocess outputs for warm-starting the next iteration, if
        available. BEAM-run artifacts remain in the pinned postprocess closure
        and coupler rather than this public handoff.
    """

    pass


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = settings.beam
    skim_cfg = beam_cfg.full_skim if beam_cfg is not None else None
    if skim_cfg is None:
        return "disabled"
    return skim_cfg.run_schedule


def _stringify_mapping_values(mapping: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    return {str(key): str(value) for key, value in dict(mapping or {}).items()}


def _mapping_from_runtime_attr(owner: Any, name: str) -> Optional[Mapping[str, Any]]:
    value = getattr(owner, name, None)
    if callable(value):
        try:
            value = value()
        except TypeError:
            return None
    if isinstance(value, Mapping):
        return value
    return None


def _beam_restart_identity_context(
    *,
    scenario: Optional[Any] = None,
    state: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Read optional original-vs-resumed cache identity diagnostics from runtime state.

    Consist already exposes cache-miss and identity summaries on run metadata.
    When a restart harness or scenario wrapper has captured those summaries,
    include them here instead of inventing a new Consist API.
    """
    context: Dict[str, Any] = {}
    for owner in (scenario, state):
        if owner is None:
            continue
        for attr_name in (
            "beam_restart_binding_context",
            "beam_restart_identity_context",
        ):
            attr_context = _mapping_from_runtime_attr(owner, attr_name)
            if attr_context:
                context.update(attr_context)
        for attr_name, context_key in (
            ("cache_miss_explanation", "cache_miss_explanation"),
            ("latest_cache_miss_explanation", "cache_miss_explanation"),
            ("beam_cache_miss_explanation", "cache_miss_explanation"),
            ("identity_summary", "identity_summary"),
            ("latest_identity_summary", "identity_summary"),
            ("beam_identity_summary", "identity_summary"),
        ):
            attr_context = _mapping_from_runtime_attr(owner, attr_name)
            if attr_context and context_key not in context:
                context[context_key] = attr_context
    return context


def beam_preprocess_binding_diagnostic_payload(
    *,
    binding: ResolvedStepInputs,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    scenario: Optional[Any] = None,
    identity_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize the resumed BEAM binding surface before cache hashes are trusted."""
    beam_region_input_dir = None
    beam_primary_config_file = None
    try:
        beam_region_input_dir = os.path.join(
            workspace.get_beam_mutable_data_dir(),
            settings.run.region,
        )
        beam_config = settings.beam.config if settings.beam is not None else None
        if beam_config:
            beam_primary_config_file = os.path.join(
                beam_region_input_dir,
                beam_config,
            )
    except Exception:
        pass

    required_local_inputs = {
        "beam_region_input_dir": beam_region_input_dir,
        "beam_primary_config_file": beam_primary_config_file,
    }
    missing_local_inputs = sorted(
        key
        for key, path in required_local_inputs.items()
        if path and not os.path.exists(path)
    )
    missing_required = [
        role
        for role in binding.required_roles
        if binding.source_by_role.get(role) in {None, "missing"}
    ]
    missing_restart_inputs = sorted(set(missing_required) | set(missing_local_inputs))
    resolved_identity_context = dict(
        identity_context
        if identity_context is not None
        else _beam_restart_identity_context(scenario=scenario, state=state)
    )
    cache_miss_explanation = resolved_identity_context.get("cache_miss_explanation")
    identity_summary = resolved_identity_context.get("identity_summary")
    if not isinstance(cache_miss_explanation, Mapping):
        cache_miss_explanation = None
    if not isinstance(identity_summary, Mapping):
        identity_summary = None
    drift_components: Dict[str, Any] = {}
    if cache_miss_explanation:
        for key in (
            "mismatched_components",
            "config_keys_changed",
            "adapter_identity_changed",
            "identity_inputs_changed",
            "input_keys_changed",
            "missing_input_keys",
        ):
            value = cache_miss_explanation.get(key)
            if value:
                drift_components[key] = value
    if missing_restart_inputs:
        drift_classification = "missing_restart_inputs"
    elif drift_components:
        drift_classification = "content_or_config_drift"
    elif cache_miss_explanation:
        drift_classification = "cache_miss_without_binding_gap"
    else:
        drift_classification = "binding_surface_complete"
    return {
        "key": "beam_restart_binding",
        "artifact_family": "beam_restart_diagnostic",
        "diagnostic": "beam_restart_binding",
        "restart_run": bool(state.is_restart_run),
        "workflow_year": state.year,
        "forecast_year": state.forecast_year,
        "iteration": state.iteration,
        "input_keys": sorted(binding.required_roles),
        "optional_input_keys": sorted(binding.optional_roles),
        "bound_input_keys": sorted((binding.binding.inputs or {}).keys()),
        "missing_required": sorted(missing_required),
        "missing_restart_inputs": missing_restart_inputs,
        "source_by_key": dict(sorted((binding.source_by_role).items())),
        "coupler_key_by_key": dict(sorted((binding.selected_key_by_role).items())),
        "required_local_inputs": _stringify_mapping_values(required_local_inputs),
        "missing_local_inputs": missing_local_inputs,
        "identity_summary": dict(identity_summary or {}),
        "cache_miss_explanation": dict(cache_miss_explanation or {}),
        "cache_miss_reason": (
            cache_miss_explanation.get("reason") if cache_miss_explanation else None
        ),
        "identity_drift_components": drift_components,
        "drift_classification": drift_classification,
    }


def _emit_beam_preprocess_binding_diagnostic(
    *,
    binding: ResolvedStepInputs,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    scenario: Optional[Any] = None,
) -> None:
    if not state.is_restart_run:
        return
    payload = beam_preprocess_binding_diagnostic_payload(
        binding=binding,
        state=state,
        settings=settings,
        workspace=workspace,
        scenario=scenario,
    )
    logger.info(
        "[BEAM][restart] preprocess binding diagnostic: classification=%s missing_restart_inputs=%s bound_input_keys=%s",
        payload["drift_classification"],
        payload["missing_restart_inputs"],
        payload["bound_input_keys"],
    )
    _emit_artifact_lifecycle_event("beam_restart_binding", **payload)


def _raise_if_restart_beam_config_missing(
    *,
    binding: ResolvedStepInputs,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
) -> None:
    if not state.is_restart_run:
        return

    try:
        from pilates.beam.config_hocon import beam_primary_config_path

        expected_path = Path(beam_primary_config_path(settings, workspace=workspace))
        expected = str(expected_path)
    except Exception:
        expected_path = None
        expected = "<unresolved>"
    if expected_path is not None and expected_path.exists():
        return

    binding_inputs = binding.binding.inputs or {}
    config_value = binding_inputs.get(BEAM_CONFIG_FILE)
    config_hint = f" Resolved binding value: {config_value}." if config_value else ""
    raise RuntimeError(
        "BEAM restart cannot continue because beam_config_file is missing. "
        f"Expected primary config at {expected}. This must be restored or "
        f"bootstrapped before BEAM can run.{config_hint}"
    )


def _should_run_full_skim(settings: PilatesConfig, iteration: int) -> bool:
    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        return True
    if schedule == "after_each_iteration":
        return True
    if schedule == "after_final_iteration":
        total_iters = settings.run.supply_demand_iters
        return iteration == total_iters - 1
    return False


def _is_iteration_scoped_artifact_key(
    key: str, *, prefix: str, year: int, iteration: int
) -> bool:
    base = f"{prefix}_{year}_{iteration}"
    return key == base or key.startswith(f"{base}_sub")


def _build_beam_postprocess_input_keys(
    *,
    upstream_keys: Iterable[str],
    year: int,
    iteration: int,
    include_zarr_skims: bool,
) -> Optional[list[str]]:
    """
    Select BEAM postprocess coupler inputs from BEAM run outputs.

    BEAM postprocess only consumes BEAM events parquet and OD skims artifacts
    from the run output store, plus upstream ActivitySim ``zarr_skims`` when
    available. Trimming input keys to this set keeps run identity aligned with
    actual behavior while avoiding unnecessary cache invalidation from unrelated
    BEAM outputs.
    """
    selected: list[str] = []
    keys = list(upstream_keys)

    for key in keys:
        if _is_iteration_scoped_artifact_key(
            key, prefix="events_parquet", year=year, iteration=iteration
        ):
            selected.append(key)
            continue
        if _is_iteration_scoped_artifact_key(
            key, prefix="raw_od_skims", year=year, iteration=iteration
        ):
            selected.append(key)
            continue
        if _is_iteration_scoped_artifact_key(
            key, prefix="raw_od_skims_zarr", year=year, iteration=iteration
        ):
            selected.append(key)

    # Conservative fallback for naming drift: keep skim/event dependencies if
    # exact iteration-scoped keys are absent.
    if not any(key.startswith("raw_od_skims") for key in selected):
        selected.extend(key for key in keys if key.startswith("raw_od_skims"))
    if not any(key.startswith("events_parquet_") for key in selected):
        selected.extend(key for key in keys if key.startswith("events_parquet_"))

    if include_zarr_skims:
        selected.append(ZARR_SKIMS)

    deduped = list(dict.fromkeys(selected))
    for prefix in ("events_parquet", "raw_od_skims", "raw_od_skims_zarr"):
        final_key = f"{prefix}_{year}_{iteration}"
        if final_key in deduped:
            deduped = [key for key in deduped if not key.startswith(f"{final_key}_sub")]
    return deduped or None


def _collect_previous_beam_outputs(
    *,
    coupler: CouplerProtocol,
    workspace: Workspace,
    state: WorkflowState,
    iteration: int,
    previous_beam_outputs: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Resolve previous BEAM outputs for warm-starting.

    When explicit previous outputs are unavailable, this attempts to hydrate
    a minimal promoted store from coupler keys written by BEAM postprocess.
    """
    _ = state, iteration
    if previous_beam_outputs is not None:
        return previous_beam_outputs

    if not isinstance(coupler, (CouplerProtocol, Mapping)):
        return None

    promoted_outputs: Dict[str, Any] = {}
    for key in (LINKSTATS_WARMSTART, LINKSTATS, BEAM_PLANS_OUT):
        value = coupler_storage_value(coupler, key)
        if value is None:
            continue
        if artifact_to_existing_path(value, workspace):
            promoted_outputs[key] = value
    return promoted_outputs or None


def _archive_run_dir_for_restart(state: WorkflowState) -> Optional[Path]:
    run_info_path = state.run_info_path
    if not run_info_path:
        archive_dir = state.archive_run_dir
        return Path(archive_dir).expanduser().resolve() if archive_dir else None
    try:
        return Path(run_info_path).expanduser().resolve().parent
    except Exception:
        return None


def _archive_beam_run_outputs_native(
    *,
    run_id: str,
    tracker: Tracker,
    recovery_root: Path,
    outputs: Mapping[str, Any],
) -> None:
    """Archive beam step output files and directory artifacts to recovery_root."""
    from pilates.runtime.run_output_archive import _snapshot_directory_artifact

    file_keys: list[str] = []
    dir_keys: list[str] = []
    for key, artifact in outputs.items():
        if isinstance(artifact, (Artifact, ArtifactRecordLike)):
            if artifact.driver == "artifact_set":
                continue
            if (
                isinstance(artifact.meta, dict)
                and artifact.meta.get("directory_artifact") is True
            ):
                dir_keys.append(key)
                continue
            file_keys.append(key)

    if isinstance(tracker, Tracker):
        persisted_outputs = tracker.get_run_outputs(run_id)
        ws_mount = tracker.mounts.get("workspace")
        if ws_mount:
            ws_path = Path(ws_mount).resolve()
            for key in [*file_keys, *dir_keys]:
                artifact = persisted_outputs.get(key) or outputs.get(key)
                if isinstance(artifact, Artifact):
                    tracker.set_artifact_recovery_roots(
                        artifact, [ws_path], append=True
                    )
                    existing_artifact = outputs.get(key)
                    if (
                        isinstance(existing_artifact, Artifact)
                        and isinstance(existing_artifact.meta, dict)
                        and isinstance(artifact.meta, dict)
                    ):
                        existing_artifact.meta.update(artifact.meta)
        resolved_recovery_root = recovery_root.resolve()
        for key in dir_keys:
            artifact = persisted_outputs.get(key) or outputs.get(key)
            if isinstance(artifact, Artifact):
                _snapshot_directory_artifact(
                    tracker=tracker,
                    artifact=artifact,
                    recovery_root=resolved_recovery_root,
                )
                existing_artifact = outputs.get(key)
                if (
                    isinstance(existing_artifact, Artifact)
                    and isinstance(existing_artifact.meta, dict)
                    and isinstance(artifact.meta, dict)
                ):
                    existing_artifact.meta.update(artifact.meta)

    if file_keys or not dir_keys:
        report = archive_run_output_files(
            run_id=run_id,
            recovery_root=recovery_root,
            tracker=tracker,
            keys=file_keys if file_keys else None,
            preserve_existing=True,
            verify=True,
            append=True,
        )
        if not report.complete:
            failures = [
                f"{key}: copy={f.copy_status}, verify={f.verification_status}, msg={f.message}"
                for key, f in report.items()
                if not (
                    f.metadata_committed
                    and f.verification_status in {"verified", "not_requested"}
                )
            ]
            raise RuntimeError(
                "Archiving beam_run outputs failed verification or copy: "
                + "; ".join(failures)
            )

        for key, archived_file in report.items():
            existing_artifact = outputs.get(key)
            if (
                isinstance(existing_artifact, Artifact)
                and isinstance(existing_artifact.meta, dict)
                and isinstance(archived_file.artifact.meta, dict)
            ):
                existing_artifact.meta.update(archived_file.artifact.meta)


def _derive_beam_run_input_keys(
    *,
    beam_preprocess_inputs: Mapping[str, Any],
    activity_demand_outputs: Optional[Dict[str, Any]],
) -> list[str]:
    """
    Derive BEAM run input keys from preprocess outputs and warm-start signals.

    beam_preprocess always publishes the canonical plans/households/persons trio,
    regardless of whether they came from ActivitySim outputs or from existing
    default files in the copied BEAM scenario directory.
    """
    _ = activity_demand_outputs
    run_input_keys = [
        BEAM_CONFIG_FILE,
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
    ]

    # Only require LINKSTATS_WARMSTART at BEAM run time when that explicit key
    # is provided to preprocess. Other linkstats* artifacts may exist for
    # bookkeeping/history but do not guarantee a warm-start input artifact.
    if LINKSTATS_WARMSTART in beam_preprocess_inputs:
        run_input_keys.append(LINKSTATS_WARMSTART)
    else:
        logger.debug(
            "[BEAM] linkstats warmstart not available; omitting %s from inputs",
            LINKSTATS_WARMSTART,
        )

    return run_input_keys


def _maybe_fail_after_beam_run_for_canary(*, year: int, iteration: int) -> None:
    """Inject a controlled restart-canary failure after BEAM run completion."""
    if os.environ.get(_FAIL_AFTER_BEAM_RUN_ENV) != "1":
        return

    message = (
        "Injected failure after completed beam_run for restart canary "
        f"({_FAIL_AFTER_BEAM_RUN_ENV}=1, year={year}, iteration={iteration}). "
        "Unset the environment variable before restarting from run_state.yaml."
    )
    logger.error(message)
    raise RuntimeError(message)


def _run_beam_preprocess_step(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    year: int,
    iteration: int,
) -> Any:
    """Execute the native BEAM preprocess definition exactly once."""

    _, outputs = execute_step(
        scenario=scenario,
        definition=beam_preprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="preprocess",
    )
    return outputs


def _compile_beam_launch_config(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    workspace: Workspace,
    year: int,
    iteration: int,
    preprocess_outputs: Any,
) -> BeamLaunchConfig:
    """Create the one derived config tree consumed by the next BEAM process."""

    source_root = beam_config_root(settings, workspace=workspace)
    output_dir = (
        Path(workspace.full_path)
        / ".pilates-beam-launch-config"
        / f"y{year}"
        / f"i{iteration}"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=output_dir,
        identity=scenario.tracker.identity,
        prepared_inputs=preprocess_outputs.prepared_inputs,
    )
    prepare_r5_raw_rebuild(
        settings=settings,
        workspace=workspace,
        config_root=launch_config.root,
    )
    return launch_config


def _run_beam_steps(
    *,
    scenario: ScenarioWithCoupler,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
) -> Optional[Dict[str, Any]]:
    """Dispatch native BEAM steps with Consist caching and verified archival."""

    settings = context.settings
    state = context.state
    workspace = context.workspace
    archive_run_dir = _archive_run_dir_for_restart(state)

    _, preprocess_outputs = execute_step(
        scenario=scenario,
        definition=beam_preprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="preprocess",
    )
    launch_config = _compile_beam_launch_config(
        scenario=scenario,
        settings=settings,
        workspace=workspace,
        year=year,
        iteration=iteration,
        preprocess_outputs=preprocess_outputs,
    )
    run_inputs = beam_run.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=scenario.coupler,
        launch_config=launch_config,
    )
    run_result, _ = execute_step(
        scenario=scenario,
        definition=beam_run,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="run",
        runtime_kwargs={"beam_launch_config": launch_config},
        resolved_inputs=run_inputs,
    )

    # Archive beam_run outputs natively with verify=True
    base_archive = (
        archive_run_dir
        if archive_run_dir is not None
        else Path(scenario.tracker.run_dir) / "archive"
    )
    if not getattr(run_result, "cache_hit", False):
        recovery_root = (
            base_archive
            if base_archive.name == run_result.run.id
            else base_archive / run_result.run.id
        )
        _archive_beam_run_outputs_native(
            run_id=run_result.run.id,
            tracker=scenario.tracker,
            recovery_root=recovery_root,
            outputs=run_result.outputs,
        )

    logger.info("[beam] completed native beam_run run_id=%s", run_result.run.id)
    _maybe_fail_after_beam_run_for_canary(year=year, iteration=iteration)

    postprocess_inputs = beam_postprocess.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=scenario.coupler,
        beam_run_outputs=run_result.outputs,
    )
    postprocess_inputs.require_complete()

    postprocess_result, postprocess_outputs = execute_step(
        scenario=scenario,
        definition=beam_postprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="postprocess",
        resolved_inputs=postprocess_inputs,
    )
    if isinstance(postprocess_result, RunResult) and not getattr(
        postprocess_result, "cache_hit", False
    ):
        postprocess_recovery_root = (
            base_archive
            if base_archive.name == postprocess_result.run.id
            else base_archive / postprocess_result.run.id
        )
        _archive_beam_run_outputs_native(
            run_id=postprocess_result.run.id,
            tracker=scenario.tracker,
            recovery_root=postprocess_recovery_root,
            outputs=postprocess_result.outputs,
        )

    combined_outputs = step_output_handoff_mapping(
        postprocess_outputs, coupler=scenario.coupler
    )
    if _should_run_full_skim(settings, iteration):
        full_skim_outputs = _run_beam_full_skim_step(
            scenario=scenario,
            year=year,
            iteration=iteration,
            context=context,
            launch_config=launch_config,
        )
        if full_skim_outputs is not None:
            combined_outputs.update(full_skim_outputs)
    return combined_outputs


def _run_beam_full_skim_step(
    *,
    scenario: ScenarioWithCoupler,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
    launch_config: BeamLaunchConfig,
) -> Optional[Dict[str, Any]]:
    """Execute full skims through its native definition."""

    full_skim_inputs = beam_full_skim.resolve_inputs(
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        coupler=scenario.coupler,
        launch_config=launch_config,
    )
    _, outputs = execute_step(
        scenario=scenario,
        definition=beam_full_skim,
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="full_skim",
        runtime_kwargs={"beam_launch_config": launch_config},
        resolved_inputs=full_skim_inputs,
    )
    handoff = step_output_handoff_mapping(outputs, coupler=scenario.coupler)
    result = {key: path for key, path, _description in outputs._iter_record_items()}
    result.update(handoff)
    return result


def _standalone_full_skim_preprocess_outputs(
    *,
    scenario: ScenarioWithCoupler,
    workspace: Workspace,
    settings: Optional[PilatesConfig] = None,
) -> Any:
    from pilates.beam.outputs import BeamPreprocessOutputs

    prepared: Dict[str, Path] = {}
    existing_warmstart = artifact_to_existing_path(
        coupler_storage_value(scenario.coupler, LINKSTATS_WARMSTART),
        workspace=workspace,
    )
    if existing_warmstart is not None:
        prepared[LINKSTATS_WARMSTART] = Path(existing_warmstart)
    if LINKSTATS_WARMSTART not in prepared:
        existing_linkstats = artifact_to_existing_path(
            coupler_storage_value(scenario.coupler, LINKSTATS),
            workspace=workspace,
        )
        if existing_linkstats is not None:
            prepared[LINKSTATS_WARMSTART] = Path(existing_linkstats)
    if LINKSTATS_WARMSTART not in prepared and settings is not None:
        from pilates.utils.beam_warmstart import resolve_initial_linkstats_path

        resolved_warmstart = resolve_initial_linkstats_path(settings, workspace)
        if resolved_warmstart is not None:
            prepared[LINKSTATS_WARMSTART] = Path(resolved_warmstart)
            if isinstance(scenario.coupler, dict):
                scenario.coupler[LINKSTATS_WARMSTART] = prepared[LINKSTATS_WARMSTART]
            else:
                set_coupler_from_artifact(
                    scenario.coupler,
                    LINKSTATS_WARMSTART,
                    prepared[LINKSTATS_WARMSTART],
                )
    if settings is not None:
        shared_cfg = getattr(settings, "shared", None)
        geography_cfg = getattr(shared_cfg, "geography", None)
        zones_cfg = getattr(geography_cfg, "zones", None)
        if zones_cfg is not None:
            from pilates.beam.preprocessor import _prepare_beam_zone_shapefile

            _prepare_beam_zone_shapefile(workspace, settings)
    return BeamPreprocessOutputs(
        beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
        prepared_inputs=prepared,
    )


def _run_traffic_assignment_phase(
    *,
    scenario: ScenarioWithCoupler,
    inputs: TrafficAssignmentPhaseInputs,
    context: WorkflowRuntimeContext,
) -> TrafficAssignmentPhaseOutputs:
    """
    Run BEAM for a single supply-demand iteration.

    This sequences native BEAM definitions for one supply-demand iteration.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state tracking iterations and sub-stage completion.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    inputs : TrafficAssignmentPhaseInputs
        Inputs required for this iteration.
    Returns
    -------
    TrafficAssignmentPhaseOutputs
        Combined BEAM outputs for warm-starting the next iteration.
    """
    settings = context.settings
    state = context.state
    workspace = context.workspace

    formatted_print("TRAFFIC ASSIGNMENT MODEL")

    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        preprocess_outputs = _standalone_full_skim_preprocess_outputs(
            scenario=scenario,
            workspace=workspace,
            settings=settings,
        )
        launch_config = _compile_beam_launch_config(
            scenario=scenario,
            settings=settings,
            workspace=workspace,
            year=inputs.year,
            iteration=inputs.iteration,
            preprocess_outputs=preprocess_outputs,
        )
        combined_beam_outputs = _run_beam_full_skim_step(
            scenario=scenario,
            year=inputs.year,
            iteration=inputs.iteration,
            context=context,
            launch_config=launch_config,
        )
    else:
        combined_beam_outputs = _run_beam_steps(
            scenario=scenario,
            year=inputs.year,
            iteration=inputs.iteration,
            context=context,
        )
    del combined_beam_outputs

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.traffic_assignment,
    )

    return TrafficAssignmentPhaseOutputs()
