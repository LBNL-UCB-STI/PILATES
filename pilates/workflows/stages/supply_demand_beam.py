from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

from pilates.config.models import PilatesConfig
from pilates.runtime.context import (
    WorkflowRuntimeContext,
    ensure_workflow_runtime_context,
)
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import (
    CouplerProtocol,
    ScenarioRestorationLike,
    ScenarioWithCoupler,
)
from pilates.utils.coupler_helpers import (
    _emit_artifact_lifecycle_event,
    artifact_to_existing_path,
    set_coupler_from_artifact,
)
from pilates.utils.formatting import formatted_print
from pilates.beam.outputs import BeamRunOutputs
from pilates.workflows.resume import (
    HistoricalOutputRequest,
    ResumeBoundaryPolicy,
    ResumeDecision,
    ResumeDisposition,
    ResumePlanningError,
    ResumeProjectionError,
    RestoreExecutionResult,
    build_resume_plan,
    execute_restore_decision,
)
from pilates.workflows.beam_checkpoint import (
    PinnedClosureMember,
    beam_checkpoint_record_present,
    beam_postprocess_is_in_progress,
    checkpoint_fingerprint_matches,
    hydrate_pinned_closure,
    is_verified_hydrated_recovery_output,
    is_verified_hydrated_zarr_directory,
    mark_beam_postprocess_in_progress,
    read_beam_run_checkpoint,
    snapshot_and_publish_beam_run_checkpoint,
    validate_pinned_closure_snapshot,
    verify_archive_visible_pinned_closure_bytes,
)
from pilates.workflows.binding import BindingPlan
from pilates.workflows.orchestration import run_workflow
from pilates.workflows.orchestration import StageRunner
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.steps import (
    StepOutputsHolder,
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
    from pilates.workflows.surface import EnabledWorkflowSurface


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
    activity_demand_outputs: Optional[Dict[str, Any]]
    previous_beam_outputs: Optional[Dict[str, Any]]


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

    previous_beam_outputs: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class _RecoveredBeamRun:
    """A historical BEAM projection plus its direct producer identity."""

    outputs: BeamRunOutputs
    producer_run_id: Optional[str]


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = getattr(settings, "beam", None)
    skim_cfg = getattr(beam_cfg, "full_skim", None) if beam_cfg else None
    if skim_cfg is None:
        return "disabled"
    return getattr(skim_cfg, "run_schedule", "standalone")


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
    binding: BindingPlan,
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
        beam_config = getattr(getattr(settings, "beam", None), "config", None)
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
    missing_required = (
        binding.missing_required if binding.missing_required is not None else []
    )
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
        "restart_run": bool(getattr(state, "is_restart_run", False)),
        "workflow_year": getattr(state, "year", getattr(state, "current_year", None)),
        "forecast_year": getattr(state, "forecast_year", None),
        "iteration": getattr(
            state,
            "iteration",
            getattr(state, "current_inner_iter", None),
        ),
        "input_keys": sorted(
            binding.input_keys if binding.input_keys is not None else []
        ),
        "optional_input_keys": sorted(
            binding.optional_input_keys
            if binding.optional_input_keys is not None
            else []
        ),
        "bound_input_keys": sorted(
            (binding.inputs if binding.inputs is not None else {}).keys()
        ),
        "missing_required": sorted(missing_required),
        "missing_restart_inputs": missing_restart_inputs,
        "source_by_key": dict(
            sorted(
                (
                    binding.source_by_key if binding.source_by_key is not None else {}
                ).items()
            )
        ),
        "coupler_key_by_key": dict(
            sorted(
                (
                    binding.coupler_key_by_key
                    if binding.coupler_key_by_key is not None
                    else {}
                ).items()
            )
        ),
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
    binding: BindingPlan,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    scenario: Optional[Any] = None,
) -> None:
    if not bool(getattr(state, "is_restart_run", False)):
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
    binding: BindingPlan,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
) -> None:
    if not bool(getattr(state, "is_restart_run", False)):
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

    binding_inputs = binding.inputs if binding.inputs is not None else {}
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
            deduped = [
                key
                for key in deduped
                if not key.startswith(f"{final_key}_sub")
            ]
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

    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return None

    promoted_outputs: Dict[str, Any] = {}
    for key in (LINKSTATS_WARMSTART, LINKSTATS, BEAM_PLANS_OUT):
        value = get_value(key)
        if value is None:
            continue
        if artifact_to_existing_path(value, workspace):
            promoted_outputs[key] = value
    return promoted_outputs or None


def _archive_run_dir_for_restart(state: WorkflowState) -> Optional[Path]:
    run_info_path = getattr(state, "run_info_path", None)
    if not run_info_path:
        return None
    try:
        return Path(run_info_path).expanduser().resolve().parent
    except Exception:
        return None


def _beam_restart_output_requests(
    linked_output_keys: Iterable[str],
    workspace: Workspace,
    year: int,
    iteration: int,
) -> tuple[HistoricalOutputRequest, ...]:
    """Map selected BEAM output keys to their exact current destinations."""

    selected = (
        _build_beam_postprocess_input_keys(
            upstream_keys=linked_output_keys,
            year=year,
            iteration=iteration,
            include_zarr_skims=False,
        )
        or []
    )
    selected.extend(
        key
        for key in linked_output_keys
        if key == LINKSTATS or key.startswith(BEAM_PLANS_OUT)
    )
    selected = list(dict.fromkeys(selected))
    output_dir = Path(workspace.get_beam_output_dir())
    requests: list[HistoricalOutputRequest] = []
    for key in selected:
        if key == LINKSTATS:
            destination = output_dir / f"{iteration}.linkstats.csv.gz"
        elif key.startswith("beam_plans_out"):
            destination = output_dir / "plans.csv.gz"
        elif key.startswith("events_parquet_"):
            destination = output_dir / f"{iteration}.events.parquet"
        elif key.startswith("raw_od_skims_zarr"):
            destination = output_dir / f"{iteration}.skimsActivitySimOD.zarr"
        elif key.startswith("raw_od_skims"):
            destination = output_dir / f"{iteration}.skimsActivitySimOD_current.omx"
        else:
            raise ResumePlanningError(
                "destination_contract_error",
                f"No deterministic BEAM destination exists for output key={key}.",
            )
        requests.append(
            HistoricalOutputRequest(key=key, destination=destination, required=True)
        )
    return tuple(requests)


def _project_hydrated_beam_run_outputs(
    *,
    hydration_result: Any,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> tuple[BeamRunOutputs, tuple[str, ...]]:
    """Validate accepted BEAM hydration items and publish current roles only."""

    raw_outputs: Dict[str, Path] = {}
    published_keys: list[str] = []
    for key, item in hydration_result.items():
        if item.path is None:
            if item.resolvable:
                raise ResumeProjectionError(
                    "unsupported_output_representation",
                    f"BEAM output key={key} has no hydrated destination.",
                )
            continue
        if (
            item.status == "materialized_from_filesystem"
            and item.resolvable
            and item.path.is_dir()
        ):
            raise ResumeProjectionError(
                "unsupported_output_representation",
                f"BEAM output key={key} is not a file destination.",
            )
        is_file = (
            item.status == "materialized_from_filesystem"
            and item.resolvable
            and item.path.is_file()
        )
        is_zarr_directory = is_verified_hydrated_zarr_directory(
            item, destination=item.path
        )
        if not is_file and not is_zarr_directory:
            continue
        raw_outputs[str(key)] = item.path
        set_coupler_from_artifact(
            coupler, str(key), item.artifact, fallback=str(item.path)
        )
        published_keys.append(str(key))
        if key == LINKSTATS:
            set_coupler_from_artifact(
                coupler,
                LINKSTATS_WARMSTART,
                item.artifact,
                fallback=str(item.path),
            )
            published_keys.append(LINKSTATS_WARMSTART)

    outputs = BeamRunOutputs(
        beam_output_dir=Path(workspace.get_beam_output_dir()), raw_outputs=raw_outputs
    )
    outputs.validate()
    return outputs, tuple(published_keys)


def _restored_beam_parent_years(
    *,
    state: WorkflowState,
    run_year: int,
) -> list[int]:
    years: list[int] = []
    for value in (run_year, getattr(state, "forecast_year", None)):
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        if year not in years:
            years.append(year)
    return years


def _beam_checkpoint_scope(
    *, state: WorkflowState, year: int, iteration: int
) -> dict[str, int]:
    return {
        "year": int(state.year),
        "forecast_year": int(year),
        "iteration": int(iteration),
    }


def _beam_checkpoint_skim_variant(settings: PilatesConfig) -> str:
    full_skim = settings.beam.full_skim
    return str(full_skim.run_schedule) if full_skim is not None else "disabled"


def beam_checkpoint_resume_requested(*, state: WorkflowState) -> bool:
    """Whether restart state contains an authoritative BEAM checkpoint record."""

    if not state.is_restart_run:
        return False
    archive_run_dir = _archive_run_dir_for_restart(state)
    return archive_run_dir is not None and beam_checkpoint_record_present(
        archive_run_dir
    )


def _closure_artifact_kind(artifact: Any) -> str:
    meta = artifact.meta
    if not isinstance(meta, dict):
        raise RuntimeError(
            f"BEAM checkpoint artifact {artifact.key!r} has invalid metadata."
        )
    if meta.get("directory_artifact") is True:
        return "directory"
    if meta.get("file_bundle_artifact") is True:
        return "file_bundle"
    return "file"


def _beam_postprocess_destination(
    *,
    role: str,
    output_key: str,
    workspace: Workspace,
    year: int,
    iteration: int,
) -> Path:
    """Return the native resolver's exact postprocess input destination."""

    if role == ZARR_SKIMS:
        return Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
    del year, iteration
    input_root = Path(workspace.get_beam_output_dir()) / ".pilates-consist-inputs"
    if output_key.startswith("events_parquet_"):
        return input_root / f"{output_key}.parquet"
    if output_key.startswith("raw_od_skims_zarr"):
        return input_root / f"{output_key}.zarr"
    if output_key.startswith("raw_od_skims"):
        return input_root / f"{output_key}.omx"
    raise RuntimeError(
        f"BEAM postprocess role has no exact native destination: {role}."
    )


def _rebind_beam_postprocess_closure_destinations(
    *,
    members: tuple[PinnedClosureMember, ...],
    workspace: Workspace,
    archive_run_dir: Path,
    year: int,
    iteration: int,
) -> tuple[PinnedClosureMember, ...]:
    """Bind a prior job's logical destinations to this job's workspace root."""

    workspace_root = Path(workspace.full_path).expanduser().resolve()
    run_name = archive_run_dir.name
    rebound: list[PinnedClosureMember] = []
    for member in members:
        destination = (
            _beam_postprocess_destination(
                role=member.role,
                output_key=member.output_key,
                workspace=workspace,
                year=year,
                iteration=iteration,
            )
            .expanduser()
            .resolve()
        )
        try:
            workspace_relative = destination.relative_to(workspace_root)
        except ValueError as error:
            raise RuntimeError(
                "BEAM checkpoint destination resolver escaped the current workspace: "
                f"{destination}."
            ) from error

        expected_suffix = (run_name, *workspace_relative.parts)
        if member.destination.parts[-len(expected_suffix) :] != expected_suffix:
            raise RuntimeError(
                "Committed BEAM checkpoint has a wrong logical destination for "
                f"{member.output_key}: {member.destination}."
            )
        rebound.append(replace(member, destination=destination))
    return tuple(rebound)


def _resolve_beam_postprocess_closure(
    *,
    scenario: ScenarioWithCoupler,
    tracker: Any,
    resolved_inputs: ResolvedStepInputs,
    workspace: Workspace,
    year: int,
    activitysim_year: int,
    iteration: int,
    beam_run_id: str,
) -> tuple[PinnedClosureMember, ...]:
    """Freeze exactly one resolved native postprocess input closure."""

    try:
        activitysim_run_id = scenario._activitysim_run_ids[
            (activitysim_year, iteration)
        ]  # type: ignore[attr-defined]
    except (AttributeError, KeyError):
        activitysim_run_id = None

    outputs_by_run: dict[str, Mapping[str, Any]] = {}
    members: list[PinnedClosureMember] = []
    destinations_seen: dict[Path, tuple[str, str]] = {}
    selected_roles = tuple(
        role
        for role in (*resolved_inputs.required_roles, *resolved_inputs.optional_roles)
        if resolved_inputs.source_by_role.get(role) != "missing"
    )
    if not selected_roles:
        raise RuntimeError("Cannot commit an empty BEAM postprocess input closure.")

    for role in selected_roles:
        if role not in (resolved_inputs.binding.inputs or {}):
            raise RuntimeError(
                f"BEAM postprocess resolved role {role!r} is not a concrete input."
            )
        producer_run_id = activitysim_run_id if role == ZARR_SKIMS else beam_run_id
        if producer_run_id is None:
            raise RuntimeError(
                "Cannot commit BEAM postprocess Zarr without the direct "
                "ActivitySim run ID."
            )
        output_key = resolved_inputs.selected_key_by_role.get(role, role)
        destination = resolved_inputs.logical_destinations.get(role)
        if destination is None:
            raise RuntimeError(
                f"BEAM postprocess input {role!r} has no logical destination."
            )
        destination = Path(destination).expanduser().resolve()
        producer_output = (producer_run_id, output_key)
        prior_producer_output = destinations_seen.get(destination)
        if prior_producer_output is not None:
            if prior_producer_output == producer_output:
                continue
            raise RuntimeError(
                "BEAM postprocess closure has conflicting artifacts for "
                f"destination {destination}: {prior_producer_output!r} and "
                f"{producer_output!r}."
            )
        destinations_seen[destination] = producer_output
        outputs = outputs_by_run.get(producer_run_id)
        if outputs is None:
            outputs = tracker.get_run_outputs(producer_run_id)
            outputs_by_run[producer_run_id] = outputs
        artifact = outputs.get(output_key)
        if artifact is None:
            raise RuntimeError(
                f"Cannot commit BEAM postprocess input without output link {output_key}."
            )
        if not artifact.hash:
            raise RuntimeError(
                f"Cannot commit BEAM postprocess input without identity {output_key}."
            )
        members.append(
            PinnedClosureMember(
                member_id=f"{producer_run_id}:{output_key}",
                role=role,
                producer_run_id=producer_run_id,
                output_key=output_key,
                artifact_identity=str(artifact.hash),
                artifact_kind=_closure_artifact_kind(artifact),
                driver=(str(artifact.driver) if artifact.driver is not None else None),
                destination=destination,
                required=role in resolved_inputs.required_roles,
            )
        )
    return tuple(members)


def _open_beam_checkpoint_tracker(snapshot_path: Path, archive_run_dir: Path) -> Any:
    """Open only the pinned archive-local DB for committed BEAM restoration."""

    return cr.consist.Tracker(
        run_dir=archive_run_dir,
        db_path=snapshot_path,
        allow_external_paths=True,
        access_mode="read_only",
    )


def _try_restore_completed_beam_run_for_restart(
    *,
    scenario: ScenarioRestorationLike,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    surface: "EnabledWorkflowSurface",
    year: int,
    iteration: int,
) -> Optional[_RecoveredBeamRun]:
    if not bool(getattr(state, "is_restart_run", False)):
        return None
    if outputs_holder.beam_run is not None:
        return _RecoveredBeamRun(
            outputs=outputs_holder.beam_run,
            producer_run_id=None,
        )
    archive_run_dir = _archive_run_dir_for_restart(state)
    if archive_run_dir is not None and beam_checkpoint_record_present(archive_run_dir):
        raise RuntimeError(
            "Committed BEAM checkpoint must use checkpoint-first postprocess dispatch."
        )

    tracker = cr.current_tracker()
    if tracker is None:
        return None
    policy = ResumeBoundaryPolicy(
        step_name="beam_run",
        rerun_forbidden=True,
        allows_restore=lambda candidate_state, _surface: candidate_state.is_restart_run,
        output_requests=_beam_restart_output_requests,
    )
    plan = build_resume_plan(
        state=state,
        surface=surface,
        settings=settings,
        workspace=workspace,
        tracker=tracker,
        year=year,
        iteration=iteration,
        policy=policy,
    )
    decision = plan.decisions["beam_run"]
    if decision.disposition is not ResumeDisposition.RESTORE:
        return None

    execution: RestoreExecutionResult = execute_restore_decision(
        decision=decision,
        tracker=tracker,
        source_root=_archive_run_dir_for_restart(state),
        projection_adapter=lambda hydration: _project_hydrated_beam_run_outputs(
            hydration_result=hydration,
            workspace=workspace,
            coupler=coupler,
        ),
        required_output_validator=lambda request, item: (
            is_verified_hydrated_recovery_output(item, destination=request.destination)
        ),
    )
    _emit_beam_restart_recovery_readiness_diagnostic(
        state=state,
        decision=decision,
        execution=execution,
        iteration=iteration,
    )
    if not execution.succeeded or not isinstance(
        execution.projected_outputs, BeamRunOutputs
    ):
        raise RuntimeError(
            "BEAM restart found a completed same-run beam_run but could not "
            "hydrate the required outputs for postprocess. Refusing to rerun "
            f"BEAM from restart because the completed run is authoritative. "
            f"run_id={decision.source_run_id} failed_keys={execution.failed_keys} "
            f"category={execution.failure_category}"
        )

    outputs = execution.projected_outputs
    outputs_holder.beam_run = outputs
    for parent_year in _restored_beam_parent_years(
        state=state,
        run_year=year,
    ):
        scenario.remember_restored_run_id(
            model_name="beam_run",
            year=parent_year,
            iteration=iteration,
            run_id=decision.source_run_id,
        )
    restored_keys = sorted(outputs.raw_outputs.keys())
    logger.info(
        "[BEAM][restart] restored completed beam_run from Consist run_id=%s hydrated_keys=%s",
        decision.source_run_id,
        restored_keys,
    )
    _emit_artifact_lifecycle_event(
        "beam_restart_binding",
        key="beam_restart_binding",
        artifact_family="beam_restart_diagnostic",
        diagnostic="beam_restart_binding",
        restart_run=True,
        workflow_year=getattr(state, "year", getattr(state, "current_year", None)),
        forecast_year=getattr(state, "forecast_year", None),
        iteration=iteration,
        recovery_mode="consist_completed_run_hydration",
        recovered_run_id=decision.source_run_id,
        hydrated_output_keys=restored_keys,
        drift_classification="completed_beam_run_recovered",
    )
    return _RecoveredBeamRun(
        outputs=outputs,
        producer_run_id=decision.source_run_id,
    )


def _publish_completed_beam_run_checkpoint(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    postprocess_inputs: ResolvedStepInputs,
    year: int,
    iteration: int,
    producer_run_id: Optional[str] = None,
) -> None:
    """Commit a BEAM-only overlay without advancing traffic-assignment state."""

    archive_run_dir = _archive_run_dir_for_restart(state)
    if archive_run_dir is None:
        return
    tracker = cr.current_tracker()
    if tracker is None:
        raise RuntimeError(
            "Cannot commit completed BEAM run without an active tracker."
        )
    if producer_run_id is None:
        try:
            run_id = scenario._beam_run_ids[(year, iteration)]  # type: ignore[attr-defined]
        except (AttributeError, KeyError) as error:
            raise RuntimeError(
                "Cannot commit completed BEAM run without its direct run ID."
            ) from error
    else:
        run_id = producer_run_id
    closure_members = _resolve_beam_postprocess_closure(
        scenario=scenario,
        tracker=tracker,
        resolved_inputs=postprocess_inputs,
        workspace=workspace,
        year=year,
        activitysim_year=int(state.forecast_year),
        iteration=iteration,
        beam_run_id=run_id,
    )
    verify_archive_visible_pinned_closure_bytes(
        tracker=tracker,
        archive_run_dir=archive_run_dir,
        members=closure_members,
    )
    snapshot_and_publish_beam_run_checkpoint(
        tracker=tracker,
        open_snapshot=lambda snapshot_path: _open_beam_checkpoint_tracker(
            snapshot_path, archive_run_dir
        ),
        archive_run_dir=archive_run_dir,
        producer_run_id=run_id,
        scope=_beam_checkpoint_scope(state=state, year=year, iteration=iteration),
        skim_variant=_beam_checkpoint_skim_variant(settings),
        output_requests=(),
        closure_members=closure_members,
    )


def _validate_rebound_postprocess_inputs(
    *,
    checkpoint: Any,
    members: tuple[PinnedClosureMember, ...],
    resolved_inputs: ResolvedStepInputs,
) -> None:
    """Fail closed unless the normal resolver exactly reproduces the closure."""

    closure_by_role = {member.role: member for member in members}
    if len(closure_by_role) != len(members):
        raise RuntimeError("Committed BEAM checkpoint closure has duplicate roles.")
    resolved_roles = {
        role
        for role in (*resolved_inputs.required_roles, *resolved_inputs.optional_roles)
        if resolved_inputs.source_by_role.get(role) != "missing"
    }
    if resolved_roles != set(closure_by_role):
        raise RuntimeError(
            "Committed BEAM checkpoint closure does not match normal postprocess "
            "resolver roles."
        )

    inputs = resolved_inputs.binding.inputs or {}
    for role, member in closure_by_role.items():
        if member.required != (role in resolved_inputs.required_roles):
            raise RuntimeError(
                f"Committed BEAM checkpoint requiredness drifted for {role!r}."
            )
        if resolved_inputs.selected_key_by_role.get(role, role) != member.output_key:
            raise RuntimeError(
                f"Committed BEAM checkpoint selected key drifted for {role!r}."
            )
        destination = resolved_inputs.logical_destinations.get(role)
        if (
            destination is None
            or Path(destination).expanduser().resolve() != member.destination
        ):
            raise RuntimeError(
                f"Committed BEAM checkpoint destination drifted for {role!r}."
            )
        artifact = inputs.get(role)
        if getattr(artifact, "hash", None) != member.artifact_identity:
            raise RuntimeError(
                f"Committed BEAM checkpoint hydrated identity drifted for {role!r}."
            )
    if checkpoint.producer_run_id not in {member.producer_run_id for member in members}:
        raise RuntimeError("Committed BEAM checkpoint producer is absent from closure.")


def _try_resume_committed_beam_postprocess(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
) -> Optional[Dict[str, Any]]:
    """Hydrate one committed successor closure and invoke postprocess only."""

    state = context.state
    if not state.is_restart_run:
        return None
    archive_run_dir = _archive_run_dir_for_restart(state)
    if archive_run_dir is None or not beam_checkpoint_record_present(archive_run_dir):
        return None
    if beam_postprocess_is_in_progress(archive_run_dir):
        raise RuntimeError(
            "BEAM checkpoint is non-restartable: beam_postprocess_in_progress."
        )
    checkpoint = read_beam_run_checkpoint(archive_run_dir)
    if checkpoint is None or not checkpoint.closure_members:
        raise RuntimeError("Committed BEAM checkpoint is invalid or incomplete.")
    snapshot_path = (archive_run_dir / checkpoint.snapshot_ref).resolve()
    if (
        not snapshot_path.is_relative_to(archive_run_dir.resolve())
        or not snapshot_path.is_file()
    ):
        raise RuntimeError(
            "Committed BEAM checkpoint snapshot is missing or outside archive."
        )
    expected_scope = _beam_checkpoint_scope(
        state=state,
        year=year,
        iteration=iteration,
    )
    if not checkpoint_fingerprint_matches(
        checkpoint,
        scope=expected_scope,
        skim_variant=_beam_checkpoint_skim_variant(context.settings),
        output_requests=(),
        closure_members=checkpoint.closure_members,
    ):
        raise RuntimeError(
            "Committed BEAM checkpoint recovery fingerprint does not match."
        )

    snapshot_tracker = _open_beam_checkpoint_tracker(
        snapshot_path,
        archive_run_dir,
    )
    validate_pinned_closure_snapshot(
        tracker=snapshot_tracker,
        members=checkpoint.closure_members,
        scope=expected_scope,
    )
    closure_members = _rebind_beam_postprocess_closure_destinations(
        members=checkpoint.closure_members,
        workspace=context.workspace,
        archive_run_dir=archive_run_dir,
        year=year,
        iteration=iteration,
    )
    restored = hydrate_pinned_closure(
        tracker=snapshot_tracker,
        source_root=archive_run_dir,
        members=closure_members,
    )

    for member in closure_members:
        item = restored[member.member_id]
        if item.path is None:
            raise RuntimeError(
                f"Committed BEAM checkpoint restored no path for {member.output_key}."
            )
        set_coupler_from_artifact(
            coupler,
            member.output_key,
            item.artifact,
            fallback=str(item.path),
        )
        if member.role != member.output_key:
            set_coupler_from_artifact(
                coupler,
                member.role,
                item.artifact,
                fallback=str(item.path),
            )
    postprocess_inputs = beam_postprocess.resolve_inputs(
        settings=context.settings,
        state=state,
        workspace=context.workspace,
        coupler=coupler,
    )
    postprocess_inputs.require_complete()
    _validate_rebound_postprocess_inputs(
        checkpoint=checkpoint,
        members=closure_members,
        resolved_inputs=postprocess_inputs,
    )
    mark_beam_postprocess_in_progress(archive_run_dir, checkpoint)
    _, postprocess_outputs = execute_step(
        scenario=scenario,
        definition=beam_postprocess,
        settings=context.settings,
        state=state,
        workspace=context.workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="postprocess",
        resolved_inputs=postprocess_inputs,
    )
    return step_output_handoff_mapping(postprocess_outputs, coupler=coupler)


def _emit_beam_restart_recovery_readiness_diagnostic(
    *,
    state: WorkflowState,
    decision: ResumeDecision,
    execution: RestoreExecutionResult,
    iteration: int,
) -> None:
    """Record the existing diagnostic from the planner and hydration result."""
    required_keys = [request.key for request in decision.outputs]
    missing_required = sorted(set(execution.failed_keys) & set(required_keys))
    accepted_keys = (
        sorted(execution.projected_outputs.raw_outputs)
        if isinstance(execution.projected_outputs, BeamRunOutputs)
        else []
    )
    readiness_classification = "complete" if execution.succeeded else "restore_failed"
    logger.info(
        "[BEAM][restart] recovery readiness diagnostic: matchable=%s run_id=%s "
        "missing_required=%s required_keys=%s output_keys=%s",
        decision.disposition is ResumeDisposition.RESTORE,
        decision.source_run_id,
        missing_required,
        required_keys,
        accepted_keys,
    )
    _emit_artifact_lifecycle_event(
        "beam_restart_recovery_readiness",
        key="beam_restart_recovery_readiness",
        artifact_family="beam_restart_diagnostic",
        diagnostic="beam_restart_recovery_readiness",
        restart_run=bool(getattr(state, "is_restart_run", False)),
        workflow_year=getattr(state, "year", getattr(state, "current_year", None)),
        forecast_year=getattr(state, "forecast_year", None),
        iteration=iteration,
        run_scope=decision.semantic_target.get("run_scope"),
        query_status=decision.semantic_target.get("status"),
        matched_completed_run_id=decision.source_run_id,
        matched_run_id=decision.source_run_id,
        matchable=decision.disposition is ResumeDisposition.RESTORE,
        output_keys=accepted_keys,
        matched_output_keys=accepted_keys,
        required_restored_inputs=required_keys,
        required_postprocess_keys=required_keys,
        missing_restored_inputs=missing_required,
        missing_required_keys=missing_required,
        hydration_api_available=True,
        identity_summary={},
        cache_miss_explanation={},
        identity_drift_components={},
        drift_classification=readiness_classification,
        diagnostic_error=execution.failure_category,
    )


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


def _finalize_beam_run_input_keys(
    *,
    beam_run_input_keys: Optional[list[str]],
    outputs_holder: StepOutputsHolder,
) -> list[str]:
    """
    Reconcile BEAM run inputs with the artifacts actually published by preprocess.

    The pre-run key derivation happens before BEAM preprocess executes, but
    preprocess may decide to publish ``linkstats_warmstart`` after resolving
    previous outputs. Use the realized preprocess outputs as the final contract.
    """
    finalized_keys = list(
        beam_run_input_keys
        or [
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ]
    )
    preprocess_outputs = outputs_holder.beam_preprocess
    prepared_inputs = (
        preprocess_outputs.prepared_inputs if preprocess_outputs is not None else {}
    )
    has_warmstart = LINKSTATS_WARMSTART in prepared_inputs
    has_vehicles = _BEAM_VEHICLES_IN in prepared_inputs
    if has_warmstart and LINKSTATS_WARMSTART not in finalized_keys:
        finalized_keys.append(LINKSTATS_WARMSTART)
    if not has_warmstart and LINKSTATS_WARMSTART in finalized_keys:
        finalized_keys = [key for key in finalized_keys if key != LINKSTATS_WARMSTART]
    if has_vehicles and _BEAM_VEHICLES_IN not in finalized_keys:
        finalized_keys.append(_BEAM_VEHICLES_IN)
    return finalized_keys


def _make_beam_stage_runner(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    year: int,
    iteration: int,
    runtime_kwargs_extra: Optional[Mapping[str, Any]] = None,
    context: WorkflowRuntimeContext,
) -> StageRunner:
    """Build the execution context shared by BEAM stage slices."""
    return StageRunner(
        stage_name="beam",
        scenario=scenario,
        state=context.state,
        settings=context.settings,
        workspace=context.workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=f"{year}_iter{iteration}",
        iteration=iteration,
        runtime_kwargs_extra=runtime_kwargs_extra,
        run_workflow_fn=run_workflow,
    )


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


def _run_beam_steps(
    *,
    scenario: ScenarioWithCoupler,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
) -> Optional[Dict[str, Any]]:
    """Dispatch the single public BEAM run/postprocess checkpoint boundary."""

    settings = context.settings
    state = context.state
    workspace = context.workspace
    if beam_checkpoint_resume_requested(state=state):
        return _try_resume_committed_beam_postprocess(
            scenario=scenario,
            coupler=scenario.coupler,
            year=year,
            iteration=iteration,
            context=context,
        )
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
    del preprocess_outputs
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
    )
    postprocess_inputs = beam_postprocess.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=scenario.coupler,
    )
    postprocess_inputs.require_complete()
    _publish_completed_beam_run_checkpoint(
        scenario=scenario,
        settings=settings,
        state=state,
        workspace=workspace,
        postprocess_inputs=postprocess_inputs,
        year=year,
        iteration=iteration,
        producer_run_id=run_result.run.id,
    )
    logger.info("[beam] completed native beam_run run_id=%s", run_result.run.id)
    _maybe_fail_after_beam_run_for_canary(year=year, iteration=iteration)
    archive_run_dir = _archive_run_dir_for_restart(state)
    if archive_run_dir is not None:
        checkpoint = read_beam_run_checkpoint(archive_run_dir)
        if checkpoint is None:
            raise RuntimeError(
                "BEAM checkpoint publication did not commit before postprocess."
            )
        mark_beam_postprocess_in_progress(archive_run_dir, checkpoint)
    _, postprocess_outputs = execute_step(
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
    return step_output_handoff_mapping(postprocess_outputs, coupler=scenario.coupler)


def _run_beam_full_skim_step(
    *,
    scenario: ScenarioWithCoupler,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
) -> Optional[Dict[str, Any]]:
    """Execute full skims through its native definition."""

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
    )
    return {key: path for key, path, _description in outputs._iter_record_items()}


def _run_traffic_assignment_phase(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    inputs: TrafficAssignmentPhaseInputs,
    outputs_holder: object | None = None,
    context: Optional[WorkflowRuntimeContext] = None,
    state: Optional[WorkflowState] = None,
    settings: Optional[PilatesConfig] = None,
    workspace: Optional[Workspace] = None,
    surface: Optional["EnabledWorkflowSurface"] = None,
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
    runtime_context = ensure_workflow_runtime_context(
        context=context,
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    settings = runtime_context.settings
    state = runtime_context.state
    workspace = runtime_context.workspace
    del coupler, outputs_holder

    formatted_print("TRAFFIC ASSIGNMENT MODEL")

    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        _run_beam_preprocess_step(
            scenario=scenario,
            settings=settings,
            state=state,
            workspace=workspace,
            year=inputs.year,
            iteration=inputs.iteration,
        )
        combined_beam_outputs = _run_beam_full_skim_step(
            scenario=scenario,
            year=inputs.year,
            iteration=inputs.iteration,
            context=runtime_context,
        )
    else:
        combined_beam_outputs = _run_beam_steps(
            scenario=scenario,
            year=inputs.year,
            iteration=inputs.iteration,
            context=runtime_context,
        )
        if _should_run_full_skim(settings, inputs.iteration):
            full_skim_outputs = _run_beam_full_skim_step(
                scenario=scenario,
                year=inputs.year,
                iteration=inputs.iteration,
                context=runtime_context,
            )
            if full_skim_outputs is not None:
                if combined_beam_outputs is None:
                    combined_beam_outputs = {}
                combined_beam_outputs.update(full_skim_outputs)

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.traffic_assignment,
    )

    return TrafficAssignmentPhaseOutputs(previous_beam_outputs=combined_beam_outputs)
