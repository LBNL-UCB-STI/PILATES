"""Typed, stage-owned planning and execution for historical output restoration."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from consist import Artifact, HydratedRunOutputsResult

from pilates.config.models import PilatesConfig
from pilates.runtime.restart import restart_target_for_step
from pilates.workspace import Workspace
from workflow_state import WorkflowState

from pilates.workflows.surface import EnabledWorkflowSurface


class ResumeDisposition(StrEnum):
    """The one permitted outcome for a stage-owned recovery boundary."""

    RESTORE = "restore"
    RUN = "run"
    SKIP = "skip"


@dataclass(frozen=True)
class HistoricalOutputRequest:
    """One historical output key and its exact current-workspace destination."""

    key: str
    destination: Path
    required: bool


@dataclass(frozen=True)
class ResumeDecision:
    """A planner-owned recovery decision for one workflow step."""

    step_name: str
    disposition: ResumeDisposition
    reason: str
    semantic_target: Mapping[str, object]
    source_run_id: str | None
    outputs: tuple[HistoricalOutputRequest, ...]
    rerun_forbidden: bool


@dataclass(frozen=True)
class ResumePlan:
    """The decisions for the active stage boundary."""

    workflow_instance_scope: str
    decisions: Mapping[str, ResumeDecision]


class ResumePlanningError(RuntimeError):
    """A non-recoverable violation of the historical selection contract."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category


class ResumeProjectionError(RuntimeError):
    """A stage projection failure with an operable recovery category."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category


class HistoricalRun(Protocol):
    """The selected Consist run fields used by the planner."""

    id: str
    status: str


class ResumeTracker(Protocol):
    """The direct Consist API required by the resume seam."""

    def find_matching_runs(self, **kwargs: object) -> list[HistoricalRun]: ...

    def get_run_outputs(self, run_id: str) -> Mapping[str, Artifact]: ...

    def hydrate_run_outputs_to_destinations(
        self,
        run_id: str,
        *,
        destinations_by_key: Mapping[str, Path],
        source_root: Path | None,
        preserve_existing: bool,
        on_missing: str,
        db_fallback: str,
    ) -> HydratedRunOutputsResult: ...


class HydratedOutput(Protocol):
    """The hydration fields required to admit one restored output."""

    path: Path | None
    status: str
    resolvable: bool


OutputRequestBuilder = Callable[
    [Collection[str], Workspace, int, int], tuple[HistoricalOutputRequest, ...]
]
RestoreEligibility = Callable[[WorkflowState, EnabledWorkflowSurface], bool]
ProjectionAdapter = Callable[[HydratedRunOutputsResult], tuple[object, tuple[str, ...]]]
RequiredOutputValidator = Callable[[HistoricalOutputRequest, HydratedOutput], bool]


@dataclass(frozen=True)
class ResumeBoundaryPolicy:
    """The stage-owned rules that the generic planner is allowed to apply."""

    step_name: str
    rerun_forbidden: bool
    allows_restore: RestoreEligibility
    output_requests: OutputRequestBuilder


@dataclass(frozen=True)
class RestoreExecutionResult:
    """The structured outcome of one exact-destination restore attempt."""

    decision: ResumeDecision
    hydration_result: HydratedRunOutputsResult | None
    projected_outputs: object | None
    published_role_keys: tuple[str, ...]
    failure_category: str | None
    failed_keys: tuple[str, ...]
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether every required output hydrated and projected successfully."""

        return self.failure_category is None


def _decision(
    *,
    policy: ResumeBoundaryPolicy,
    disposition: ResumeDisposition,
    reason: str,
    target: Mapping[str, object],
    source_run_id: str | None = None,
    outputs: tuple[HistoricalOutputRequest, ...] = (),
) -> ResumeDecision:
    return ResumeDecision(
        step_name=policy.step_name,
        disposition=disposition,
        reason=reason,
        semantic_target=target,
        source_run_id=source_run_id,
        outputs=outputs,
        rerun_forbidden=policy.rerun_forbidden,
    )


def build_resume_plan(
    *,
    state: WorkflowState,
    surface: EnabledWorkflowSurface,
    settings: PilatesConfig,
    workspace: Workspace,
    tracker: ResumeTracker,
    year: int,
    iteration: int,
    policy: ResumeBoundaryPolicy,
) -> ResumePlan:
    """Build one scoped, cardinality-safe historical restore decision."""

    if not surface.step_enabled(policy.step_name):
        decision = _decision(
            policy=policy,
            disposition=ResumeDisposition.SKIP,
            reason="outside_enabled_surface",
            target={},
        )
        return ResumePlan(
            workflow_instance_scope="", decisions={policy.step_name: decision}
        )

    if not policy.allows_restore(state, surface):
        decision = _decision(
            policy=policy,
            disposition=ResumeDisposition.RUN,
            reason="not_eligible_for_restore",
            target={},
        )
        return ResumePlan(
            workflow_instance_scope="", decisions={policy.step_name: decision}
        )

    target = restart_target_for_step(
        settings=settings,
        step_name=policy.step_name,
        year=year,
        iteration=iteration,
        state=state,
        workspace=workspace,
    )
    scope_value = target.get("run_scope")
    if not isinstance(scope_value, str) or not scope_value.strip():
        raise ResumePlanningError(
            "missing_workflow_scope",
            f"Historical restore for step={policy.step_name} requires run_scope.",
        )

    candidates = tracker.find_matching_runs(**target, limit=2)
    if len(candidates) > 1:
        raise ResumePlanningError(
            "ambiguous_completed_match",
            f"Historical restore for step={policy.step_name} found {len(candidates)} matches.",
        )
    if not candidates:
        decision = _decision(
            policy=policy,
            disposition=ResumeDisposition.RUN,
            reason="no_completed_match",
            target=target,
        )
        return ResumePlan(
            workflow_instance_scope=scope_value,
            decisions={policy.step_name: decision},
        )

    selected = candidates[0]
    if not isinstance(selected.id, str) or not selected.id.strip():
        raise ResumePlanningError(
            "malformed_completed_match",
            f"Historical restore for step={policy.step_name} returned a run without an ID.",
        )
    if selected.status != "completed":
        raise ResumePlanningError(
            "malformed_completed_match",
            f"Historical restore for step={policy.step_name} selected non-completed run_id={selected.id}.",
        )

    output_keys = tuple(tracker.get_run_outputs(selected.id).keys())
    requests = policy.output_requests(output_keys, workspace, year, iteration)
    if not any(request.required for request in requests):
        raise ResumePlanningError(
            "destination_contract_error",
            f"Historical restore for step={policy.step_name} has no required destinations.",
        )

    decision = _decision(
        policy=policy,
        disposition=ResumeDisposition.RESTORE,
        reason="completed_match",
        target=target,
        source_run_id=selected.id,
        outputs=requests,
    )
    return ResumePlan(
        workflow_instance_scope=scope_value,
        decisions={policy.step_name: decision},
    )


def _restore_failure(
    *,
    decision: ResumeDecision,
    hydration_result: HydratedRunOutputsResult | None,
    category: str,
    failed_keys: tuple[str, ...] = (),
    message: str | None = None,
) -> RestoreExecutionResult:
    return RestoreExecutionResult(
        decision=decision,
        hydration_result=hydration_result,
        projected_outputs=None,
        published_role_keys=(),
        failure_category=category,
        failed_keys=failed_keys,
        message=message,
    )


def execute_restore_decision(
    *,
    decision: ResumeDecision,
    tracker: ResumeTracker,
    source_root: Path | None,
    projection_adapter: ProjectionAdapter,
    required_output_validator: RequiredOutputValidator | None = None,
) -> RestoreExecutionResult:
    """Materialize one already-selected run only to exact stage destinations."""

    if decision.disposition is not ResumeDisposition.RESTORE:
        raise ValueError(
            f"execute_restore_decision requires RESTORE, got {decision.disposition}."
        )
    if decision.source_run_id is None:
        raise ValueError("RESTORE decision requires source_run_id.")

    required_requests = tuple(
        request for request in decision.outputs if request.required
    )
    if not required_requests:
        return _restore_failure(
            decision=decision,
            hydration_result=None,
            category="destination_contract_error",
            message="RESTORE decision has no required destinations.",
        )
    preexisting = tuple(
        request.key for request in required_requests if request.destination.exists()
    )
    if preexisting:
        return _restore_failure(
            decision=decision,
            hydration_result=None,
            category="preexisting_restore_destination",
            failed_keys=preexisting,
            message="Required restore destination already exists.",
        )

    result = tracker.hydrate_run_outputs_to_destinations(
        decision.source_run_id,
        destinations_by_key={
            request.key: request.destination for request in decision.outputs
        },
        source_root=source_root,
        preserve_existing=False,
        on_missing="warn",
        db_fallback="never",
    )
    if result.source_run_id != decision.source_run_id:
        return _restore_failure(
            decision=decision,
            hydration_result=result,
            category="source_run_mismatch",
            message=(
                f"Hydration returned source_run_id={result.source_run_id!r}, "
                f"expected {decision.source_run_id!r}."
            ),
        )

    validator = required_output_validator or _is_exact_file_hydration
    failed_keys = tuple(
        request.key
        for request in required_requests
        if ((item := result.get(request.key)) is None or not validator(request, item))
    )
    if failed_keys:
        return _restore_failure(
            decision=decision,
            hydration_result=result,
            category="missing_required_output",
            failed_keys=failed_keys,
            message="Required outputs did not materialize from the filesystem.",
        )

    try:
        projected_outputs, published_role_keys = projection_adapter(result)
    except ResumeProjectionError as error:
        return _restore_failure(
            decision=decision,
            hydration_result=result,
            category=error.category,
            message=str(error),
        )
    except Exception as error:
        return _restore_failure(
            decision=decision,
            hydration_result=result,
            category="projection_validation_failed",
            message=str(error),
        )
    return RestoreExecutionResult(
        decision=decision,
        hydration_result=result,
        projected_outputs=projected_outputs,
        published_role_keys=published_role_keys,
        failure_category=None,
        failed_keys=(),
    )


def _is_exact_file_hydration(
    request: HistoricalOutputRequest, item: HydratedOutput
) -> bool:
    """Accept the default Consist file-hydration representation."""

    return (
        item.resolvable
        and item.path == request.destination
        and item.status == "materialized_from_filesystem"
    )
