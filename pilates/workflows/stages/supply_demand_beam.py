from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, TYPE_CHECKING


from pilates.config.models import PilatesConfig
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils.consist_types import (
    ScenarioWithCoupler,
)
from pilates.utils.coupler_helpers import (
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
from pilates.workflows.steps import (
    beam_full_skim,
    beam_postprocess,
    beam_preprocess,
    beam_run,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.artifact_keys import (
    LINKSTATS,
    LINKSTATS_WARMSTART,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


_FAIL_AFTER_BEAM_RUN_ENV = "PILATES_FAIL_AFTER_BEAM_RUN"


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
    """

    year: int
    iteration: int


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = settings.beam
    skim_cfg = beam_cfg.full_skim if beam_cfg is not None else None
    if skim_cfg is None:
        return "disabled"
    return skim_cfg.run_schedule


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


def _archive_beam_launch_config(
    launch_config_dir: Path,
    *,
    year: int,
    iteration: int,
    targets: Sequence[Path | None],
) -> None:
    """Archive compiled BEAM launch config tree to durable run directories."""
    if not launch_config_dir.exists():
        return
    launch_config_resolved = launch_config_dir.resolve()
    for target in targets:
        if target is None:
            continue
        dest_root = Path(target).expanduser().resolve()
        dest_dir = (
            dest_root / ".pilates-beam-launch-config" / f"y{year}" / f"i{iteration}"
        )
        if dest_dir.resolve() == launch_config_resolved:
            continue
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            launch_config_dir,
            dest_dir,
            dirs_exist_ok=True,
            ignore_dangling_symlinks=True,
        )
        logger.debug(
            "[beam] archived launch config to %s",
            dest_dir,
        )


def _compile_beam_launch_config(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    workspace: Workspace,
    year: int,
    iteration: int,
    preprocess_outputs: Any,
    archive_run_dir: Optional[Path] = None,
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
    tracker_run_dir = getattr(getattr(scenario, "tracker", None), "run_dir", None)
    _archive_beam_launch_config(
        output_dir,
        year=year,
        iteration=iteration,
        targets=[
            archive_run_dir,
            Path(tracker_run_dir) if tracker_run_dir else None,
        ],
    )
    return launch_config


def _run_beam_steps(
    *,
    scenario: ScenarioWithCoupler,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
) -> None:
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
        archive_run_dir=archive_run_dir,
    )
    run_inputs = beam_run.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=scenario.coupler,
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
    del postprocess_outputs
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

    if _should_run_full_skim(settings, iteration):
        _run_beam_full_skim_step(
            scenario=scenario,
            year=year,
            iteration=iteration,
            context=context,
            launch_config=launch_config,
        )
    return None


def _run_beam_full_skim_step(
    *,
    scenario: ScenarioWithCoupler,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
    launch_config: BeamLaunchConfig,
) -> None:
    """Execute full skims through its native definition."""

    full_skim_inputs = beam_full_skim.resolve_inputs(
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        coupler=scenario.coupler,
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
    del outputs
    return None


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
) -> None:
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
        archive_run_dir = _archive_run_dir_for_restart(state)
        launch_config = _compile_beam_launch_config(
            scenario=scenario,
            settings=settings,
            workspace=workspace,
            year=inputs.year,
            iteration=inputs.iteration,
            preprocess_outputs=preprocess_outputs,
            archive_run_dir=archive_run_dir,
        )
        _run_beam_full_skim_step(
            scenario=scenario,
            year=inputs.year,
            iteration=inputs.iteration,
            context=context,
            launch_config=launch_config,
        )
    else:
        _run_beam_steps(
            scenario=scenario,
            year=inputs.year,
            iteration=inputs.iteration,
            context=context,
        )

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.traffic_assignment,
    )

    return None
