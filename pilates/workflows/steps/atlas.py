from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Sequence

from consist import (
    ExecutionOptions,
    OutputSet,
    Tracker,
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
from pilates.utils import consist_runtime as cr
from pilates.utils.archive_paths import resolve_archive_run_dir
from pilates.utils.consist_runtime import require_runtime_kwargs
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.utils.io import get_land_use_model
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.state_access import is_start_year
from pilates.utils.usim_h5 import create_usim_population_source_snapshot
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_H5,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.consist_metadata import consist_step_meta
from pilates.workflows.contracts import (
    ConfigContract,
    InputContract,
    StepCallInputs,
    StepDefinition,
)
from pilates.workflows.sim_state import ArtifactRef, SimState
from pilates.workflows.state_helpers import requires_prior_beam_skim_handoff
from pilates.workspace import Workspace
from pilates.workflows.state import WorkflowState

from .shared import (
    _build_step_call_inputs,
    _execution_options_with_launch_context,
    _metadata_cache_options,
    _native_contract_output_paths,
    _native_execution_options,
    _persisted_output_path,
)
from .urbansim import (
    _urbansim_population_source_path,
    _urbansim_run_native_output_paths,
)

logger = logging.getLogger(__name__)

# Native Consist step definitions.

_ATLAS_HOUSEHOLDV_CSV_INPUT = "atlas_householdv_csv"
_ATLAS_VEHICLES_CSV_INPUT = "atlas_vehicles_csv"


def _atlas_run_execution_options(
    *,
    step_inputs: StepCallInputs | None = None,
    **_: Any,
) -> ExecutionOptions:
    """Pass the resolver-owned ATLAS launch context through runtime kwargs."""

    return _execution_options_with_launch_context(
        step_inputs=step_inputs,
        step_name="atlas_run",
        context_key="atlas_launch_context",
        expected_type=AtlasLaunchContext,
    )


def _atlas_preprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> Dict[str, Any]:
    del step_inputs
    return dict(
        _atlas_preprocess_prepared_input_paths(
            settings=settings,
            state=state,
            workspace=workspace,
        )
    )


def _atlas_prepared_input_paths_for_year(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    year: int,
    start_year: int,
) -> dict[str, Path]:
    """Project the known prepared CSV members from the declared year root."""
    root = Path(workspace.get_atlas_mutable_input_dir()) / f"year{year}"
    paths: dict[str, Path] = {
        "atlas_households_csv": root / "households.csv",
        "atlas_blocks_csv": root / "blocks.csv",
        "atlas_persons_csv": root / "persons.csv",
        "atlas_residential_csv": root / "residential.csv",
        "atlas_jobs_csv": root / "jobs.csv",
    }
    if year > start_year:
        paths["atlas_grave_csv"] = root / "grave.csv"
    if settings.atlas.beamac > 0:
        paths["atlas_accessibility_csv"] = root / f"accessibility_{year}_tract.csv"
    return paths


def _atlas_preprocess_prepared_input_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> Mapping[str, Path]:
    """Project the known prepared CSV members from the declared year root."""
    return _atlas_prepared_input_paths_for_year(
        settings=settings,
        workspace=workspace,
        year=int(state.forecast_year),
        start_year=int(state.start_year),
    )


def _atlas_run_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> Dict[str, Any]:
    del settings, step_inputs
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
    step_inputs: StepCallInputs | None = None,
) -> Mapping[str, OutputSet]:
    """Declare the ATLAS continuation surface exactly."""

    del settings, step_inputs
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
    step_inputs: StepCallInputs | None = None,
) -> Dict[str, Any]:
    """
    Declare output paths for ``atlas_postprocess``.

    Publishes ``USIM_POPULATION_SOURCE_H5`` as a distinct post-mutation snapshot
    for downstream ActivitySim runs and, when ``year`` or ``forecast_year`` is
    set on ``state``, ``ATLAS_VEHICLES2_OUTPUT`` for BEAM.
    """
    del step_inputs
    output_year = state.year if state.year is not None else state.forecast_year
    vehicles2_path = (
        Path(workspace.get_atlas_output_dir()) / f"vehicles2_{output_year}.csv"
        if output_year is not None
        else None
    )
    outputs: Dict[str, Any] = {
        USIM_POPULATION_SOURCE_H5: _urbansim_population_source_path(
            settings=settings,
            state=state,
            workspace=workspace,
        ),
    }
    if vehicles2_path is not None:
        outputs[ATLAS_VEHICLES2_OUTPUT] = vehicles2_path
    return outputs


def _prefer_base_datastore_key(
    *,
    state: WorkflowState,
    land_use_enabled: bool,
    use_base_datastore: bool,
) -> bool:
    if not use_base_datastore:
        return False
    if not land_use_enabled:
        return True
    try:
        start_year = state.start_year
        year = state.year if state.year is not None else state.forecast_year
    except AttributeError:
        return True
    if start_year is not None and year is not None and int(year) == int(start_year):
        return True
    try:
        main_forecast_year = state.main_forecast_year
    except AttributeError:
        main_forecast_year = None
    try:
        atlas_interval_start_year = state.atlas_interval_start_year
    except AttributeError:
        atlas_interval_start_year = None
    if (
        year is not None
        and main_forecast_year is not None
        and int(year) < int(main_forecast_year)
        and (
            atlas_interval_start_year is None
            or int(year) != int(atlas_interval_start_year)
        )
    ):
        return True
    return False


def _select_atlas_datastore_from_sim_state(
    *,
    sim_state: SimState,
    use_base_datastore: bool,
    prefer_base_key: bool = True,
    expected_filename: str,
) -> ArtifactRef | None:
    if use_base_datastore:
        candidates = (
            (sim_state.land_use.prior_h5, sim_state.land_use.current_h5)
            if prefer_base_key
            else (sim_state.land_use.current_h5, sim_state.land_use.prior_h5)
        )
    else:
        forecast_candidate = sim_state.land_use.forecast_output
        current_candidate = sim_state.land_use.current_h5
        current_name = os.path.basename(artifact_to_path(current_candidate) or "")
        if forecast_candidate is not None and current_name != expected_filename:
            return forecast_candidate
        pop_candidate = sim_state.land_use.population_source
        pop_name = os.path.basename(artifact_to_path(pop_candidate) or "")
        expected_stem = Path(expected_filename).stem
        if (
            pop_candidate is not None
            and pop_name.startswith(expected_stem)
            and current_name != expected_filename
        ):
            return pop_candidate
        candidates = (
            sim_state.land_use.current_h5,
            sim_state.land_use.forecast_output,
            sim_state.land_use.prior_h5,
        )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return sim_state.land_use.current_h5


def _resolve_atlas_datastore_target(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
) -> tuple[bool, Path, ArtifactRef | None]:
    land_use_enabled = bool(
        get_setting(
            settings,
            "land_use_enabled",
            get_land_use_model(settings) is not None
            or get_setting(settings, "urbansim") is not None,
        )
    )
    use_base_datastore = is_start_year(state, settings) or not land_use_enabled
    prefer_base_key = _prefer_base_datastore_key(
        state=state,
        land_use_enabled=land_use_enabled,
        use_base_datastore=use_base_datastore,
    )
    usim_input_fname = get_usim_datastore_fname(
        settings,
        io="input" if (use_base_datastore and prefer_base_key) else "output",
        year=state.forecast_year,
    )
    selected_datastore = _select_atlas_datastore_from_sim_state(
        sim_state=sim_state,
        use_base_datastore=use_base_datastore,
        prefer_base_key=prefer_base_key,
        expected_filename=usim_input_fname,
    )
    mutable_dir = Path(workspace.get_usim_mutable_data_dir())
    if use_base_datastore and prefer_base_key:
        selected_name = os.path.basename(artifact_to_path(selected_datastore) or "")
        try:
            main_forecast_year = state.main_forecast_year
        except AttributeError:
            main_forecast_year = None
        if selected_name.startswith("input_data_for_") and selected_name.endswith(
            "_outputs.h5"
        ):
            datastore_dest = mutable_dir / selected_name
        elif (
            land_use_enabled
            and main_forecast_year is not None
            and state.forecast_year is not None
            and int(state.forecast_year) < int(main_forecast_year)
        ):
            datastore_dest = (
                mutable_dir / f"input_data_for_{int(main_forecast_year)}_outputs.h5"
            )
        else:
            datastore_dest = mutable_dir / usim_input_fname
    else:
        datastore_dest = Path(
            _urbansim_run_native_output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
            )[USIM_DATASTORE_H5]
        )
    return use_base_datastore, datastore_dest, selected_datastore


def _resolve_atlas_preprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState | None = None,
    **_: Any,
) -> StepCallInputs:
    active_sim_state = sim_state if sim_state is not None else SimState()
    requires_beam_skim = requires_prior_beam_skim_handoff(
        settings=settings,
        state=state,
    )
    _, atlas_input, selected_datastore = _resolve_atlas_datastore_target(
        settings=settings,
        state=state,
        workspace=workspace,
        sim_state=active_sim_state,
    )
    return _build_step_call_inputs(
        candidates={
            USIM_DATASTORE_H5: selected_datastore,
            FINAL_SKIMS_OMX: active_sim_state.network.skims_omx,
        },
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
    )


def _resolve_atlas_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState | None = None,
    prior_outputs: Mapping[str, ArtifactRef] | None = None,
    **_: Any,
) -> StepCallInputs:
    active_sim_state = sim_state if sim_state is not None else SimState()
    prepared_inputs = _atlas_preprocess_prepared_input_paths(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    candidates: dict[str, ArtifactRef | None] = {
        key: (
            prior_outputs.get(key)
            if prior_outputs is not None
            else active_sim_state.extras.get(key)
        )
        for key in prepared_inputs
    }
    launch_context = AtlasLaunchContext(
        input_root=Path(workspace.get_atlas_mutable_input_dir()),
        output_root=Path(workspace.get_atlas_output_dir()),
    )
    return _build_step_call_inputs(
        candidates=candidates,
        required_roles=tuple(prepared_inputs),
        optional_roles=(),
        logical_destinations=prepared_inputs,
        metadata={"atlas_launch_context": launch_context},
    )


def _resolve_atlas_postprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState | None = None,
    **_: Any,
) -> StepCallInputs:
    active_sim_state = sim_state if sim_state is not None else SimState()
    forecast_year = state.forecast_year
    _, datastore_dest, selected_datastore = _resolve_atlas_datastore_target(
        settings=settings,
        state=state,
        workspace=workspace,
        sim_state=active_sim_state,
    )
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
        USIM_DATASTORE_H5: datastore_dest,
    }
    candidates: dict[str, ArtifactRef | None] = {
        _ATLAS_HOUSEHOLDV_CSV_INPUT: active_sim_state.atlas.households_with_cars,
        _ATLAS_VEHICLES_CSV_INPUT: active_sim_state.atlas.vehicles,
        USIM_DATASTORE_H5: selected_datastore,
    }
    return _build_step_call_inputs(
        candidates=candidates,
        required_roles=tuple(semantic_roles),
        optional_roles=(),
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
    outputs=_native_contract_output_paths(_atlas_preprocess_native_output_paths),
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


def _recover_prior_atlas_subyear_files(
    *,
    workspace: Workspace,
    state: WorkflowState | None,
    tracker: Tracker | None,
    prior_year: int,
    target_dir: Path,
    expected_files: Sequence[str],
    run_id: str | None = None,
) -> list[str]:
    """Recover missing prior-subyear files from resolved tracker outputs or archive locations.

    Returns the list of filenames that are still missing after all recovery attempts.
    """
    missing = [f for f in expected_files if not (target_dir / f).is_file()]
    if not missing:
        return []

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Attempt recovery from tracker resolved output paths
    if tracker is not None and run_id is not None:
        try:
            resolved_runs = tracker.resolve_run_output_paths(
                [str(run_id)], existing_only=True
            )
            resolved_outputs = resolved_runs.get(str(run_id), {})
            for fname in list(missing):
                for src_path in resolved_outputs.values():
                    candidate = (
                        src_path
                        if src_path.name == fname
                        else (src_path / fname if src_path.is_dir() else None)
                    )
                    if candidate is not None and candidate.is_file():
                        shutil.copy2(candidate, target_dir / fname)
                        missing.remove(fname)
                        break
        except Exception as exc:
            logger.debug(
                "Failed resolving run output paths for run %s: %s", run_id, exc
            )

    if not missing:
        return []

    # 2. Inspect candidate archive roots
    candidate_roots: list[Path] = []
    archive_dir = resolve_archive_run_dir(workspace=workspace, state=state)
    if archive_dir is not None and archive_dir.is_dir():
        candidate_roots.append(archive_dir)

    ws_archive = resolve_archive_run_dir(workspace=workspace)
    if (
        ws_archive is not None
        and ws_archive.is_dir()
        and ws_archive not in candidate_roots
    ):
        candidate_roots.append(ws_archive)

    if isinstance(tracker, Tracker) and tracker.run_dir is not None:
        tr_run_dir = Path(tracker.run_dir).expanduser().resolve()
        if tr_run_dir.is_dir() and tr_run_dir not in candidate_roots:
            candidate_roots.append(tr_run_dir)

    try:
        atlas_rel_folder = (
            Path(workspace.get_atlas_mutable_input_dir())
            .resolve()
            .relative_to(Path(workspace.full_path).resolve())
        )
    except Exception:
        atlas_rel_folder = Path("atlas/atlas_input")

    for root in candidate_roots:
        if not missing:
            break

        # Check direct path under archived atlas input directory
        for rel_dir in dict.fromkeys([atlas_rel_folder, Path("atlas/atlas_input")]):
            archived_year_dir = root / rel_dir / f"year{prior_year}"
            if archived_year_dir.is_dir():
                for fname in list(missing):
                    candidate_file = archived_year_dir / fname
                    if candidate_file.is_file():
                        shutil.copy2(candidate_file, target_dir / fname)
                        missing.remove(fname)

        if not missing:
            break

        # Check consist-recovery scopes
        recovery_root = root / "consist-recovery"
        if recovery_root.is_dir():
            for scope_dir in sorted(recovery_root.glob(f"*atlas*y{prior_year}*")):
                if not scope_dir.is_dir():
                    continue
                for fname in list(missing):
                    candidates = (
                        scope_dir / fname,
                        scope_dir / f"atlas_continuation_{prior_year}" / fname,
                        scope_dir / f"year{prior_year}" / fname,
                        scope_dir / atlas_rel_folder / f"year{prior_year}" / fname,
                        scope_dir
                        / "atlas"
                        / "atlas_input"
                        / f"year{prior_year}"
                        / fname,
                    )
                    for c in candidates:
                        if c.is_file():
                            shutil.copy2(c, target_dir / fname)
                            missing.remove(fname)
                            break

    return missing


def hydrate_atlas_subyear_continuation_outputs(
    *,
    tracker: Tracker,
    settings: PilatesConfig,
    workspace: Workspace,
    prior_year: int,
    start_year: int,
    state: WorkflowState | None = None,
    atlas_run_id: str | None = None,
    preprocess_run_id: str | None = None,
    preserve_existing: bool,
    on_missing: str,
) -> None:
    # ponytail: shared prior-subyear ATLAS continuation hydration used by both restart and native execution
    continuation_dir = (
        Path(workspace.get_atlas_mutable_input_dir()) / f"year{prior_year}"
    )
    continuation_dir.mkdir(parents=True, exist_ok=True)

    if atlas_run_id is not None:
        tracker.hydrate_run_outputs(
            str(atlas_run_id),
            target_root=workspace.full_path,
            keys=[f"atlas_continuation_{prior_year}"],
            preserve_existing=preserve_existing,
            on_missing=on_missing,
        )
        # Consist skips directory artifacts if target directory exists and preserve_existing=True.
        # Recover individual continuation files if they were omitted.
        _recover_prior_atlas_subyear_files(
            workspace=workspace,
            state=state,
            tracker=tracker,
            prior_year=prior_year,
            target_dir=continuation_dir,
            expected_files=("vehicles_output.RData", "households_output.RData"),
            run_id=atlas_run_id,
        )

    if preprocess_run_id is not None:
        prepared_destinations = _atlas_prepared_input_paths_for_year(
            settings=settings,
            workspace=workspace,
            year=prior_year,
            start_year=start_year,
        )
        tracker.hydrate_run_outputs_to_destinations(
            str(preprocess_run_id),
            destinations_by_key=prepared_destinations,
            preserve_existing=preserve_existing,
            on_missing=on_missing,
        )
        _recover_prior_atlas_subyear_files(
            workspace=workspace,
            state=state,
            tracker=tracker,
            prior_year=prior_year,
            target_dir=continuation_dir,
            expected_files=[p.name for p in prepared_destinations.values()],
            run_id=preprocess_run_id,
        )


def _ensure_prior_atlas_subyear_continuation_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    input_root: Path,
) -> None:
    try:
        forecast_year = int(state.forecast_year)
        start_year = int(state.start_year)
    except (AttributeError, TypeError, ValueError):
        return
    if forecast_year <= start_year:
        return

    prior_year = forecast_year - 2
    prior_dir = Path(input_root) / f"year{prior_year}"
    has_rdata = (prior_dir / "vehicles_output.RData").is_file() and (
        prior_dir / "households_output.RData"
    ).is_file()
    has_households = (prior_dir / "households.csv").is_file()
    if has_rdata and has_households:
        return

    tracker = cr.current_tracker()
    atlas_run_id: str | None = None
    preprocess_run_id: str | None = None
    hydration_attempted = False

    if tracker is not None:
        try:
            if not has_rdata:
                atlas_runs = tracker.find_runs(
                    year=prior_year,
                    model="atlas_run",
                    stage="vehicle_ownership",
                    phase="run",
                )
                completed_atlas = [
                    r for r in atlas_runs if getattr(r, "status", None) == "completed"
                ]
                target_atlas = (
                    completed_atlas[0]
                    if completed_atlas
                    else (atlas_runs[0] if atlas_runs else None)
                )
                if target_atlas is not None:
                    atlas_run_id = str(target_atlas.id)

            if not has_households:
                preprocess_runs = tracker.find_runs(
                    year=prior_year,
                    model="atlas_preprocess",
                    stage="vehicle_ownership",
                    phase="preprocess",
                )
                completed_preprocess = [
                    r
                    for r in preprocess_runs
                    if getattr(r, "status", None) == "completed"
                ]
                target_preprocess = (
                    completed_preprocess[0]
                    if completed_preprocess
                    else (preprocess_runs[0] if preprocess_runs else None)
                )
                if target_preprocess is not None:
                    preprocess_run_id = str(target_preprocess.id)

            if atlas_run_id is not None or preprocess_run_id is not None:
                hydrate_atlas_subyear_continuation_outputs(
                    tracker=tracker,
                    settings=settings,
                    workspace=workspace,
                    prior_year=prior_year,
                    start_year=start_year,
                    state=state,
                    atlas_run_id=atlas_run_id,
                    preprocess_run_id=preprocess_run_id,
                    preserve_existing=True,
                    on_missing="warn",
                )
                hydration_attempted = True
        except Exception as exc:
            logger.warning(
                "Encountered error during ATLAS subyear %s continuation hydration: %s",
                prior_year,
                exc,
            )

    # If tracker hydration was not run (e.g. tracker is None or finding runs failed), attempt direct archive recovery
    if not hydration_attempted:
        if (
            not (prior_dir / "vehicles_output.RData").is_file()
            or not (prior_dir / "households_output.RData").is_file()
        ):
            _recover_prior_atlas_subyear_files(
                workspace=workspace,
                state=state,
                tracker=tracker,
                prior_year=prior_year,
                target_dir=prior_dir,
                expected_files=("vehicles_output.RData", "households_output.RData"),
                run_id=atlas_run_id,
            )

        if not (prior_dir / "households.csv").is_file():
            _recover_prior_atlas_subyear_files(
                workspace=workspace,
                state=state,
                tracker=tracker,
                prior_year=prior_year,
                target_dir=prior_dir,
                expected_files=("households.csv",),
                run_id=preprocess_run_id,
            )

    # If prior subyear runs were recorded in provenance, fail fast if their outputs couldn't be recovered
    missing_required: list[str] = []
    if atlas_run_id is not None:
        if not (prior_dir / "vehicles_output.RData").is_file():
            missing_required.append("vehicles_output.RData")
        if not (prior_dir / "households_output.RData").is_file():
            missing_required.append("households_output.RData")
    if preprocess_run_id is not None:
        if not (prior_dir / "households.csv").is_file():
            missing_required.append("households.csv")

    if missing_required:
        raise RuntimeError(
            f"Cannot start ATLAS for forecast year {forecast_year}: prior subyear {prior_year} "
            f"continuation input(s) missing in {prior_dir}: {', '.join(missing_required)}. "
            f"(resolved atlas_run_id={atlas_run_id}, preprocess_run_id={preprocess_run_id})"
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
    _ensure_prior_atlas_subyear_continuation_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        input_root=atlas_launch_context.input_root,
    )
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
    outputs=_native_contract_output_paths(_atlas_postprocess_native_output_paths),
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
    if not state.is_start_year():
        from pilates.workflows.orchestrator import (
            _validate_population_h5_for_activitysim_year,
        )

        _validate_population_h5_for_activitysim_year(
            path=destination_h5,
            year=state.forecast_year,
            context=f"ATLAS postprocess local output y{state.forecast_year}",
        )


def _project_atlas_preprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> AtlasPreprocessOutputs:
    del step_inputs
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
    return projected


def _project_atlas_run(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> AtlasRunOutputs:
    del step_inputs
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
    return projected


def _project_atlas_postprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_inputs: StepCallInputs | None = None,
) -> AtlasPostprocessOutputs:
    del step_inputs
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
    return projected


ATLAS_PREPROCESS = StepDefinition(
    name="atlas_preprocess",
    function=_native_atlas_preprocess,
    resolve_inputs=_resolve_atlas_preprocess_inputs,
    project_outputs=_project_atlas_preprocess,
    input_contract=_ATLAS_PREPROCESS_INPUT_CONTRACT,
    outputs=_atlas_preprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_metadata_cache_options(),
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
    cache_options=_metadata_cache_options(),
)

ATLAS_POSTPROCESS = StepDefinition(
    name="atlas_postprocess",
    function=_native_atlas_postprocess,
    resolve_inputs=_resolve_atlas_postprocess_inputs,
    project_outputs=_project_atlas_postprocess,
    input_contract=_ATLAS_POSTPROCESS_INPUT_CONTRACT,
    outputs=_atlas_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_metadata_cache_options(),
)

# Canonical registry members use the workflow step names. Uppercase aliases keep
# the definitions convenient to import in focused model tests.
atlas_preprocess = ATLAS_PREPROCESS
atlas_run = ATLAS_RUN
atlas_postprocess = ATLAS_POSTPROCESS
