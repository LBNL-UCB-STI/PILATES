from __future__ import annotations

import inspect
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, Mapping, Optional

from pilates.config.models import PilatesConfig
from pilates.utils.coupler_helpers import artifact_to_existing_path
from pilates.workflows.artifact_keys import (
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_INPUT_CONFIG_ARCHIVED,
    BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
    BEAM_INPUT_HOUSEHOLDS_ARCHIVED,
    BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED,
    BEAM_INPUT_PERSONS_ARCHIVED,
    BEAM_INPUT_PLANS_ARCHIVED,
    BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
    BEAM_INPUT_VEHICLES_ARCHIVED,
    BEAM_NETWORK_FINAL,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workspace import Workspace

# Model-specific step factories for BEAM.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PLANS_OUT,
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
    CouplerProtocol,
    StandardStepSpec,
    StepOutputsHolder,
    WorkflowState,
    _beam_log_facet_meta,
    _beam_postprocess_split_facet_meta,
    build_standard_step,
    _log_beam_r5_osm_input,
    _log_step_records,
    _schema_outputs_from_class,
    cr,
    find_last_run_output_plans,
    log_and_set_output,
    log_input_only,
    log_output_only,
)
from pilates.workflows.tracker_outputs import (
    load_tracker_run_outputs,
    merge_canonical_output_mappings,
)


def _primary_beam_config_path(
    settings: PilatesConfig,
    workspace: Workspace,
) -> Path:
    return (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.beam.config
    )


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


def _is_beam_sub_iteration_key(short_name: Optional[str]) -> bool:
    return bool(
        short_name and ("_sub" in short_name or "__beam_sub_iter" in short_name)
    )


def _beam_linkstats_publication_meta(
    short_name: Optional[str],
    *,
    family: str,
) -> Dict[str, Any]:
    if not short_name:
        return {}
    for prefix in ("linkstats_parquet_", "linkstats_"):
        if not short_name.startswith(prefix):
            continue
        tail = short_name[len(prefix) :]
        parts = tail.split("_")
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            iteration = int(parts[1])
        except ValueError:
            continue
        facet: Dict[str, Any] = {
            "artifact_family": family,
            "year": year,
            "iteration": iteration,
        }
        if len(parts) > 2 and parts[2].startswith("sub"):
            try:
                facet["beam_sub_iteration"] = int(parts[2][3:])
            except ValueError:
                continue
        return {
            "facet": facet,
            "facet_schema_version": "v1",
            "facet_index": True,
        }
    return {}


_BEAM_RUN_ARCHIVE_KEY_MAP: Dict[str, str] = {
    BEAM_PLANS_IN: BEAM_INPUT_PLANS_ARCHIVED,
    BEAM_HOUSEHOLDS_IN: BEAM_INPUT_HOUSEHOLDS_ARCHIVED,
    BEAM_PERSONS_IN: BEAM_INPUT_PERSONS_ARCHIVED,
    BEAM_CONFIG_FILE: BEAM_INPUT_CONFIG_ARCHIVED,
    "vehicles_beam_in": BEAM_INPUT_VEHICLES_ARCHIVED,
    LINKSTATS_WARMSTART: BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED,
    BEAM_PLANS_OUT: BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
    BEAM_OUTPUT_PLANS_XML: BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
    BEAM_EXPERIENCED_PLANS_XML: BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML: (
        BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED
    ),
}

_BEAM_RUN_ARCHIVE_DESCRIPTION_MAP: Dict[str, str] = {
    BEAM_INPUT_PLANS_ARCHIVED: "Archived BEAM runner plans input snapshot",
    BEAM_INPUT_HOUSEHOLDS_ARCHIVED: "Archived BEAM runner households input snapshot",
    BEAM_INPUT_PERSONS_ARCHIVED: "Archived BEAM runner persons input snapshot",
    BEAM_INPUT_CONFIG_ARCHIVED: "Archived BEAM runner config input snapshot",
    BEAM_INPUT_VEHICLES_ARCHIVED: "Archived BEAM runner vehicles input snapshot",
    BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED: (
        "Archived BEAM runner warm-start linkstats input snapshot"
    ),
    BEAM_INPUT_PLANS_WARMSTART_ARCHIVED: (
        "Archived BEAM runner warm-start plans input snapshot"
    ),
    BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED: (
        "Archived BEAM runner warm-start experienced plans input snapshot"
    ),
}


def _beam_run_snapshot_dir(
    *,
    workspace: Workspace,
    state: WorkflowState,
) -> Path:
    return (
        Path(workspace.get_beam_output_dir())
        / f"inputs-year-{state.year}-iteration-{state.iteration}"
    )


def _beam_input_archive_meta(
    *,
    archive_key: str,
    year: int,
    iteration: int,
) -> Dict[str, Any]:
    input_name = archive_key.removeprefix("beam_input_").removesuffix("_archived")
    return {
        "facet": {
            "artifact_family": "beam_input_archived",
            "input_name": input_name,
            "year": year,
            "iteration": iteration,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _resolve_existing_coupler_input(
    *,
    coupler: CouplerProtocol,
    key: str,
    workspace: Workspace,
) -> Optional[tuple[str, str]]:
    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return None
    resolved_path = artifact_to_existing_path(
        get_value(key),
        workspace=workspace,
        materialize_from_archive=True,
    )
    if resolved_path is None:
        return None
    return key, resolved_path


def _resolve_beam_run_warmstart_inputs(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> tuple[Optional[tuple[str, str]], Optional[tuple[str, str]]]:
    plans_match = (
        _resolve_existing_coupler_input(
            coupler=coupler,
            key=BEAM_OUTPUT_PLANS_XML,
            workspace=workspace,
        )
        or _resolve_existing_coupler_input(
            coupler=coupler,
            key=BEAM_PLANS_OUT,
            workspace=workspace,
        )
    )
    experienced_match = (
        _resolve_existing_coupler_input(
            coupler=coupler,
            key=BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
            workspace=workspace,
        )
        or _resolve_existing_coupler_input(
            coupler=coupler,
            key=BEAM_EXPERIENCED_PLANS_XML,
            workspace=workspace,
        )
    )

    output_root = Path(workspace.get_beam_output_dir()) / settings.run.region
    if plans_match is None or experienced_match is None:
        scanned_plans_path, scanned_experienced_path = find_last_run_output_plans(
            output_root, "year-"
        )
        if plans_match is None and scanned_plans_path is not None and scanned_plans_path.exists():
            scanned_plans_key = (
                BEAM_OUTPUT_PLANS_XML
                if scanned_plans_path.name == "output_plans.xml.gz"
                else BEAM_PLANS_OUT
            )
            plans_match = (scanned_plans_key, str(scanned_plans_path))
        if (
            experienced_match is None
            and scanned_experienced_path is not None
            and scanned_experienced_path.exists()
        ):
            scanned_experienced_key = (
                BEAM_OUTPUT_EXPERIENCED_PLANS_XML
                if scanned_experienced_path.name == "output_experienced_plans.xml.gz"
                else BEAM_EXPERIENCED_PLANS_XML
            )
            experienced_match = (scanned_experienced_key, str(scanned_experienced_path))
    return plans_match, experienced_match


def _collect_beam_run_snapshot_sources(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    holder: StepOutputsHolder,
    coupler: CouplerProtocol,
) -> Dict[str, Path]:
    upstream = holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")

    snapshot_sources: Dict[str, Path] = {
        BEAM_INPUT_CONFIG_ARCHIVED: _require_primary_beam_config(settings, workspace),
    }
    for short_name, path, _description in upstream._iter_record_items():
        archive_key = _BEAM_RUN_ARCHIVE_KEY_MAP.get(short_name)
        if archive_key is None:
            continue
        snapshot_sources[archive_key] = Path(path)

    plans_match, experienced_match = _resolve_beam_run_warmstart_inputs(
        settings=settings,
        workspace=workspace,
        coupler=coupler,
    )
    if plans_match is not None and Path(plans_match[1]).exists():
        snapshot_sources[BEAM_INPUT_PLANS_WARMSTART_ARCHIVED] = Path(plans_match[1])
    if experienced_match is not None and Path(experienced_match[1]).exists():
        snapshot_sources[BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED] = Path(
            experienced_match[1]
        )
    return snapshot_sources


def _archive_beam_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    holder: StepOutputsHolder,
    coupler: CouplerProtocol,
) -> None:
    snapshot_dir = _beam_run_snapshot_dir(workspace=workspace, state=state)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for archive_key, source_path in _collect_beam_run_snapshot_sources(
        settings=settings,
        workspace=workspace,
        holder=holder,
        coupler=coupler,
    ).items():
        if not source_path.exists():
            raise FileNotFoundError(
                f"BEAM run input snapshot source is missing for {archive_key}: {source_path}"
            )
        target_path = snapshot_dir / f"{archive_key}{''.join(source_path.suffixes)}"
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)
        log_output_only(
            key=archive_key,
            path=str(target_path),
            description=_BEAM_RUN_ARCHIVE_DESCRIPTION_MAP[archive_key],
            step_name="beam_run",
            **_beam_input_archive_meta(
                archive_key=archive_key,
                year=state.year,
                iteration=state.iteration,
            ),
        )


def _publish_beam_run_outputs(
    *,
    outputs: BeamRunOutputs,
    coupler: CouplerProtocol,
) -> None:
    promoted_linkstats = outputs.promoted_linkstats_for_publication()
    if promoted_linkstats is not None:
        source_key, path = promoted_linkstats
        linkstats_meta = _beam_linkstats_publication_meta(
            source_key,
            family="linkstats",
        )
        log_and_set_output(
            key=LINKSTATS,
            path=str(path),
            description="BEAM linkstats output for downstream runs",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
            **linkstats_meta,
        )
        log_and_set_output(
            key=LINKSTATS_WARMSTART,
            path=str(path),
            description="BEAM warm-start linkstats for downstream runs",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
            **linkstats_meta,
        )

    for short_name, path in outputs.iter_linkstats_parquet_outputs():
        linkstats_meta = _beam_log_facet_meta(short_name)
        if _is_beam_sub_iteration_key(short_name):
            log_output_only(
                key=short_name,
                path=str(path),
                description="BEAM linkstats parquet output for downstream runs",
                step_name="beam_run",
                profile_file_schema=True,
                **linkstats_meta,
            )
            continue
        log_and_set_output(
            key=short_name,
            path=str(path),
            description="BEAM linkstats parquet output for downstream runs",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
            **linkstats_meta,
        )

    for short_name, path in outputs.iter_unmodified_phys_sim_outputs():
        record_meta = _beam_log_facet_meta(short_name)
        if _is_beam_sub_iteration_key(short_name):
            log_output_only(
                key=short_name,
                path=str(path),
                description=(
                    "BEAM unmodified linkstats parquet output for phys sim "
                    "sub-iteration"
                ),
                step_name="beam_run",
                profile_file_schema=True,
                **record_meta,
            )
            continue
        log_and_set_output(
            key=short_name,
            path=str(path),
            description=(
                "BEAM unmodified linkstats parquet output for phys sim sub-iteration"
            ),
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
            **record_meta,
        )

    promoted_plans = outputs.promoted_plans_for_publication()
    if promoted_plans is not None:
        _, path = promoted_plans
        log_and_set_output(
            key=BEAM_PLANS_OUT,
            path=str(path),
            description="BEAM plans output for downstream runs",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
        )

    promoted_output_plans_xml = outputs.promoted_output_plans_xml_for_publication()
    if promoted_output_plans_xml is not None:
        _, path = promoted_output_plans_xml
        log_and_set_output(
            key=BEAM_OUTPUT_PLANS_XML,
            path=str(path),
            description="BEAM output plans XML for downstream warm-start reuse",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
        )

    promoted_output_experienced_plans_xml = (
        outputs.promoted_output_experienced_plans_xml_for_publication()
    )
    if promoted_output_experienced_plans_xml is not None:
        _, path = promoted_output_experienced_plans_xml
        log_and_set_output(
            key=BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
            path=str(path),
            description=(
                "BEAM output experienced plans XML for downstream warm-start reuse"
            ),
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
        )

    promoted_experienced_plans_xml = (
        outputs.promoted_experienced_plans_xml_for_publication()
    )
    if promoted_experienced_plans_xml is not None:
        _, path = promoted_experienced_plans_xml
        log_and_set_output(
            key=BEAM_EXPERIENCED_PLANS_XML,
            path=str(path),
            description="BEAM experienced plans XML for downstream warm-start reuse",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
        )


def _execute_beam_preprocess(
    preprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    activity_demand_outputs: Optional[Dict[str, Any]] = None,
    previous_beam_outputs: Optional[Dict[str, Any]] = None,
    beam_preprocess_inputs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> BeamPreprocessOutputs:
    return preprocessor.preprocess(
        workspace,
        activity_demand_outputs=activity_demand_outputs,
        previous_beam_outputs=previous_beam_outputs,
        beam_preprocess_inputs=beam_preprocess_inputs,
    )


def _execute_beam_run(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    extra_inputs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> BeamRunOutputs:
    upstream = outputs_holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")
    if not isinstance(upstream, BeamPreprocessOutputs):
        raise TypeError("beam_run requires BeamPreprocessOutputs from beam_preprocess")
    return runner.run(
        upstream,
        workspace,
        extra_inputs=extra_inputs,
    )


def _execute_beam_postprocess(
    postprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    zarr_skims: Optional[Any] = None,
    **_: Any,
) -> BeamPostprocessOutputs:
    upstream = outputs_holder.beam_run
    if upstream is None:
        raise RuntimeError("BEAM run must complete first")
    if not isinstance(upstream, BeamRunOutputs):
        raise TypeError("beam_postprocess requires BeamRunOutputs from beam_run")
    if zarr_skims is not None:
        try:
            parameters = inspect.signature(postprocessor.postprocess).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_zarr_skims = "zarr_skims" in parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in parameters.values()
        )
        if accepts_zarr_skims:
            return postprocessor.postprocess(
                upstream,
                workspace,
                zarr_skims=zarr_skims,
            )
    return postprocessor.postprocess(upstream, workspace)


def _execute_beam_full_skim(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    previous_beam_outputs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> BeamFullSkimOutputs:
    upstream = outputs_holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")
    if not isinstance(upstream, BeamPreprocessOutputs):
        raise TypeError(
            "beam_full_skim requires BeamPreprocessOutputs from beam_preprocess"
        )
    return runner.run(
        upstream,
        workspace,
        previous_beam_outputs=previous_beam_outputs,
    )


def _resolve_cached_run_outputs(run_id: Optional[str]) -> Dict[str, Any]:
    return load_tracker_run_outputs(run_id)


def _recovered_cached_paths(
    *,
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
    workspace: Workspace,
) -> Dict[str, Path]:
    merged = merge_canonical_output_mappings(
        cached_outputs,
        _resolve_cached_run_outputs(run_id),
    )
    recovered: Dict[str, Path] = {}
    for key, value in merged.items():
        path = artifact_to_existing_path(
            value,
            workspace=workspace,
            materialize_from_archive=True,
        )
        if path is not None:
            recovered[key] = Path(path)
    return recovered


def _recover_beam_run_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[BeamRunOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    if not recovered_paths:
        return None
    return BeamRunOutputs(
        beam_output_dir=Path(workspace.get_beam_output_dir()),
        raw_outputs=recovered_paths,
    )


def _recover_beam_postprocess_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[BeamPostprocessOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    if not recovered_paths:
        return None
    return BeamPostprocessOutputs(
        zarr_skims=recovered_paths.get("zarr_skims"),
        final_skims_omx=recovered_paths.get("final_skims_omx"),
        split_events={
            key: path
            for key, path in recovered_paths.items()
            if key.startswith("events_parquet_") and "_type_" in key
        },
        split_event_links={
            key: path
            for key, path in recovered_paths.items()
            if key.startswith("path_traversal_links_")
        },
    )


def _recover_beam_full_skim_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[BeamFullSkimOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    full_skims = recovered_paths.get("beam_full_skims")
    if full_skims is None:
        return None
    return BeamFullSkimOutputs(full_skims=full_skims)


def make_beam_preprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM preprocess step function.

    This step builds the BEAM scenario inputs by transforming ActivitySim
    demand outputs, adding ATLAS vehicles (if enabled), and staging warm-start
    artifacts such as linkstats.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for BEAM preprocess.

    Notes
    -----
    This step focuses on generating BEAM inputs and canonicalizing BEAM config.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        _require_primary_beam_config(settings, workspace)
        return {}

    def _log_outputs(
        outputs: BeamPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        """
        Log BEAM preprocess outputs and update the coupler.

        This helper logs prepared BEAM input artifacts into the coupler for
        downstream BEAM run and postprocess steps.

        Parameters
        ----------
        outputs : BeamPreprocessOutputs
            Typed outputs containing prepared BEAM inputs.
        settings : PilatesConfig
            Simulation settings for config root resolution.
        state : WorkflowState
            Current workflow state (used for log metadata only).
        workspace : Workspace
            Workspace used to resolve mutable BEAM config paths.
        holder : StepOutputsHolder
            Outputs holder (unused for this helper).
        """
        _log_step_records(
            record_items=(
                (
                    key,
                    path,
                    f"BEAM prepared input {key} for year {state.year}, iter {state.iteration}",
                )
                for key, path in outputs.prepared_inputs.items()
            ),
            log_fn=lambda key, path, description, **meta: log_and_set_output(
                key=key,
                path=path,
                description=description,
                coupler=coupler,
                step_name="beam_preprocess",
                **meta,
            ),
            profile_schema_keys={
                "plans_beam_in",
                "vehicles_beam_in",
            },
        )

    return build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="beam_preprocess",
            model_name="beam",
            phase="preprocess",
            outputs_class=BeamPreprocessOutputs,
            component_getter=lambda factory, state: factory.get_preprocessor("beam", state),
            component_executor=lambda component, workspace, outputs_holder, **kwargs: (
                _execute_beam_preprocess(
                    component,
                    workspace,
                    outputs_holder,
                    **kwargs,
                )
            ),
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            schema_outputs=_schema_outputs_from_class(BeamPreprocessOutputs),
            use_logged_wrapper=False,
        ),
    )


def make_beam_run_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM run step function.

    This step performs the traffic assignment simulation for the current
    iteration and produces linkstats, skims, plans, and event outputs.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for BEAM run.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.beam_preprocess
        if upstream is None:
            raise RuntimeError("BEAM preprocess must complete first")

        config_path = _require_primary_beam_config(settings, workspace)
        log_input_only(
            key=BEAM_CONFIG_FILE,
            path=str(config_path),
            description="BEAM config file consumed by the BEAM run",
        )

        tracker = cr.current_tracker()
        if tracker is not None:
            _log_beam_r5_osm_input(
                tracker=tracker,
                settings=settings,
                workspace=workspace,
            )
        for short_name, path, description in upstream._iter_record_items():
            log_input_only(
                key=short_name,
                path=str(path),
                description=description,
            )

        plans_match, experienced_match = _resolve_beam_run_warmstart_inputs(
            settings=settings,
            workspace=workspace,
            coupler=coupler,
        )
        if plans_match is not None and Path(plans_match[1]).exists():
            log_input_only(
                key=plans_match[0],
                path=plans_match[1],
                description=(
                    "BEAM warm-start plans (selected by BEAM from previous outputs)"
                ),
            )
        if experienced_match is not None and Path(experienced_match[1]).exists():
            log_input_only(
                key=experienced_match[0],
                path=experienced_match[1],
                description=(
                    "BEAM warm-start experienced plans (selected by BEAM from previous outputs)"
                ),
            )
        return {}

    def _log_outputs(
        outputs: BeamRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        def _beam_run_extra_meta(
            short_name: str,
            _path: str,
            _description: str,
        ) -> Dict[str, Any]:
            meta: Dict[str, Any] = {}
            facet_meta = _beam_log_facet_meta(short_name)
            if facet_meta:
                meta.update(facet_meta)
            if short_name == BEAM_NETWORK_FINAL:
                meta.update(
                    {
                        "profile_file_schema": "if_changed",
                        "reuse_if_unchanged": True,
                        "reuse_scope": "any_uri",
                    }
                )
                beam_network_schema: Any = None
                try:
                    from pilates.database.schema.beam_schema import BeamNetworkFinal
                except Exception:
                    beam_network_schema = None
                else:
                    beam_network_schema = BeamNetworkFinal
                if beam_network_schema is not None:
                    meta["schema"] = beam_network_schema
            return meta

        _archive_beam_run_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            holder=holder,
            coupler=coupler,
        )

        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=lambda key, path, description, **meta: log_output_only(
                key=key,
                path=path,
                description=description,
                step_name="beam_run",
                **meta,
            ),
            extra_meta_fn=_beam_run_extra_meta,
        )

    return build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="beam_run",
            model_name="beam",
            phase="run",
            outputs_class=BeamRunOutputs,
            component_getter=lambda factory, state: factory.get_runner("beam", state),
            component_executor=lambda component, workspace, outputs_holder, **kwargs: (
                _execute_beam_run(
                    component,
                    workspace,
                    outputs_holder,
                    **kwargs,
                )
            ),
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_beam_run_outputs,
            schema_outputs=_schema_outputs_from_class(BeamRunOutputs),
            use_logged_wrapper=False,
        ),
    )


def make_beam_postprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM postprocess step function.

    This step merges BEAM outputs into updated skims and produces final
    skim artifacts for ActivitySim and UrbanSim inputs.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for BEAM postprocess.
    """

    def _log_outputs(
        outputs: BeamPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
                step_name="beam_postprocess",
            )
        for short_name, path in outputs.split_events.items():
            facet_meta = _beam_postprocess_split_facet_meta(short_name)
            log_output_only(
                key=short_name,
                path=str(path),
                description=f"BEAM events parquet split ({short_name})",
                step_name="beam_postprocess",
                profile_file_schema=True,
                **facet_meta,
            )
        for short_name, path in outputs.split_event_links.items():
            facet_meta = _beam_postprocess_split_facet_meta(short_name)
            log_output_only(
                key=short_name,
                path=str(path),
                description=f"BEAM events link table ({short_name})",
                step_name="beam_postprocess",
                profile_file_schema=True,
                **facet_meta,
            )
        upstream = holder.beam_run
        if upstream is None:
            return
        _publish_beam_run_outputs(outputs=upstream, coupler=coupler)

    return build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="beam_postprocess",
            model_name="beam",
            phase="postprocess",
            outputs_class=BeamPostprocessOutputs,
            component_getter=lambda factory, state: factory.get_postprocessor(
                "beam", state
            ),
            component_executor=lambda component, workspace, outputs_holder, **kwargs: (
                _execute_beam_postprocess(
                    component,
                    workspace,
                    outputs_holder,
                    **kwargs,
                )
            ),
            declared_outputs=[ZARR_SKIMS],
            output_logger=_log_outputs,
            output_recoverer=_recover_beam_postprocess_outputs,
            schema_outputs=_schema_outputs_from_class(BeamPostprocessOutputs),
            use_logged_wrapper=False,
        ),
    )


def make_beam_full_skim_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM full-skim step function.

    This step runs BEAM's FullSkimsCreatorApp to produce background skims
    from prepared BEAM inputs and optional warm-start linkstats.
    """

    def _log_outputs(
        outputs: BeamFullSkimOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
                step_name="beam_full_skim",
            )

    return build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="beam_full_skim",
            model_name="beam_full",
            phase="skim",
            outputs_class=BeamFullSkimOutputs,
            component_getter=lambda factory, state: factory.get_runner(
                "beam_full_skim", state
            ),
            component_executor=lambda component, workspace, outputs_holder, **kwargs: (
                _execute_beam_full_skim(
                    component,
                    workspace,
                    outputs_holder,
                    **kwargs,
                )
            ),
            output_logger=_log_outputs,
            output_recoverer=_recover_beam_full_skim_outputs,
            schema_outputs=_schema_outputs_from_class(BeamFullSkimOutputs),
            use_logged_wrapper=False,
        ),
    )
