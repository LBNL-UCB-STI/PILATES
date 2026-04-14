from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    resolve_existing_path,
    resolve_artifact_from_value,
    set_coupler_from_artifact,
)
from pilates.utils.io import locate_beam_file
from pilates.utils.step_manifest import load_step_manifest
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.orchestration import _restore_outputs_from_manifest
from pilates.workflows.outputs_base import (
    iter_step_output_items,
    step_output_handoff_mapping,
)
from pilates.generic.records import sanitize_artifact_key
from pilates.workflows.steps import StepOutputsHolder
from pilates.workspace import Workspace
from workflow_state import WorkflowState

_TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS = (
    "beam_plans_asim_out",
    "households_asim_out",
    "persons_asim_out",
    ZARR_SKIMS,
)
_ARCHIVED_ZARR_SKIMS_KEY = "asim_input_skims_zarr_archived"


def _resolved_existing_restore_path(value: Any, workspace: Workspace) -> Optional[str]:
    return artifact_to_existing_path(
        value,
        workspace=workspace,
        materialize_from_archive=True,
    )


def _find_input_scenario_dir(
    settings: PilatesConfig,
    workspace: Workspace,
    filename: str,
    filetype: str = "parquet",
) -> str:
    beam_settings = settings.beam
    if beam_settings is None:
        raise RuntimeError("BEAM config is required for traffic-assignment inputs.")
    scenario_dir = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
        beam_settings.scenario_folder,
    )
    return locate_beam_file(scenario_dir, filename, filetype)


def _restore_supply_demand_usim_inputs_for_resume(
    *,
    coupler: CouplerProtocol,
    workspace: Workspace,
    state: WorkflowState,
    settings: PilatesConfig,
) -> Dict[str, str]:
    """
    Rebuild the year-scoped UrbanSim H5 roles needed after land use is skipped.

    On fresh-workspace restart, the coupler may not yet carry the resolved H5
    handles that land use would normally publish for later ActivitySim stages.
    Reconstruct those roles from deterministic workspace/archive paths and
    republish them into the coupler.
    """

    forecast_year = getattr(state, "forecast_year", None)
    if forecast_year is None:
        return {}
    get_usim_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    if not callable(get_usim_dir):
        return {}

    from pilates.urbansim.postprocessor import get_usim_datastore_fname

    usim_data_dir = Path(get_usim_dir())
    base_candidate = usim_data_dir / get_usim_datastore_fname(settings, io="input")
    current_candidate = usim_data_dir / get_usim_datastore_fname(
        settings,
        io="output",
        year=forecast_year,
    )

    restored: Dict[str, str] = {}
    get_value = getattr(coupler, "get", None)
    set_value = getattr(coupler, "set", None)
    coupler_current = get_value(USIM_DATASTORE_CURRENT_H5) if callable(get_value) else None
    coupler_population = get_value(USIM_POPULATION_SOURCE_H5) if callable(get_value) else None
    if coupler_current is None and coupler_population is not None and callable(set_value):
        set_value(USIM_DATASTORE_CURRENT_H5, coupler_population)
        coupler_current = get_value(USIM_DATASTORE_CURRENT_H5) if callable(get_value) else coupler_population
    if coupler_population is None and coupler_current is not None and callable(set_value):
        set_value(USIM_POPULATION_SOURCE_H5, coupler_current)
        coupler_population = get_value(USIM_POPULATION_SOURCE_H5) if callable(get_value) else coupler_current
    resolved_coupler_current = _resolved_existing_restore_path(coupler_current, workspace)
    resolved_coupler_population = _resolved_existing_restore_path(
        coupler_population,
        workspace,
    )
    base_path = resolve_existing_path(
        str(base_candidate),
        workspace=workspace,
        materialize_from_archive=True,
    )
    current_path = resolve_existing_path(
        str(current_candidate),
        workspace=workspace,
        materialize_from_archive=True,
    )
    if base_path:
        restored[USIM_DATASTORE_BASE_H5] = base_path
    if resolved_coupler_current:
        restored[USIM_DATASTORE_CURRENT_H5] = resolved_coupler_current
    elif resolved_coupler_population:
        restored[USIM_DATASTORE_CURRENT_H5] = resolved_coupler_population
    elif current_path:
        restored[USIM_DATASTORE_CURRENT_H5] = current_path
    elif base_path:
        restored[USIM_DATASTORE_CURRENT_H5] = base_path

    if resolved_coupler_population:
        restored[USIM_POPULATION_SOURCE_H5] = resolved_coupler_population
    elif restored.get(USIM_DATASTORE_CURRENT_H5):
        restored[USIM_POPULATION_SOURCE_H5] = restored[USIM_DATASTORE_CURRENT_H5]

    if current_path:
        restored[USIM_FORECAST_OUTPUT] = current_path

    for key, path in restored.items():
        set_coupler_from_artifact(
            coupler,
            key,
            None,
            fallback=path,
        )
    return restored


def _restore_activity_demand_outputs_for_resume(
    *,
    scenario: Optional[Any] = None,
    coupler: CouplerProtocol,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    state: WorkflowState,
    settings: PilatesConfig,
    manifest_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Rehydrate ActivitySim outputs for BEAM when resuming after a skipped substage.

    On restart directly into ``traffic_assignment``, the current iteration's
    ``StepOutputsHolder`` starts empty even though restart recovery may already
    have restored the required ActivitySim artifacts into the coupler. Promote
    the narrow BEAM-facing subset back into a plain mapping so BEAM preprocess
    sees the same inputs it would have received after a live ActivitySim run.
    """

    def _publish_restored_outputs(
        restored_outputs: Dict[str, Any],
        resolved_paths: Dict[str, Path],
    ) -> None:
        for key, path in resolved_paths.items():
            value = restored_outputs.get(key)
            resolved_artifact = resolve_artifact_from_value(
                value,
                key=key,
                workspace=workspace,
            )
            artifact = (
                resolved_artifact
                if (
                    hasattr(resolved_artifact, "container_uri")
                    or hasattr(resolved_artifact, "uri")
                )
                else None
            )
            set_coupler_from_artifact(
                coupler,
                key,
                artifact,
                fallback=str(path),
            )

    def _promote_archived_zarr_skims(
        restored_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Promote the archived ActivitySim skims input into the BEAM contract.

        ActivitySim postprocess persists the archived Zarr input under
        ``asim_input_skims_zarr_archived``. BEAM restart expects the same
        artifact under the canonical ``zarr_skims`` key, so make that mapping
        explicit before validation.
        """
        if ZARR_SKIMS in restored_outputs:
            return restored_outputs
        processed_outputs = getattr(postprocess_outputs, "processed_outputs", None) or {}
        archived_zarr = processed_outputs.get(_ARCHIVED_ZARR_SKIMS_KEY)
        if archived_zarr is None:
            return restored_outputs
        promoted = dict(restored_outputs)
        promoted[ZARR_SKIMS] = archived_zarr
        return promoted

    def _resolved_restore_outputs(
        restored_outputs: Dict[str, Any],
        resolved_paths: Dict[str, Path],
    ) -> Dict[str, Any]:
        return {
            key: str(resolved_paths.get(key, value))
            for key, value in restored_outputs.items()
        }

    def _remember_restored_activitysim_run_id(
        manifest_data: Optional[Mapping[str, Any]],
    ) -> None:
        if not isinstance(manifest_data, Mapping):
            return
        remember_restored_run_id = getattr(scenario, "remember_restored_run_id", None)
        if not callable(remember_restored_run_id):
            return
        activitysim_run = manifest_data.get("activitysim_run")
        if not isinstance(activitysim_run, Mapping):
            return
        run_id = activitysim_run.get("run_id")
        if not run_id:
            return
        remember_restored_run_id(
            model_name="activitysim_run",
            year=getattr(state, "forecast_year", None) or getattr(state, "year", None),
            iteration=getattr(state, "current_inner_iter", None),
            run_id=run_id,
        )

    def _require_complete_restore(
        restored_outputs: Dict[str, Any], source: str
    ) -> Optional[tuple[Dict[str, Any], Dict[str, Path]]]:
        if not restored_outputs:
            return None
        resolved_paths = {
            key: _resolved_existing_restore_path(value, workspace)
            for key, value in restored_outputs.items()
        }
        missing_keys = sorted(
            key
            for key in _TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS
            if key not in restored_outputs
            or not resolved_paths.get(key)
        )
        if missing_keys:
            raise RuntimeError(
                "Restart into traffic_assignment found incomplete ActivitySim "
                f"outputs from {source}; missing {missing_keys}"
            )
        return restored_outputs, {
            key: Path(path)
            for key, path in resolved_paths.items()
            if path is not None
        }

    def _resolved_resume_manifest_path() -> Optional[str]:
        if manifest_path is None:
            return None
        resolved_manifest_path = resolve_existing_path(
            str(manifest_path),
            workspace=workspace,
            materialize_from_archive=True,
        )
        if resolved_manifest_path is not None:
            return resolved_manifest_path
        archive_state_path = getattr(state, "file_loc", None)
        if not archive_state_path:
            return None
        try:
            manifest_relative = Path(manifest_path).resolve().relative_to(
                Path(workspace.full_path).resolve()
            )
        except Exception:
            return None
        archive_state_root = Path(archive_state_path).resolve().parent
        archive_workspace_candidates = (
            archive_state_root / "run",
            archive_state_root,
        )
        for archive_workspace_root in archive_workspace_candidates:
            archive_manifest_path = archive_workspace_root / manifest_relative
            if archive_manifest_path.exists():
                return str(archive_manifest_path)
        return None

    def _manifest_handoff_mapping(
        outputs_data: Mapping[str, Any],
    ) -> Dict[str, Any]:
        restored: Dict[str, Any] = {}
        processed_outputs = outputs_data.get("processed_outputs")
        if isinstance(processed_outputs, Mapping):
            for key, value in processed_outputs.items():
                sanitized_key = sanitize_artifact_key(str(key))
                if sanitized_key is None or sanitized_key in restored:
                    continue
                restored[sanitized_key] = value
        archived_zarr = restored.get(_ARCHIVED_ZARR_SKIMS_KEY)
        if ZARR_SKIMS not in restored and archived_zarr is not None:
            restored[ZARR_SKIMS] = archived_zarr
        return restored

    def _supplement_from_coupler(
        restored_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        get_value = getattr(coupler, "get", None)
        if not callable(get_value):
            return restored_outputs
        supplemented = dict(restored_outputs)
        for key in _TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS:
            if key in supplemented:
                continue
            value = get_value(key)
            if value is None:
                continue
            if _resolved_existing_restore_path(value, workspace) is None:
                continue
            supplemented[key] = value
        return supplemented

    _ = state
    postprocess_outputs = outputs_holder.activitysim_postprocess
    used_manifest_restore = False
    manifest_outputs_data: Optional[Mapping[str, Any]] = None
    if postprocess_outputs is None and manifest_path is not None:
        resolved_manifest_path = _resolved_resume_manifest_path()
        if resolved_manifest_path is not None:
            manifest = load_step_manifest(Path(resolved_manifest_path))
        else:
            manifest = None
        if manifest:
            _remember_restored_activitysim_run_id(manifest)
            manifest_step_info = manifest.get("activitysim_postprocess", {})
            outputs_data = manifest_step_info.get("outputs", {})
            if isinstance(outputs_data, Mapping):
                manifest_outputs_data = outputs_data
            postprocess_outputs = _restore_outputs_from_manifest(
                "activitysim_postprocess",
                manifest,
                workspace,
                settings=settings,
                state=state,
            )
            if postprocess_outputs is not None:
                outputs_holder.activitysim_postprocess = postprocess_outputs
                used_manifest_restore = True
    if postprocess_outputs is not None:
        if used_manifest_restore:
            # Manifest recovery should be path-driven so stale coupler values from
            # the previous workspace cannot override the restored outputs.
            restored_outputs = {}
            for key, path, _ in iter_step_output_items(postprocess_outputs):
                sanitized_key = sanitize_artifact_key(key)
                if sanitized_key is None or sanitized_key in restored_outputs:
                    continue
                restored_outputs[sanitized_key] = str(path)
            restored_outputs = _supplement_from_coupler(
                _promote_archived_zarr_skims(restored_outputs)
            )
            manifest_restored_outputs = None
            if manifest_outputs_data:
                manifest_restored_outputs = _supplement_from_coupler(
                    _promote_archived_zarr_skims(
                        _manifest_handoff_mapping(manifest_outputs_data)
                    )
                )
            typed_missing = sorted(
                key
                for key in _TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS
                if key not in restored_outputs
                or not _resolved_existing_restore_path(
                    restored_outputs.get(key), workspace
                )
            )
            if typed_missing and manifest_restored_outputs:
                restored_outputs = manifest_restored_outputs
                validated = _require_complete_restore(
                    restored_outputs,
                    "manifest outputs",
                )
                if validated is not None:
                    restored_outputs, resolved_paths = validated
                    outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
                        usim_datastore_h5=getattr(
                            postprocess_outputs, "usim_datastore_h5", None
                        ),
                        asim_output_dir=getattr(
                            postprocess_outputs, "asim_output_dir", None
                        ),
                        processed_outputs=resolved_paths,
                    )
                    _publish_restored_outputs(restored_outputs, resolved_paths)
                    return _resolved_restore_outputs(
                        restored_outputs,
                        resolved_paths,
                    )
        else:
            restored_outputs = step_output_handoff_mapping(
                postprocess_outputs,
                coupler=coupler,
            )
        restored_outputs = _supplement_from_coupler(
            _promote_archived_zarr_skims(restored_outputs)
        )
        validated = _require_complete_restore(restored_outputs, "step outputs")
        if validated is None:
            return None
        restored_outputs, resolved_paths = validated
        _publish_restored_outputs(restored_outputs, resolved_paths)
        return _resolved_restore_outputs(restored_outputs, resolved_paths)

    if manifest_outputs_data:
        restored_outputs = _supplement_from_coupler(
            _manifest_handoff_mapping(manifest_outputs_data)
        )
        validated = _require_complete_restore(restored_outputs, "manifest outputs")
        if validated is not None:
            restored_outputs, resolved_paths = validated
            outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
                usim_datastore_h5=None,
                asim_output_dir=None,
                processed_outputs=resolved_paths,
            )
            _publish_restored_outputs(restored_outputs, resolved_paths)
            return _resolved_restore_outputs(restored_outputs, resolved_paths)

    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        raise RuntimeError(
            "Restart into traffic_assignment requires coupler access to restored "
            "ActivitySim outputs, but the coupler does not expose get()."
        )

    restored_outputs: Dict[str, Any] = {}
    for key in _TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS:
        value = get_value(key)
        if value is None:
            continue
        if _resolved_existing_restore_path(value, workspace):
            restored_outputs[key] = value
    validated = _require_complete_restore(restored_outputs, "coupler artifacts")
    if validated:
        restored_outputs, resolved_paths = validated
        outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
            usim_datastore_h5=None,
            asim_output_dir=None,
            processed_outputs=resolved_paths,
        )
        _publish_restored_outputs(restored_outputs, resolved_paths)
        return _resolved_restore_outputs(restored_outputs, resolved_paths)
    raise RuntimeError(
        "Restart into traffic_assignment requires hydrated ActivitySim outputs "
        "in the coupler; missing one or more of "
        f"{sorted(_TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS)}."
    )
