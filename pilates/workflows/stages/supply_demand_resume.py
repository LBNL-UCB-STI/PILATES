from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

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
from pilates.workflows.artifact_keys import ZARR_SKIMS
from pilates.workflows.orchestration import _restore_outputs_from_manifest
from pilates.workflows.outputs_base import (
    step_output_handoff_mapping,
    step_output_mapping,
)
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


def _restore_activity_demand_outputs_for_resume(
    *,
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

    _ = state
    postprocess_outputs = outputs_holder.activitysim_postprocess
    used_manifest_restore = False
    if postprocess_outputs is None and manifest_path is not None:
        resolved_manifest_path = resolve_existing_path(
            str(manifest_path),
            workspace=workspace,
            materialize_from_archive=True,
        )
        if resolved_manifest_path is not None:
            manifest = load_step_manifest(Path(resolved_manifest_path))
        else:
            manifest = None
        if manifest:
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
            restored_outputs = step_output_mapping(
                postprocess_outputs,
                warn_lossy=False,
            )
        else:
            restored_outputs = step_output_handoff_mapping(
                postprocess_outputs,
                coupler=coupler,
            )
        restored_outputs = _promote_archived_zarr_skims(restored_outputs)
        validated = _require_complete_restore(restored_outputs, "step outputs")
        if validated is None:
            return None
        restored_outputs, resolved_paths = validated
        _publish_restored_outputs(restored_outputs, resolved_paths)
        return restored_outputs

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
        return restored_outputs
    raise RuntimeError(
        "Restart into traffic_assignment requires hydrated ActivitySim outputs "
        "in the coupler; missing one or more of "
        f"{sorted(_TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS)}."
    )
