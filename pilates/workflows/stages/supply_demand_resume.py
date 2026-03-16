from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.utils.io import locate_beam_file
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.steps import StepOutputsHolder
from pilates.workspace import Workspace
from workflow_state import WorkflowState

_TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS = (
    "beam_plans_asim_out",
    "households_asim_out",
    "persons_asim_out",
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
) -> Optional[Dict[str, Any]]:
    """
    Rehydrate ActivitySim outputs for BEAM when resuming after a skipped substage.

    On restart directly into ``traffic_assignment``, the current iteration's
    ``StepOutputsHolder`` starts empty even though restart recovery may already
    have restored the required ActivitySim artifacts into the coupler. Promote
    the narrow BEAM-facing subset back into a plain mapping so BEAM preprocess
    sees the same inputs it would have received after a live ActivitySim run.
    """

    def _require_complete_restore(
        restored_outputs: Dict[str, Any], source: str
    ) -> Optional[Dict[str, Any]]:
        if not restored_outputs:
            return None
        missing_keys = sorted(
            key
            for key in _TRAFFIC_ASSIGNMENT_RESUME_REQUIRED_OUTPUTS
            if key not in restored_outputs
            or not artifact_to_path(restored_outputs.get(key), workspace)
            or not os.path.exists(artifact_to_path(restored_outputs.get(key), workspace))
        )
        if missing_keys:
            raise RuntimeError(
                "Restart into traffic_assignment found incomplete ActivitySim "
                f"outputs from {source}; missing {missing_keys}"
            )
        return restored_outputs

    postprocess_outputs = outputs_holder.activitysim_postprocess
    if postprocess_outputs is not None:
        restored_outputs = step_output_handoff_mapping(
            postprocess_outputs,
            coupler=coupler,
        )
        return _require_complete_restore(restored_outputs, "step outputs")

    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return None

    restored_outputs: Dict[str, Any] = {}
    for key in (
        "beam_plans_asim_out",
        "beam_plans_out",
        "households_asim_out",
        "linkstats",
        "persons_asim_out",
    ):
        value = get_value(key)
        if value is None:
            continue
        path = artifact_to_path(value, workspace)
        if path and os.path.exists(path):
            restored_outputs[key] = value
    if restored_outputs:
        outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
            usim_datastore_h5=None,
            asim_output_dir=None,
            processed_outputs={
                key: Path(path)
                for key, value in restored_outputs.items()
                for path in [artifact_to_path(value, workspace)]
                if path is not None
            },
        )
        return _require_complete_restore(restored_outputs, "coupler artifacts")

    iter_dir = Path(workspace.get_asim_output_dir()) / (
        f"year-{state.current_year}-iteration-{state.current_inner_iter}"
    )
    filesystem_candidates = {
        "beam_plans_asim_out": iter_dir / "beam_plans.parquet",
        "households_asim_out": iter_dir / "households.parquet",
        "persons_asim_out": iter_dir / "persons.parquet",
    }
    restored_outputs = {
        key: str(path)
        for key, path in filesystem_candidates.items()
        if path.exists()
    }
    restored_outputs = _require_complete_restore(
        restored_outputs,
        "filesystem iteration outputs",
    )
    if restored_outputs:
        outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
            usim_datastore_h5=None,
            asim_output_dir=iter_dir,
            processed_outputs={
                key: Path(path) for key, path in restored_outputs.items()
            },
        )
    return restored_outputs
