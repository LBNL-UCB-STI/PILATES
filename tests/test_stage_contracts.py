"""Native stage sequencing contracts.

The individual native definitions and ``test_stubbed_workflow_matrix`` own
fixture-heavy execution coverage.  These tests deliberately keep the stage
boundary assertion narrow: stages sequence committed definitions through the
single ``execute_step`` seam and do not reconstruct legacy runner, holder, or
manifest state.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import h5py

from pilates.workflows.stages import land_use
from pilates.workflows.stages import postprocessing
from pilates.workflows.stages import supply_demand
from pilates.workflows.stages import supply_demand_activity
from pilates.workflows.stages import supply_demand_beam
from pilates.workflows.stages import vehicle_ownership


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".h5":
        with h5py.File(path, "w") as handle:
            handle.create_dataset("dummy", data=[1])
        return
    path.write_text(content)


def _function_source(function: object) -> str:
    return inspect.getsource(function)


def _assert_native_sequence(function: object, definitions: tuple[str, ...]) -> None:
    source = _function_source(function)
    offsets = [source.index(f"definition={definition}") for definition in definitions]
    assert offsets == sorted(offsets)
    assert source.count("execute_step(") == len(definitions)


def test_land_use_stage_sequences_native_urbansim_definitions() -> None:
    _assert_native_sequence(
        land_use.run_land_use_stage,
        ("urbansim_preprocess", "urbansim_run", "urbansim_postprocess"),
    )


def test_vehicle_ownership_stage_sequences_native_atlas_definitions() -> None:
    _assert_native_sequence(
        vehicle_ownership.run_vehicle_ownership_stage,
        ("atlas_preprocess", "atlas_run", "atlas_postprocess"),
    )


def test_activity_demand_phase_sequences_native_activitysim_definitions() -> None:
    _assert_native_sequence(
        supply_demand_activity._run_activity_demand_phase,
        ("activitysim_preprocess", "activitysim_run", "activitysim_postprocess"),
    )


def test_beam_phase_sequences_native_beam_definitions() -> None:
    _assert_native_sequence(
        supply_demand_beam._run_beam_steps,
        ("beam_preprocess", "beam_run", "beam_postprocess"),
    )


def test_full_skim_uses_its_native_definition() -> None:
    _assert_native_sequence(
        supply_demand_beam._run_beam_full_skim_step, ("beam_full_skim",)
    )


def test_postprocessing_stage_uses_its_native_definition() -> None:
    _assert_native_sequence(
        postprocessing.run_postprocessing_stage, ("postprocessing",)
    )


def test_direct_native_stage_paths_do_not_restore_legacy_runners() -> None:
    for function in (
        land_use.run_land_use_stage,
        vehicle_ownership.run_vehicle_ownership_stage,
        supply_demand_activity._run_activity_demand_phase,
        supply_demand_beam._run_beam_steps,
        postprocessing.run_postprocessing_stage,
    ):
        source = _function_source(function)
        assert "StageRunner" not in source
        assert "run_workflow" not in source


def test_supply_demand_rewinds_skipped_activitysim_without_beam_checkpoint() -> None:
    source = _function_source(supply_demand.run_supply_demand_stage)
    assert "_rewind_uncheckpointed_activitysim_restart" in source
    assert "_restore_activity_demand_outputs_for_resume" not in source
    assert "Step" + "OutputsHolder" not in source
