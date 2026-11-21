"""Native orchestrator stage sequencing contracts."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from pilates.workflows import orchestrator
from pilates.workflows.sim_state import SimState


def _function_source(function: object) -> str:
    return inspect.getsource(function)


def _assert_native_sequence(function: object, definitions: tuple[str, ...]) -> None:
    source = _function_source(function)
    offsets = [source.index(f"definition={definition}") for definition in definitions]
    assert offsets == sorted(offsets)


def test_land_use_stage_sequences_native_urbansim_definitions() -> None:
    _assert_native_sequence(
        orchestrator.run_land_use_stage,
        ("urbansim_run", "urbansim_postprocess"),
    )


def test_vehicle_ownership_stage_sequences_native_atlas_definitions() -> None:
    _assert_native_sequence(
        orchestrator.run_vehicle_ownership_stage,
        ("atlas_preprocess", "atlas_run", "atlas_postprocess"),
    )


def test_activity_demand_phase_sequences_native_activitysim_definitions() -> None:
    _assert_native_sequence(
        orchestrator.run_activity_demand_phase,
        ("activitysim_preprocess", "activitysim_run", "activitysim_postprocess"),
    )


def test_beam_phase_sequences_native_beam_definitions() -> None:
    _assert_native_sequence(
        orchestrator.run_traffic_assignment_phase,
        ("beam_preprocess", "beam_run", "beam_postprocess"),
    )


def test_full_skim_uses_its_native_definition() -> None:
    source = _function_source(orchestrator.run_traffic_assignment_phase)
    assert "definition=beam_full_skim" in source


def test_postprocessing_stage_uses_its_native_definition() -> None:
    _assert_native_sequence(orchestrator.run_postprocessing_stage, ("postprocessing",))


def test_direct_native_stage_paths_do_not_restore_legacy_runners() -> None:
    for function in (
        orchestrator.run_land_use_stage,
        orchestrator.run_vehicle_ownership_stage,
        orchestrator.run_activity_demand_phase,
        orchestrator.run_traffic_assignment_phase,
        orchestrator.run_postprocessing_stage,
    ):
        source = _function_source(function)
        assert "StageRunner" not in source
        assert "run_workflow" not in source


def test_supply_demand_loop_uses_imperative_orchestration_without_should_run() -> None:
    source = _function_source(orchestrator._run_supply_demand_loop)
    assert "should_run" not in source
    assert "complete_step" not in source
    assert "Step" + "OutputsHolder" not in source


def test_traffic_assignment_phase_uses_forecast_year_when_set(monkeypatch) -> None:
    captured: list[int] = []

    def fake_execute_step(*, definition, year, **kwargs):
        del kwargs
        captured.append(year)
        if definition.name == "beam_preprocess":
            return SimpleNamespace(outputs={}), SimpleNamespace(prepared_inputs={})
        if definition.name == "beam_run":
            return SimpleNamespace(
                run=SimpleNamespace(id="r1"), outputs={}
            ), SimpleNamespace(raw_outputs={})
        return SimpleNamespace(outputs={}), SimpleNamespace(
            network_final=None,
            linkstats=None,
            final_skims_omx=None,
            zarr_skims=None,
            full_skims=None,
        )

    from dataclasses import replace

    monkeypatch.setattr(orchestrator, "execute_step", fake_execute_step)
    monkeypatch.setattr(
        orchestrator,
        "beam_postprocess",
        replace(
            orchestrator.beam_postprocess,
            resolve_inputs=lambda **_: SimpleNamespace(
                require_complete=lambda *a, **kw: None
            ),
        ),
    )

    state = MagicMock()
    state.year = 2017
    state.forecast_year = 2019
    settings = SimpleNamespace(
        run=SimpleNamespace(models=SimpleNamespace(traffic_assignment="beam")),
        beam=SimpleNamespace(full_skim=SimpleNamespace(run_schedule="off")),
    )

    orchestrator.run_traffic_assignment_phase(
        scenario=MagicMock(),
        settings=settings,
        state=state,
        workspace=MagicMock(),
        sim_state=SimState(),
        year=2017,
        iteration=0,
    )

    assert captured == [2019, 2019, 2019]
