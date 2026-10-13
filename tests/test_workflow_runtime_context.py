from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.workflows.stages.handoffs import LandUseToSupplyDemandHandoff
from pilates.workflows.stages.postprocessing import run_postprocessing_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.supply_demand_activity import (
    ActivityDemandPhaseOutputs,
)
from pilates.workflows.stages.supply_demand_beam import (
    TrafficAssignmentPhaseOutputs,
)


def test_workflow_runtime_context_builds_surface_when_missing(monkeypatch) -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace()
    workspace = SimpleNamespace()
    built_surface = SimpleNamespace(profile=SimpleNamespace())
    captured = {}

    def _fake_build_enabled_workflow_surface(received_settings, *, state):
        captured["settings"] = received_settings
        captured["state"] = state
        return built_surface

    monkeypatch.setattr(
        "pilates.workflows.surface.build_enabled_workflow_surface",
        _fake_build_enabled_workflow_surface,
    )

    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert context.surface is built_surface
    assert captured == {"settings": settings, "state": state}


def test_workflow_runtime_context_reuses_explicit_surface() -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace()
    workspace = SimpleNamespace()
    surface = SimpleNamespace(profile=SimpleNamespace())

    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )

    assert context.surface is surface


def test_run_supply_demand_stage_passes_runtime_context_to_phase_helpers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeStage:
        supply_demand_loop = "supply_demand_loop"
        activity_demand = "activity_demand"
        traffic_assignment = "traffic_assignment"
        land_use = "land_use"

    class _FakeState:
        Stage = _FakeStage
        iteration = 0
        current_major_stage = _FakeStage.supply_demand_loop
        year = 2018
        forecast_year = 2018
        is_restart_run = False

        def __iter__(self):
            return iter([2018])

        def should_run(self, major_stage, iteration=None, sub_stage=None):
            if major_stage != self.Stage.supply_demand_loop:
                return False
            if sub_stage is None:
                return True
            return sub_stage in {
                self.Stage.activity_demand,
                self.Stage.traffic_assignment,
            }

        def is_enabled(self, _stage):
            return False

        def complete_step(self, *args, **kwargs):
            return None

    settings = SimpleNamespace(
        run=SimpleNamespace(
            supply_demand_iters=1,
            models=SimpleNamespace(activity_demand=object()),
        )
    )
    state = _FakeState()
    workspace = SimpleNamespace(full_path=str(tmp_path))
    surface = SimpleNamespace(profile=SimpleNamespace())
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    seen = {}

    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand._run_activity_demand_phase",
        lambda **kwargs: (
            seen.setdefault("activity_context", kwargs["context"]),
            ActivityDemandPhaseOutputs(activity_demand_outputs={"plans": "ok"}),
        )[1],
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand._run_traffic_assignment_phase",
        lambda **kwargs: (
            seen.setdefault("beam_context", kwargs["context"]),
            TrafficAssignmentPhaseOutputs(),
        )[1],
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand.flush_archive_queue",
        lambda *args, **kwargs: None,
    )
    run_supply_demand_stage(
        scenario=SimpleNamespace(),
        coupler=SimpleNamespace(),
        year=2018,
        handoff=LandUseToSupplyDemandHandoff(),
        context=context,
    )

    assert seen["activity_context"] is context
    assert seen["beam_context"] is context


def test_supply_demand_snapshots_after_activitysim_before_beam(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A completed ActivitySim handoff is durable before BEAM begins."""

    class _FakeStage:
        supply_demand_loop = "supply_demand_loop"
        activity_demand = "activity_demand"
        traffic_assignment = "traffic_assignment"

    class _FakeState:
        Stage = _FakeStage
        iteration = 0
        current_major_stage = _FakeStage.supply_demand_loop
        year = 2018
        forecast_year = 2018
        is_restart_run = False

        def should_run(self, major_stage, iteration=None, sub_stage=None):
            return major_stage == self.Stage.supply_demand_loop

        def complete_step(self, *_args, **_kwargs):
            return None

    context = WorkflowRuntimeContext.from_parts(
        settings=SimpleNamespace(
            run=SimpleNamespace(
                supply_demand_iters=1,
                models=SimpleNamespace(activity_demand="activitysim"),
            )
        ),
        state=_FakeState(),
        workspace=SimpleNamespace(full_path=str(tmp_path)),
        surface=SimpleNamespace(profile=SimpleNamespace()),
    )
    events: list[str] = []
    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand._run_activity_demand_phase",
        lambda **_kwargs: events.append("activitysim"),
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand._run_traffic_assignment_phase",
        lambda **_kwargs: events.append("beam"),
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand.flush_archive_queue",
        lambda **_kwargs: events.append("flush"),
    )

    run_supply_demand_stage(
        scenario=SimpleNamespace(),
        coupler=SimpleNamespace(),
        year=2018,
        handoff=LandUseToSupplyDemandHandoff(),
        on_activity_demand_boundary=lambda _iteration: events.append("snapshot"),
        context=context,
    )

    assert events[:4] == ["activitysim", "flush", "snapshot", "beam"]


def test_supply_demand_fails_closed_for_uncommitted_restart_without_handoff(
    tmp_path: Path,
) -> None:
    class _FakeStage:
        supply_demand_loop = "supply_demand_loop"
        activity_demand = "activity_demand"
        traffic_assignment = "traffic_assignment"

    class _FakeState:
        Stage = _FakeStage
        iteration = 0
        current_major_stage = _FakeStage.supply_demand_loop
        current_sub_stage = _FakeStage.traffic_assignment
        year = 2018
        forecast_year = 2018
        is_restart_run = False

        def should_run(self, major_stage, iteration=None, sub_stage=None):
            if major_stage != self.Stage.supply_demand_loop:
                return False
            if iteration != self.iteration:
                return False
            if sub_stage is None:
                return True
            return (self.Stage.activity_demand, self.Stage.traffic_assignment).index(
                self.current_sub_stage
            ) <= (self.Stage.activity_demand, self.Stage.traffic_assignment).index(
                sub_stage
            )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            supply_demand_iters=1,
            models=SimpleNamespace(activity_demand="activitysim"),
        )
    )
    state = _FakeState()
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=SimpleNamespace(full_path=str(tmp_path)),
        surface=SimpleNamespace(profile=SimpleNamespace()),
    )
    with pytest.raises(
        RuntimeError,
        match="ActivitySim is skipped outside the committed BEAM checkpoint",
    ):
        run_supply_demand_stage(
            scenario=SimpleNamespace(),
            coupler=SimpleNamespace(),
            year=2018,
            handoff=LandUseToSupplyDemandHandoff(),
            context=context,
        )


def test_run_postprocessing_stage_uses_runtime_context(
    monkeypatch, tmp_path: Path
) -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace(iteration=0)
    workspace = SimpleNamespace(full_path=str(tmp_path))
    surface = SimpleNamespace(profile=SimpleNamespace())
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    captured = {}

    monkeypatch.setattr(
        "pilates.workflows.stages.postprocessing.execute_step",
        lambda **kwargs: (
            captured.update(kwargs),
            (SimpleNamespace(), SimpleNamespace()),
        )[1],
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.postprocessing.flush_archive_queue",
        lambda *args, **kwargs: None,
    )

    run_postprocessing_stage(
        scenario="scenario",
        coupler="coupler",
        year=2018,
        context=context,
    )

    assert captured["settings"] is settings
    assert captured["state"] is state
    assert captured["workspace"] is workspace
