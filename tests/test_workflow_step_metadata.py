from __future__ import annotations

from types import SimpleNamespace

from consist import define_step

from pilates.workflows.artifact_keys import ASIM_HOUSEHOLDS_IN
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.orchestration import WorkflowStage, WorkflowStepSpec
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_preprocess_step,
)


class _DummyCoupler:
    def __init__(self) -> None:
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value) -> None:
        self._data[key] = value


class _FakeScenario:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(cache_hit=False)


def test_make_step_factories_attach_consist_metadata():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()

    preprocess_step = make_activitysim_preprocess_step(
        coupler=coupler,
        outputs_holder=holder,
    )
    compile_step = make_activitysim_compile_step(
        coupler=coupler,
        outputs_holder=holder,
    )

    assert hasattr(preprocess_step, "__consist_step__")
    assert hasattr(compile_step, "__consist_step__")

    preprocess_meta = preprocess_step.__consist_step__
    compile_meta = compile_step.__consist_step__

    assert preprocess_meta.model == "activitysim_preprocess"
    assert compile_meta.model == "activitysim_compile"
    assert compile_meta.outputs == ["zarr_skims"]
    assert preprocess_meta.name_template == "{func_name}__y{year}__i{iteration}__phase_{phase}"
    assert callable(preprocess_meta.config)
    assert callable(preprocess_meta.config_plan)
    assert callable(preprocess_meta.facet)


def test_workflow_stage_uses_decorator_metadata_without_legacy_consist_kwargs(
    monkeypatch,
):
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    @define_step(model="dummy_step")
    def _decorated_step(settings, state, workspace):
        return None

    spec = WorkflowStepSpec(name="dummy_step", step_func=_decorated_step)

    def _unexpected(*args, **kwargs):
        raise AssertionError("build_step_consist_kwargs should not be called")

    monkeypatch.setattr(
        "pilates.workflows.orchestration.build_step_consist_kwargs",
        _unexpected,
    )

    stage = WorkflowStage(name="unit_stage", stage_type="unit", steps=[spec])
    stage.run(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix="unit",
    )

    call = scenario.calls[0]
    assert "config" not in call
    assert "facet" not in call
    assert "hash_inputs" not in call


def test_workflow_stage_falls_back_to_legacy_consist_kwargs_for_undecorated_steps(
    monkeypatch,
):
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    def _legacy_step(settings, state, workspace):
        return None

    spec = WorkflowStepSpec(name="legacy_step", step_func=_legacy_step)

    monkeypatch.setattr(
        "pilates.workflows.orchestration.build_step_consist_kwargs",
        lambda step_name, settings, workspace_path: {
            "config": {"step": step_name, "workspace": workspace_path}
        },
    )

    stage = WorkflowStage(name="unit_stage", stage_type="unit", steps=[spec])
    stage.run(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix="unit",
    )

    call = scenario.calls[0]
    assert "config" in call
    assert call["config"]["step"] == "legacy_step"


def test_build_coupler_schema_collects_step_metadata_and_extras():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()

    preprocess_step = make_activitysim_preprocess_step(
        coupler=coupler,
        outputs_holder=holder,
    )

    schema = build_coupler_schema([preprocess_step], settings=SimpleNamespace())

    assert ASIM_HOUSEHOLDS_IN in schema
    assert "urbansim/usim_datastore_h5" in schema
