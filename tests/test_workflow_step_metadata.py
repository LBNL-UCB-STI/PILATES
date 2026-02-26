from __future__ import annotations

from types import SimpleNamespace

import pytest
from consist import define_step
from consist.types import CacheOptions, OutputPolicyOptions

from pilates.workflows.artifact_keys import ASIM_HOUSEHOLDS_IN
from pilates.workflows.artifact_keys import ASIM_LAND_USE_IN, ASIM_PERSONS_IN
from pilates.workflows.artifact_keys import USIM_DATASTORE_H5
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.orchestration import StepRef, WorkflowStage
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_preprocess_step,
    make_urbansim_run_step,
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
    assert preprocess_meta.outputs == [
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
    ]
    assert compile_meta.outputs == ["zarr_skims"]
    assert preprocess_meta.name_template == "{func_name}__y{year}__i{iteration}__phase_{phase}"
    assert callable(preprocess_meta.adapter)
    assert callable(preprocess_meta.config)
    assert callable(preprocess_meta.identity_inputs)
    assert callable(preprocess_meta.facet)


def test_urbansim_run_declares_strict_output_contract():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()

    run_step = make_urbansim_run_step(coupler=coupler, outputs_holder=holder)
    assert hasattr(run_step, "__consist_step__")
    meta = run_step.__consist_step__
    assert meta.model == "urbansim_run"
    assert meta.outputs == [USIM_DATASTORE_H5]


def test_workflow_stage_uses_decorator_metadata_without_legacy_consist_kwargs():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    @define_step(model="dummy_step")
    def _decorated_step(settings, state, workspace):
        return None

    spec = StepRef(name="dummy_step", step_func=_decorated_step)

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
    assert "identity_inputs" not in call
    assert "adapter" not in call
    assert call["execution_options"].runtime_kwargs == {
        "settings": settings,
        "state": state,
        "workspace": workspace,
    }
    assert call["execution_options"].load_inputs is None


def test_workflow_stage_uses_top_level_runtime_kwargs_with_load_inputs_option():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    @define_step(model="dummy_step")
    def _decorated_step(settings, state, workspace):
        return None

    spec = StepRef(name="dummy_step", step_func=_decorated_step, load_inputs=True)

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
    assert call["execution_options"].runtime_kwargs == {
        "settings": settings,
        "state": state,
        "workspace": workspace,
    }
    assert call["execution_options"].load_inputs is True


def test_workflow_stage_propagates_consist_code_identity_override():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace(
        run=SimpleNamespace(consist_code_identity="callable_module")
    )
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    @define_step(model="dummy_step")
    def _decorated_step(settings, state, workspace):
        return None

    spec = StepRef(name="dummy_step", step_func=_decorated_step)

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
    assert call["cache_options"] == CacheOptions(code_identity="callable_module")


def test_workflow_stage_infers_strict_output_enforcement_from_step_outputs_metadata():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    @define_step(model="dummy_step", outputs=["artifact_a", "artifact_b"])
    def _decorated_step(settings, state, workspace):
        return None

    spec = StepRef(name="dummy_step", step_func=_decorated_step)
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
    assert call["outputs"] == ["artifact_a", "artifact_b"]
    assert call["output_policy"] == OutputPolicyOptions(
        output_missing="error",
        output_mismatch="error",
    )


def test_workflow_stage_infers_strict_output_enforcement_from_step_output_class():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()
    outputs_holder.urbansim_preprocess = SimpleNamespace()

    @define_step(model="urbansim_run")
    def _decorated_step(settings, state, workspace):
        return None

    spec = StepRef(name="urbansim_run", step_func=_decorated_step)
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
    assert call["outputs"] == [USIM_DATASTORE_H5]
    assert call["output_policy"] == OutputPolicyOptions(
        output_missing="error",
        output_mismatch="error",
    )


def test_workflow_stage_explicit_output_enforcement_overrides_defaults():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    @define_step(model="dummy_step", outputs=["artifact_a"])
    def _decorated_step(settings, state, workspace):
        return None

    spec = StepRef(
        name="dummy_step",
        step_func=_decorated_step,
        required_outputs=["artifact_override"],
        required_outputs_rationale="Temporary compatibility while migrating metadata.",
        output_missing="warn",
        output_mismatch="warn",
    )
    stage = WorkflowStage(name="unit_stage", stage_type="unit", steps=[spec])
    with pytest.warns(DeprecationWarning, match="StepRef.required_outputs is deprecated"):
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
    assert call["outputs"] == ["artifact_override"]
    assert call["output_policy"] == OutputPolicyOptions(
        output_missing="warn",
        output_mismatch="warn",
    )


def test_workflow_stage_required_outputs_override_requires_rationale():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    @define_step(model="dummy_step", outputs=["artifact_a"])
    def _decorated_step(settings, state, workspace):
        return None

    spec = StepRef(
        name="dummy_step",
        step_func=_decorated_step,
        required_outputs=["artifact_override"],
    )
    stage = WorkflowStage(name="unit_stage", stage_type="unit", steps=[spec])

    with pytest.raises(
        RuntimeError,
        match="required_outputs_rationale",
    ):
        stage.run(
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            name_suffix="unit",
        )


def test_tracked_step_uses_canonical_outputs_instead_of_metadata_outputs():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    outputs_holder.urbansim_preprocess = SimpleNamespace()
    coupler = _DummyCoupler()

    @define_step(model="urbansim_run", outputs=["metadata_override"])
    def _decorated_step(settings, state, workspace):
        return None

    spec = StepRef(name="urbansim_run", step_func=_decorated_step)
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
    assert call["outputs"] == [USIM_DATASTORE_H5]


def test_workflow_stage_requires_decorated_steps():
    scenario = _FakeScenario()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    outputs_holder = StepOutputsHolder()
    coupler = _DummyCoupler()

    def _legacy_step(settings, state, workspace):
        return None

    spec = StepRef(name="legacy_step", step_func=_legacy_step)

    stage = WorkflowStage(name="unit_stage", stage_type="unit", steps=[spec])
    try:
        stage.run(
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            name_suffix="unit",
        )
    except TypeError as exc:
        assert "must be decorated" in str(exc)
    else:
        raise AssertionError("Expected undecorated steps to raise TypeError")


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


def test_define_step_identity_inputs_metadata_drives_cache_identity_end_to_end(
    tmp_path,
):
    pytest.importorskip("consist")
    from consist import Tracker
    from consist.integrations.activitysim import ActivitySimConfigAdapter

    from pilates.utils import consist_runtime as cr

    tracker = Tracker(
        run_dir=tmp_path / "consist_runs",
        db_path=str(tmp_path / "consist_test.duckdb"),
        mounts={"workspace": str(tmp_path)},
    )

    config_root = tmp_path / "activitysim" / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "settings.yaml").write_text("models: []\n")

    identity_marker = tmp_path / "identity_marker.txt"
    identity_marker.write_text("v1")
    output_path = tmp_path / "result.txt"
    calls = {"count": 0}

    @define_step(
        model="activitysim_identity_metadata_step",
        adapter=lambda ctx: ActivitySimConfigAdapter(root_dirs=[config_root]),
        identity_inputs=lambda ctx: [("identity_marker", identity_marker)],
        output_paths={"result": str(output_path)},
    )
    def _step():
        calls["count"] += 1
        output_path.write_text(f"run-{calls['count']}")

    with cr.scenario("identity-metadata-step", tracker=tracker) as scenario:
        first = scenario.run(fn=_step, year=2018, iteration=0, phase="run")
        second = scenario.run(fn=_step, year=2018, iteration=0, phase="run")

        identity_marker.write_text("v2")
        third = scenario.run(fn=_step, year=2018, iteration=0, phase="run")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert third.cache_hit is False
    assert calls["count"] == 2
