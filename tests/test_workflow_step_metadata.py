from __future__ import annotations

from types import SimpleNamespace

import pytest
from consist import define_step
from consist.types import CacheOptions, OutputPolicyOptions

from pilates.runtime import launcher as run_module
from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
)
from pilates.workflows.artifact_keys import ASIM_HOUSEHOLDS_IN
from pilates.workflows.artifact_keys import ASIM_LAND_USE_IN, ASIM_PERSONS_IN
from pilates.workflows.artifact_keys import (
    BEAM_FULL_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
)
from pilates.workflows.artifact_keys import USIM_DATASTORE_H5
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.orchestration import StepRef, WorkflowStage
from pilates.workflows.outputs_base import declared_outputs_for_step_outputs_class
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
    make_beam_full_skim_step,
    make_activitysim_preprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_postprocess_step,
    make_urbansim_run_step,
)
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
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


def test_urbansim_and_atlas_step_factories_attach_consist_metadata():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()
    steps = {
        "urbansim_preprocess": make_urbansim_preprocess_step(
            coupler=coupler,
            outputs_holder=holder,
        ),
        "urbansim_run": make_urbansim_run_step(
            coupler=coupler,
            outputs_holder=holder,
        ),
        "urbansim_postprocess": make_urbansim_postprocess_step(
            coupler=coupler,
            outputs_holder=holder,
        ),
        "atlas_preprocess": make_atlas_preprocess_step(
            coupler=coupler,
            outputs_holder=holder,
        ),
        "atlas_run": make_atlas_run_step(
            coupler=coupler,
            outputs_holder=holder,
        ),
        "atlas_postprocess": make_atlas_postprocess_step(
            coupler=coupler,
            outputs_holder=holder,
        ),
    }
    expected_outputs = {
        "urbansim_preprocess": list(
            declared_outputs_for_step_outputs_class(UrbanSimPreprocessOutputs)
        )
        or None,
        "urbansim_run": list(
            declared_outputs_for_step_outputs_class(UrbanSimRunOutputs)
        )
        or None,
        "urbansim_postprocess": list(
            declared_outputs_for_step_outputs_class(UrbanSimPostprocessOutputs)
        )
        or None,
        "atlas_preprocess": list(
            declared_outputs_for_step_outputs_class(AtlasPreprocessOutputs)
        )
        or None,
        "atlas_run": list(declared_outputs_for_step_outputs_class(AtlasRunOutputs))
        or None,
        "atlas_postprocess": list(
            declared_outputs_for_step_outputs_class(AtlasPostprocessOutputs)
        )
        or None,
    }

    for step_name, step in steps.items():
        assert hasattr(step, "__consist_step__")
        meta = step.__consist_step__
        assert meta.model == step_name
        assert meta.outputs == expected_outputs[step_name]


def test_atlas_step_metadata_config_includes_parent_forecast_year():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()
    step = make_atlas_postprocess_step(coupler=coupler, outputs_holder=holder)
    meta = step.__consist_step__

    settings = SimpleNamespace(
        run=SimpleNamespace(),
        activitysim=None,
        beam=None,
        urbansim=None,
        postprocessing=None,
        atlas=SimpleNamespace(model_dump=lambda: {"max_retries": 1}),
    )
    state = SimpleNamespace(year=2023, forecast_year=2023, main_forecast_year=2029)
    workspace = SimpleNamespace(full_path="/tmp/workspace")

    class _Ctx:
        def __init__(self, runtime):
            self._runtime = runtime

        def get_runtime(self, name, default=None):
            return self._runtime.get(name, default)

    ctx = _Ctx({"settings": settings, "state": state, "workspace": workspace})

    config = meta.config(ctx)
    facet = meta.facet(ctx)

    assert config["atlas_subyear"] == 2023
    assert config["main_forecast_year"] == 2029
    assert facet["atlas_subyear"] == 2023
    assert facet["main_forecast_year"] == 2029


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


def test_build_coupler_schema_declares_urbansim_geoid_to_zone_output():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()

    preprocess_step = make_urbansim_preprocess_step(
        coupler=coupler,
        outputs_holder=holder,
    )

    schema = build_coupler_schema([preprocess_step], settings=SimpleNamespace())

    assert "geoid_to_zone" in schema


def test_build_coupler_schema_declares_optional_beam_staged_outputs():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()

    preprocess_step = make_beam_preprocess_step(
        coupler=coupler,
        outputs_holder=holder,
    )

    schema = build_coupler_schema([preprocess_step], settings=SimpleNamespace())

    assert "vehicles_beam_in" in schema


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


class _FakeEpochTaggingScenario:
    def __init__(self, run_ids):
        self.calls = []
        self._run_ids = list(run_ids)

    def run(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        run_id = self._run_ids[index] if index < len(self._run_ids) else None
        run_obj = SimpleNamespace(id=run_id) if run_id is not None else None
        return SimpleNamespace(cache_hit=False, run=run_obj)


def _step_with_model(model_name: str):
    def _step():
        return None

    _step.__consist_step__ = SimpleNamespace(model=model_name)
    return _step


def test_epoch_tagging_proxy_injects_required_facet_and_tag_keys():
    scenario = _FakeEpochTaggingScenario(run_ids=["asim-r1"])
    proxy = run_module._EpochTaggingScenarioProxy(
        scenario,
        scenario_id="scenario-alpha",
        seed=777,
    )

    proxy.run(
        fn=_step_with_model("activitysim_run"),
        name="activitysim_run",
        year=2035,
        iteration=2,
        tags=["existing"],
        facet={"existing_key": "existing_value"},
    )

    call = scenario.calls[0]
    assert call["model"] == "activitysim_run"
    assert "existing" in call["tags"]
    assert "scenario_id:scenario-alpha" in call["tags"]
    assert "seed:777" in call["tags"]
    assert "model:activitysim_run" in call["tags"]
    assert "year:2035" in call["tags"]
    assert "iteration:2" in call["tags"]
    assert call["facet"]["existing_key"] == "existing_value"
    assert call["facet"]["scenario_id"] == "scenario-alpha"
    assert call["facet"]["seed"] == 777
    assert call["facet"]["model"] == "activitysim_run"
    assert call["facet"]["year"] == 2035
    assert call["facet"]["iteration"] == 2


def test_epoch_tagging_proxy_sets_beam_parent_to_same_epoch_activitysim():
    scenario = _FakeEpochTaggingScenario(run_ids=["asim-2030-i1", "beam-2030-i1"])
    proxy = run_module._EpochTaggingScenarioProxy(
        scenario,
        scenario_id="scenario-alpha",
        seed=777,
    )

    proxy.run(model="activitysim_run", year=2030, iteration=1)
    proxy.run(model="beam_run", year=2030, iteration=1)

    beam_call = scenario.calls[1]
    assert beam_call["parent_run_id"] == "asim-2030-i1"


def test_epoch_tagging_proxy_sets_activitysim_parent_to_previous_beam_iteration():
    scenario = _FakeEpochTaggingScenario(run_ids=["beam-2030-i0", "asim-2030-i1"])
    proxy = run_module._EpochTaggingScenarioProxy(
        scenario,
        scenario_id="scenario-alpha",
        seed=777,
    )

    proxy.run(model="beam_run", year=2030, iteration=0)
    proxy.run(model="activitysim_run", year=2030, iteration=1)

    activitysim_call = scenario.calls[1]
    assert activitysim_call["parent_run_id"] == "beam-2030-i0"


def test_epoch_tagging_proxy_does_not_replace_beam_parent_with_full_skim_sidecar():
    scenario = _FakeEpochTaggingScenario(
        run_ids=["beam-2030-i0", "beam-fullskim-2030-i0", "asim-2030-i1"]
    )
    proxy = run_module._EpochTaggingScenarioProxy(
        scenario,
        scenario_id="scenario-alpha",
        seed=777,
    )

    proxy.run(model="beam_run", year=2030, iteration=0)
    proxy.run(model="beam_full_skim", year=2030, iteration=0)
    proxy.run(model="activitysim_run", year=2030, iteration=1)

    activitysim_call = scenario.calls[2]
    assert activitysim_call["parent_run_id"] == "beam-2030-i0"


@pytest.mark.parametrize(
    ("model", "year", "iteration"),
    [
        ("beam_run", 2030, 0),
        ("activitysim_run", 2030, 1),
    ],
)
def test_epoch_tagging_proxy_missing_parent_does_not_raise(model, year, iteration):
    scenario = _FakeEpochTaggingScenario(run_ids=["run-1"])
    proxy = run_module._EpochTaggingScenarioProxy(
        scenario,
        scenario_id="scenario-alpha",
        seed=777,
    )

    proxy.run(model=model, year=year, iteration=iteration)

    call = scenario.calls[0]
    assert "parent_run_id" not in call


def test_beam_postprocess_step_metadata_tracks_current_canonical_outputs():
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()

    step = make_beam_postprocess_step(coupler=coupler, outputs_holder=holder)
    meta = step.__consist_step__

    assert meta.outputs == ["zarr_skims"]


@pytest.mark.parametrize(
    ("factory", "expected_outputs"),
    [
        (make_beam_preprocess_step, [BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN]),
        (make_beam_run_step, [LINKSTATS, BEAM_PLANS_OUT]),
        (make_beam_full_skim_step, [BEAM_FULL_SKIMS]),
    ],
)
def test_other_beam_step_metadata_keeps_model_specific_output_contracts(
    factory,
    expected_outputs,
):
    coupler = _DummyCoupler()
    holder = StepOutputsHolder()

    step = factory(coupler=coupler, outputs_holder=holder)
    meta = step.__consist_step__

    assert meta.outputs == expected_outputs
