from __future__ import annotations

from types import SimpleNamespace

import pytest
from consist import define_step

from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    ASIM_SHARROW_CACHE_DIR,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_H5_UPDATED,
)
from pilates.workflows.binding import (
    ArtifactBindingRule,
    BindingPlan,
    activitysim_datastore_selection_rules,
    binding_spec_for_step_name,
    build_binding_plan,
    build_key_only_binding_plan,
    urbansim_datastore_selection_rules,
)
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.steps import StepOutputsHolder


class _FakeScenario:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(cache_hit=False, run=SimpleNamespace(id="step-run"))


class _CouplerStub:
    def __init__(self, values):
        self._values = dict(values)

    def get(self, key, default=None):
        return self._values.get(key, default)


def test_binding_plan_converts_to_consist_binding_result():
    plan = BindingPlan(
        step_name="activitysim_compile",
        inputs={"asim_omx_skims": "/tmp/skims.omx"},
        input_keys=["asim_omx_skims"],
        optional_input_keys=["maybe_optional"],
        missing_required=[],
        metadata={"source": "unit-test"},
    )

    binding = plan.to_binding_result()

    assert binding.inputs == {"asim_omx_skims": "/tmp/skims.omx"}
    assert binding.input_keys == ["asim_omx_skims"]
    assert binding.optional_input_keys == ["maybe_optional"]
    assert binding.metadata == {"source": "unit-test"}
    assert plan.to_scenario_run_kwargs()["binding"] == binding


def test_binding_spec_for_step_name_derives_from_catalog():
    spec = binding_spec_for_step_name("activitysim_compile")

    assert spec is not None
    assert spec.step_name == "activitysim_compile"
    assert spec.semantic_input_keys() == (ASIM_OMX_SKIMS,)
    assert spec.semantic_output_keys() == ("zarr_skims",)


def test_build_binding_plan_applies_beam_preprocess_preferred_key_overrides():
    plan = build_binding_plan(
        step_name="beam_preprocess",
        explicit_inputs={
            "beam_plans_asim_out": "/tmp/plans.parquet",
            "households_asim_out": "/tmp/households.parquet",
            "persons_asim_out": "/tmp/persons.parquet",
            BEAM_CONFIG_FILE: "/tmp/beam.conf",
        },
        coupler=_CouplerStub({LINKSTATS: "/tmp/linkstats.csv.gz"}),
    )

    assert plan.inputs[BEAM_PLANS_IN] == "/tmp/plans.parquet"
    assert plan.inputs[BEAM_HOUSEHOLDS_IN] == "/tmp/households.parquet"
    assert plan.inputs[BEAM_PERSONS_IN] == "/tmp/persons.parquet"
    assert plan.inputs[BEAM_CONFIG_FILE] == "/tmp/beam.conf"
    assert plan.input_keys == []
    assert plan.optional_input_keys == [LINKSTATS]
    assert plan.source_by_key[LINKSTATS_WARMSTART] == "coupler"
    assert not plan.missing_required


def test_build_binding_plan_uses_activitysim_preprocess_fallback_provider(monkeypatch):
    from pilates.workflows import binding as binding_module

    monkeypatch.setitem(
        binding_module._FALLBACK_PROVIDERS,
        "urbansim_inputs_for_year",
        lambda **_: {USIM_DATASTORE_BASE_H5: "/tmp/base.h5"},
    )

    plan = build_binding_plan(
        step_name="activitysim_preprocess",
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2030),
        workspace=SimpleNamespace(),
        year=2030,
    )

    assert plan.inputs[USIM_H5_UPDATED] == "/tmp/base.h5"
    assert plan.source_by_key[USIM_H5_UPDATED] == "fallback"
    assert not plan.missing_required


def test_build_binding_plan_uses_activitysim_postprocess_base_datastore_provider(
    monkeypatch,
):
    from pilates.workflows import binding as binding_module

    monkeypatch.setitem(
        binding_module._FALLBACK_PROVIDERS,
        "activitysim_input_datastore",
        lambda **_: {USIM_DATASTORE_BASE_H5: "/tmp/input.h5"},
    )

    plan = build_binding_plan(
        step_name="activitysim_postprocess",
        coupler=_CouplerStub({"asim_land_use_in": "workspace://land_use.csv"}),
        required_keys=["asim_land_use_in"],
        optional_keys=[USIM_DATASTORE_BASE_H5],
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2030),
        workspace=SimpleNamespace(),
        year=2030,
    )

    assert plan.input_keys == ["asim_land_use_in"]
    assert plan.inputs[USIM_DATASTORE_BASE_H5] == "/tmp/input.h5"
    assert plan.source_by_key[USIM_DATASTORE_BASE_H5] == "fallback"
    assert not plan.missing_required


def test_build_binding_plan_supports_ad_hoc_preferred_key_rules():
    plan = build_binding_plan(
        step_name="activitysim_input_selection",
        explicit_inputs={USIM_DATASTORE_BASE_H5: "/tmp/base.h5"},
        artifact_rules=(
            ArtifactBindingRule(
                semantic_key=USIM_DATASTORE_CURRENT_H5,
                required=True,
                preferred_keys=(
                    USIM_DATASTORE_CURRENT_H5,
                    USIM_DATASTORE_BASE_H5,
                ),
            ),
        ),
        required_keys=[USIM_DATASTORE_CURRENT_H5],
    )

    assert plan.inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/base.h5"
    assert (
        plan.metadata["selected_key_by_semantic_key"][USIM_DATASTORE_CURRENT_H5]
        == USIM_DATASTORE_BASE_H5
    )
    assert not plan.missing_required


def test_activitysim_datastore_selection_rules_centralize_current_base_fallback():
    plan = build_binding_plan(
        step_name="activitysim_input_selection",
        explicit_inputs={USIM_DATASTORE_BASE_H5: "/tmp/base.h5"},
        artifact_rules=activitysim_datastore_selection_rules(),
        required_keys=[USIM_DATASTORE_CURRENT_H5],
    )

    assert plan.inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/base.h5"
    assert (
        plan.metadata["selected_key_by_semantic_key"][USIM_DATASTORE_CURRENT_H5]
        == USIM_DATASTORE_BASE_H5
    )
    assert not plan.missing_required


def test_build_binding_plan_centralizes_urbansim_input_selection_fallbacks(monkeypatch):
    from pilates.workflows import binding as binding_module

    monkeypatch.setitem(
        binding_module._FALLBACK_PROVIDERS,
        "urbansim_inputs_for_year",
        lambda **_: {USIM_DATASTORE_BASE_H5: "/tmp/base.h5"},
    )

    plan = build_binding_plan(
        step_name="urbansim_input_selection",
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2030),
        workspace=SimpleNamespace(),
        year=2030,
        artifact_rules=urbansim_datastore_selection_rules(),
        required_keys=[USIM_DATASTORE_BASE_H5, USIM_DATASTORE_CURRENT_H5],
    )

    assert plan.inputs[USIM_DATASTORE_BASE_H5] == "/tmp/base.h5"
    assert plan.inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/base.h5"
    assert (
        plan.metadata["selected_key_by_semantic_key"][USIM_DATASTORE_BASE_H5]
        == USIM_DATASTORE_BASE_H5
    )
    assert (
        plan.metadata["selected_key_by_semantic_key"][USIM_DATASTORE_CURRENT_H5]
        == USIM_DATASTORE_BASE_H5
    )
    assert not plan.missing_required


def test_build_binding_plan_centralizes_atlas_preprocess_usim_precedence():
    plan = build_binding_plan(
        step_name="atlas_preprocess",
        coupler=_CouplerStub({USIM_H5_UPDATED: "/tmp/updated.h5"}),
        fallback_inputs={
            USIM_DATASTORE_CURRENT_H5: "/tmp/fallback.h5",
            USIM_DATASTORE_BASE_H5: "/tmp/fallback.h5",
        },
        required_keys=[USIM_DATASTORE_CURRENT_H5, USIM_DATASTORE_BASE_H5],
    )

    assert plan.input_keys == [USIM_H5_UPDATED]
    assert plan.optional_input_keys == []
    assert plan.source_by_key[USIM_DATASTORE_CURRENT_H5] == "coupler"
    assert plan.source_by_key[USIM_DATASTORE_BASE_H5] == "coupler"
    assert plan.coupler_key_by_key[USIM_DATASTORE_CURRENT_H5] == USIM_H5_UPDATED
    assert plan.coupler_key_by_key[USIM_DATASTORE_BASE_H5] == USIM_H5_UPDATED
    assert not plan.missing_required


def test_build_binding_plan_preserves_atlas_linear_stage_inputs():
    plan = build_binding_plan(
        step_name="atlas_run",
        explicit_inputs={
            USIM_DATASTORE_CURRENT_H5: "/tmp/current.h5",
            USIM_DATASTORE_BASE_H5: "/tmp/base.h5",
            "psid_names": "/tmp/psid_names.Rdat",
        },
        required_keys=[USIM_DATASTORE_CURRENT_H5, USIM_DATASTORE_BASE_H5],
        optional_keys=["psid_names"],
    )

    assert plan.inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/current.h5"
    assert plan.inputs[USIM_DATASTORE_BASE_H5] == "/tmp/base.h5"
    assert plan.inputs["psid_names"] == "/tmp/psid_names.Rdat"
    assert plan.source_by_key[USIM_DATASTORE_CURRENT_H5] == "explicit"
    assert plan.source_by_key["psid_names"] == "explicit"
    assert not plan.missing_required


def test_build_binding_plan_uses_caller_scoped_fallback_inputs_for_explicit_key_sets():
    plan = build_binding_plan(
        step_name="atlas_run",
        explicit_inputs={USIM_DATASTORE_CURRENT_H5: "/tmp/current.h5"},
        fallback_inputs={
            USIM_DATASTORE_BASE_H5: "/tmp/base.h5",
            "psid_names": "/tmp/psid_names.Rdat",
        },
        required_keys=[USIM_DATASTORE_CURRENT_H5],
        optional_keys=[USIM_DATASTORE_BASE_H5, "psid_names"],
    )

    assert plan.inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/current.h5"
    assert plan.inputs[USIM_DATASTORE_BASE_H5] == "/tmp/base.h5"
    assert plan.inputs["psid_names"] == "/tmp/psid_names.Rdat"
    assert plan.source_by_key[USIM_DATASTORE_CURRENT_H5] == "explicit"
    assert plan.source_by_key[USIM_DATASTORE_BASE_H5] == "fallback"
    assert plan.source_by_key["psid_names"] == "fallback"
    assert not plan.missing_required


def test_build_key_only_binding_plan_preserves_optional_key_split():
    coupler = _CouplerStub(
        {
            "asim_land_use_in": "workspace://land_use.csv",
            "zarr_skims": "workspace://skims.zarr",
            ASIM_SHARROW_CACHE_DIR: "workspace://numba",
        }
    )
    plan = build_key_only_binding_plan(
        step_name="activitysim_run",
        input_keys=[
            "asim_land_use_in",
            "zarr_skims",
            ASIM_SHARROW_CACHE_DIR,
        ],
        optional_input_keys=[ASIM_SHARROW_CACHE_DIR],
        coupler=coupler,
    )

    assert plan.inputs == {}
    assert plan.input_keys == [
        "asim_land_use_in",
        "zarr_skims",
    ]
    assert plan.optional_input_keys == [ASIM_SHARROW_CACHE_DIR]
    assert plan.source_by_key[ASIM_SHARROW_CACHE_DIR] == "coupler"
    assert not plan.missing_required


def test_run_workflow_passes_binding_result_without_manual_input_split():
    scenario = _FakeScenario()
    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_preprocess = SimpleNamespace()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    coupler = SimpleNamespace(
        get=lambda *args, **kwargs: None,
        set=lambda *args, **kwargs: None,
        update=lambda *args, **kwargs: None,
    )

    @define_step(model="activitysim_compile")
    def _dummy_step(settings, state, workspace, **kwargs):
        return None

    step = StepRef(
        name="activitysim_compile",
        step_func=_dummy_step,
        binding=BindingPlan(
            step_name="activitysim_compile",
            inputs={ASIM_OMX_SKIMS: "/tmp/skims.omx"},
        ),
    )

    run_workflow(
        stage_name="activity_demand",
        steps=[step],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix="unit",
    )

    assert len(scenario.calls) == 1
    call = scenario.calls[0]
    assert "binding" in call
    assert call["binding"].inputs == {ASIM_OMX_SKIMS: "/tmp/skims.omx"}
    assert "inputs" not in call
    assert "input_keys" not in call
