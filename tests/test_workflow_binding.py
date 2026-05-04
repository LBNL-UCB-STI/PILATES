from __future__ import annotations

from types import SimpleNamespace

import consist
import pytest
from consist import define_step

from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    ASIM_SHARROW_CACHE_DIR,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    ATLAS_VEHICLES2_OUTPUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_H5_UPDATED,
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.binding import (
    ArtifactBindingRule,
    BindingPlan,
    activitysim_datastore_selection_rules,
    beam_preprocess_binding_plan,
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


def _surface_stub(
    *,
    activity_demand_enabled: bool,
    vehicle_ownership_model_enabled: bool = False,
):
    return SimpleNamespace(
        profile=SimpleNamespace(
            activity_demand_enabled=activity_demand_enabled,
            vehicle_ownership_model_enabled=vehicle_ownership_model_enabled,
        ),
        step_surface=lambda _name: None,
    )


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


def test_build_binding_plan_ignores_noop_coupler_placeholders() -> None:
    noop = consist.NoopArtifact(
        key=LINKSTATS,
        path="/tmp/noop-linkstats.csv.gz",
        container_uri="workspace://noop-linkstats.csv.gz",
    )

    plan = build_binding_plan(
        step_name="beam_preprocess",
        explicit_inputs={
            "beam_plans_asim_out": "/tmp/plans.parquet",
            "households_asim_out": "/tmp/households.parquet",
            "persons_asim_out": "/tmp/persons.parquet",
            BEAM_CONFIG_FILE: "/tmp/beam.conf",
        },
        coupler=_CouplerStub({LINKSTATS: noop}),
    )

    assert plan.optional_input_keys == []
    assert plan.source_by_key[LINKSTATS_WARMSTART] == "missing"


def test_beam_preprocess_binding_plan_seeds_default_exchange_and_atlas_inputs(
    monkeypatch,
):
    from pilates.workflows import binding as binding_module

    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_exchange_inputs",
        lambda **_: {
            BEAM_PLANS_IN: "/tmp/plans.parquet",
            BEAM_HOUSEHOLDS_IN: "/tmp/households.parquet",
            BEAM_PERSONS_IN: "/tmp/persons.parquet",
        },
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_warmstart_inputs",
        lambda **_: None,
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_atlas_inputs",
        lambda **_: {ATLAS_VEHICLES2_OUTPUT: "/tmp/vehicles2.csv"},
    )

    plan = beam_preprocess_binding_plan(
        coupler=_CouplerStub({}),
        settings=SimpleNamespace(
            run=SimpleNamespace(models=SimpleNamespace(activity_demand=None)),
            vehicle_ownership_model_enabled=True,
        ),
        state=SimpleNamespace(current_inner_iter=0, forecast_year=2030, year=2030),
        workspace=SimpleNamespace(),
        year=2030,
        activity_demand_outputs=None,
        previous_beam_outputs=None,
        surface=_surface_stub(
            activity_demand_enabled=False,
            vehicle_ownership_model_enabled=True,
        ),
    )

    assert plan.inputs[BEAM_PLANS_IN] == "/tmp/plans.parquet"
    assert plan.inputs[BEAM_HOUSEHOLDS_IN] == "/tmp/households.parquet"
    assert plan.inputs[BEAM_PERSONS_IN] == "/tmp/persons.parquet"
    assert plan.inputs[ATLAS_VEHICLES2_OUTPUT] == "/tmp/vehicles2.csv"
    assert plan.source_by_key[BEAM_PLANS_IN] == "explicit"
    assert plan.source_by_key[ATLAS_VEHICLES2_OUTPUT] == "explicit"


def test_beam_preprocess_binding_plan_prefers_coupler_exchange_inputs_over_defaults(
    monkeypatch,
):
    from pilates.workflows import binding as binding_module

    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_exchange_inputs",
        lambda **_: {
            BEAM_PLANS_IN: "/tmp/default-plans.parquet",
            BEAM_HOUSEHOLDS_IN: "/tmp/default-households.parquet",
            BEAM_PERSONS_IN: "/tmp/default-persons.parquet",
        },
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_warmstart_inputs",
        lambda **_: None,
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_atlas_inputs",
        lambda **_: None,
    )

    plan = beam_preprocess_binding_plan(
        coupler=_CouplerStub(
            {
                BEAM_PLANS_IN: "/tmp/coupler-plans.parquet",
                BEAM_HOUSEHOLDS_IN: "/tmp/coupler-households.parquet",
                BEAM_PERSONS_IN: "/tmp/coupler-persons.parquet",
            }
        ),
        settings=SimpleNamespace(
            run=SimpleNamespace(models=SimpleNamespace(activity_demand=None)),
            vehicle_ownership_model_enabled=False,
        ),
        state=SimpleNamespace(current_inner_iter=0, forecast_year=2030, year=2030),
        workspace=SimpleNamespace(),
        year=2030,
        activity_demand_outputs=None,
        previous_beam_outputs=None,
        surface=_surface_stub(activity_demand_enabled=False),
    )

    assert plan.inputs[BEAM_PLANS_IN] == "/tmp/coupler-plans.parquet"
    assert plan.inputs[BEAM_HOUSEHOLDS_IN] == "/tmp/coupler-households.parquet"
    assert plan.inputs[BEAM_PERSONS_IN] == "/tmp/coupler-persons.parquet"
    assert plan.source_by_key[BEAM_PLANS_IN] == "explicit"


def test_beam_preprocess_binding_plan_prefers_current_atlas_vehicle_with_activity_outputs(
    monkeypatch,
):
    from pilates.workflows import binding as binding_module

    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_exchange_inputs",
        lambda **_: {
            BEAM_PLANS_IN: "/tmp/default-plans.parquet",
            BEAM_HOUSEHOLDS_IN: "/tmp/default-households.parquet",
            BEAM_PERSONS_IN: "/tmp/default-persons.parquet",
        },
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_warmstart_inputs",
        lambda **_: None,
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_atlas_inputs",
        lambda **_: {ATLAS_VEHICLES2_OUTPUT: "/tmp/current-vehicles2.csv"},
    )

    plan = beam_preprocess_binding_plan(
        coupler=_CouplerStub({ATLAS_VEHICLES2_OUTPUT: "/tmp/restored-vehicles.csv.gz"}),
        settings=SimpleNamespace(
            run=SimpleNamespace(models=SimpleNamespace(activity_demand="activitysim")),
            vehicle_ownership_model_enabled=True,
        ),
        state=SimpleNamespace(current_inner_iter=0, forecast_year=2030, year=2030),
        workspace=SimpleNamespace(),
        year=2030,
        activity_demand_outputs={},
        previous_beam_outputs=None,
        surface=_surface_stub(
            activity_demand_enabled=True,
            vehicle_ownership_model_enabled=True,
        ),
    )

    assert plan.inputs[ATLAS_VEHICLES2_OUTPUT] == "/tmp/current-vehicles2.csv"
    assert plan.source_by_key[ATLAS_VEHICLES2_OUTPUT] == "explicit"


def test_beam_preprocess_binding_plan_prefers_restored_atlas_vehicle_without_activity_outputs(
    monkeypatch,
):
    from pilates.workflows import binding as binding_module

    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_exchange_inputs",
        lambda **_: {
            BEAM_PLANS_IN: "/tmp/default-plans.parquet",
            BEAM_HOUSEHOLDS_IN: "/tmp/default-households.parquet",
            BEAM_PERSONS_IN: "/tmp/default-persons.parquet",
        },
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_warmstart_inputs",
        lambda **_: None,
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_atlas_inputs",
        lambda **_: {ATLAS_VEHICLES2_OUTPUT: "/tmp/current-vehicles2.csv"},
    )

    plan = beam_preprocess_binding_plan(
        coupler=_CouplerStub({ATLAS_VEHICLES2_OUTPUT: "/tmp/restored-vehicles.csv.gz"}),
        settings=SimpleNamespace(
            run=SimpleNamespace(models=SimpleNamespace(activity_demand="activitysim")),
            vehicle_ownership_model_enabled=True,
        ),
        state=SimpleNamespace(current_inner_iter=0, forecast_year=2030, year=2030),
        workspace=SimpleNamespace(),
        year=2030,
        activity_demand_outputs=None,
        previous_beam_outputs={},
        surface=_surface_stub(
            activity_demand_enabled=True,
            vehicle_ownership_model_enabled=True,
        ),
    )

    assert plan.inputs[ATLAS_VEHICLES2_OUTPUT] == "/tmp/restored-vehicles.csv.gz"
    assert plan.source_by_key[ATLAS_VEHICLES2_OUTPUT] == "explicit"


def test_beam_preprocess_binding_plan_prefers_previous_linkstats_over_coupler_and_initial(
    monkeypatch,
):
    from pilates.workflows import binding as binding_module

    prev_linkstats = "/tmp/previous-linkstats.csv.gz"
    coupler_linkstats = "/tmp/coupler-linkstats.csv.gz"
    initial_linkstats = "/tmp/initial-linkstats.csv.gz"

    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_exchange_inputs",
        lambda **_: None,
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_warmstart_inputs",
        lambda **_: {LINKSTATS_WARMSTART: initial_linkstats},
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_atlas_inputs",
        lambda **_: None,
    )

    plan = beam_preprocess_binding_plan(
        coupler=_CouplerStub({LINKSTATS_WARMSTART: coupler_linkstats}),
        settings=SimpleNamespace(
            run=SimpleNamespace(models=SimpleNamespace(activity_demand="activitysim")),
            vehicle_ownership_model_enabled=False,
        ),
        state=SimpleNamespace(current_inner_iter=0, forecast_year=2030, year=2030),
        workspace=SimpleNamespace(),
        year=2030,
        activity_demand_outputs=None,
        previous_beam_outputs={LINKSTATS: prev_linkstats},
        surface=_surface_stub(activity_demand_enabled=True),
    )

    assert plan.inputs[LINKSTATS_WARMSTART] == prev_linkstats
    assert plan.source_by_key[LINKSTATS_WARMSTART] == "explicit"


def test_beam_preprocess_binding_plan_forwards_surface_to_build_binding_plan(
    monkeypatch,
):
    from pilates.workflows import binding as binding_module

    captured = {}

    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_exchange_inputs",
        lambda **_: None,
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_warmstart_inputs",
        lambda **_: None,
    )
    monkeypatch.setattr(
        binding_module,
        "_beam_preprocess_atlas_inputs",
        lambda **_: None,
    )

    def _capturing_build_binding_plan(**kwargs):
        captured.update(kwargs)
        return BindingPlan(step_name=kwargs["step_name"])

    monkeypatch.setattr(
        binding_module,
        "build_binding_plan",
        _capturing_build_binding_plan,
    )

    surface = SimpleNamespace(
        profile=SimpleNamespace(activity_demand_enabled=False),
        step_surface=lambda _name: None,
    )
    plan = beam_preprocess_binding_plan(
        coupler=_CouplerStub({}),
        settings=SimpleNamespace(
            run=SimpleNamespace(models=SimpleNamespace(activity_demand=None)),
            vehicle_ownership_model_enabled=False,
        ),
        state=SimpleNamespace(current_inner_iter=0, forecast_year=2030, year=2030),
        workspace=SimpleNamespace(),
        year=2030,
        activity_demand_outputs=None,
        previous_beam_outputs=None,
        surface=surface,
    )

    assert plan.step_name == "beam_preprocess"
    assert captured["surface"] is surface


def test_build_binding_plan_uses_activitysim_preprocess_fallback_provider(monkeypatch):
    from pilates.workflows import binding as binding_module

    monkeypatch.setitem(
        binding_module._FALLBACK_PROVIDERS,
        "activitysim_population_source",
        lambda **_: {
            USIM_POPULATION_SOURCE_H5: "/tmp/base.h5",
            USIM_POPULATION_HOUSEHOLDS_TABLE: "/2030/households",
            USIM_POPULATION_PERSONS_TABLE: "/2030/persons",
            USIM_POPULATION_JOBS_TABLE: "/2030/jobs",
            USIM_POPULATION_BLOCKS_TABLE: "/2030/blocks",
        },
    )

    plan = build_binding_plan(
        step_name="activitysim_preprocess",
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2030),
        workspace=SimpleNamespace(),
        year=2030,
    )

    assert plan.inputs[USIM_POPULATION_SOURCE_H5] == "/tmp/base.h5"
    assert plan.metadata["resolved_values_by_semantic_key"][
        USIM_POPULATION_HOUSEHOLDS_TABLE
    ] == "/2030/households"
    assert plan.source_by_key[USIM_POPULATION_SOURCE_H5] == "fallback"
    assert not plan.missing_required


def test_activitysim_preprocess_binding_prefers_explicit_population_source_over_stale_coupler_current():
    plan = build_binding_plan(
        step_name="activitysim_preprocess",
        coupler=_CouplerStub({USIM_DATASTORE_CURRENT_H5: "/tmp/stale-current.h5"}),
        explicit_inputs={USIM_POPULATION_SOURCE_H5: "/tmp/base.h5"},
        fallback_inputs={USIM_DATASTORE_BASE_H5: "/tmp/base.h5"},
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2030),
        workspace=SimpleNamespace(),
        year=2030,
    )

    assert plan.inputs[USIM_POPULATION_SOURCE_H5] == "/tmp/base.h5"
    assert plan.source_by_key[USIM_POPULATION_SOURCE_H5] == "explicit"


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


def test_activitysim_postprocess_current_datastore_does_not_fallback_to_local_input(
    monkeypatch,
):
    from pilates.workflows import binding as binding_module

    monkeypatch.setitem(
        binding_module._FALLBACK_PROVIDERS,
        "activitysim_input_datastore",
        lambda **_: {USIM_DATASTORE_CURRENT_H5: "/tmp/input.h5"},
    )

    plan = build_binding_plan(
        step_name="activitysim_postprocess",
        required_keys=[USIM_DATASTORE_CURRENT_H5],
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2030),
        workspace=SimpleNamespace(),
        year=2030,
    )

    assert USIM_DATASTORE_CURRENT_H5 not in (plan.inputs or {})
    assert plan.missing_required == [USIM_DATASTORE_CURRENT_H5]


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


def test_build_binding_plan_records_urbansim_candidate_paths_metadata(tmp_path):
    workspace_root = tmp_path / "workspace"
    mutable_usim_dir = workspace_root / "usim"
    archive_root = tmp_path / "archive"
    mutable_usim_dir.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    settings = SimpleNamespace(
        run=SimpleNamespace(region="region-a"),
        urbansim=SimpleNamespace(
            input_file_template="input_{region_id}.h5",
            output_file_template="output_{year}.h5",
            region_mappings={"region_to_region_id": {"region-a": "001"}},
        ),
    )
    state = SimpleNamespace(
        year=2030,
        is_start_year=lambda: False,
        is_enabled=lambda stage: stage == "land_use",
        Stage=SimpleNamespace(land_use="land_use"),
        run_info_path=str(archive_root / "workflow.json"),
    )
    workspace = SimpleNamespace(
        full_path=str(workspace_root),
        get_usim_mutable_data_dir=lambda: str(mutable_usim_dir),
    )

    archive_input = archive_root / "usim" / "input_001.h5"
    archive_input.parent.mkdir(parents=True, exist_ok=True)
    archive_input.write_text("input", encoding="utf-8")
    archive_output = archive_root / "usim" / "output_2030.h5"
    archive_output.write_text("output", encoding="utf-8")

    plan = build_binding_plan(
        step_name="urbansim_input_selection",
        settings=settings,
        state=state,
        workspace=workspace,
        year=2030,
        artifact_rules=urbansim_datastore_selection_rules(),
        required_keys=[USIM_DATASTORE_BASE_H5, USIM_DATASTORE_CURRENT_H5],
    )

    candidate_paths = plan.metadata["candidate_paths_by_semantic_key"]
    assert candidate_paths[USIM_DATASTORE_BASE_H5] == [
        str(mutable_usim_dir / "input_001.h5"),
        str(archive_input),
    ]
    assert candidate_paths[USIM_DATASTORE_CURRENT_H5] == [
        str(mutable_usim_dir / "output_2030.h5"),
        str(archive_output),
    ]
    assert plan.inputs[USIM_DATASTORE_BASE_H5] == str(archive_input)
    assert plan.inputs[USIM_DATASTORE_CURRENT_H5] == str(archive_output)


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
