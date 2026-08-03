from __future__ import annotations

from pilates.config.models import FullSkimsCreatorConfig, load_config
from pilates.workflows.artifact_keys import FINAL_SKIMS_OMX, OMX_SKIMS
from pilates.workflows.lineage_render import (
    render_plan_html,
    render_plan_json,
    render_plan_mermaid,
)
from pilates.workflows.planning import (
    PlannedStepRun,
    _PlanBuilder,
    _atlas_sub_years,
    build_static_execution_plan,
    build_static_execution_plan_from_file,
)
from pilates.workflows.surface import build_enabled_workflow_surface


SEATTLE_ACTIVITYSIM_BEAM = "scenarios/seattle/settings-new-asim-seattle-null.yaml"


def _build_plan(settings, *, include_postprocessing=False):
    return build_static_execution_plan(
        settings,
        include_postprocessing=include_postprocessing,
        surface=build_enabled_workflow_surface(settings),
    )


def test_static_execution_plan_builds_activitysim_beam_sequence_from_config_file():
    plan = build_static_execution_plan_from_file(SEATTLE_ACTIVITYSIM_BEAM)

    assert plan.metadata["years"] == [{"year": 2018, "forecast_year": 2018}]
    assert [step.step_name for step in plan.step_runs[:6]] == [
        "activitysim_preprocess",
        "activitysim_run",
        "activitysim_postprocess",
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
    ]
    assert plan.step_runs[-1].step_name == "postprocessing"

    activitysim_run = plan.step_runs[1]
    zarr_artifact = next(
        artifact
        for artifact in plan.artifacts
        if artifact.instance_key == "external:y2018:i0:zarr_skims"
    )
    assert zarr_artifact.external is True
    assert any(
        edge.source == zarr_artifact.id
        and edge.target == activitysim_run.id
        and edge.kind == "consumes"
        for edge in plan.edges
    )
    produced_zarr = [
        artifact
        for artifact in plan.artifacts
        if artifact.producer_step_run_id == activitysim_run.id
        and artifact.artifact_key == "zarr_skims"
    ]
    assert [artifact.instance_key for artifact in produced_zarr] == [
        "activitysim_run:y2018:i0:zarr_skims"
    ]

    json_payload = render_plan_json(plan)
    assert '"step_name": "activitysim_run"' in json_payload
    assert '"kind": "depends_on"' in json_payload
    removed_alias = "_".join(("upstream", "step", "inputs"))
    assert removed_alias not in json_payload

    html_payload = render_plan_html(plan)
    assert removed_alias not in html_payload


def test_postprocessing_is_untracked_and_only_reports_its_input_contract_gap():
    plan = build_static_execution_plan_from_file(SEATTLE_ACTIVITYSIM_BEAM)

    postprocessing_step = next(
        step for step in plan.step_runs if step.step_name == "postprocessing"
    )
    postprocessing_gaps = [
        gap for gap in plan.contract_gaps if gap.step_run_id == postprocessing_step.id
    ]

    assert [gap.kind for gap in postprocessing_gaps] == ["underdeclared_inputs"]


def test_tracked_step_with_no_outputs_still_reports_underdeclared_output_gap():
    settings = load_config(SEATTLE_ACTIVITYSIM_BEAM)
    builder = _PlanBuilder(
        settings=settings,
        surface=build_enabled_workflow_surface(settings),
        config_path=None,
        include_postprocessing=False,
    )
    step_run = PlannedStepRun(
        id="step_test",
        sequence=1,
        step_name="activitysim_run",
        stage_name="activity_demand",
        phase="run",
        label="activitysim_run",
    )
    empty_contract = {
        "input_keys": [],
        "optional_input_keys": [],
        "dynamic_input_families": [],
        "output_keys": [],
        "optional_output_keys": [],
        "dynamic_output_families": [],
        "depends_on": [],
    }

    builder._add_contract_gaps(step_run, empty_contract)

    assert [gap.kind for gap in builder.plan.contract_gaps] == ["underdeclared_outputs"]


def test_static_execution_plan_activitysim_beam_run_excludes_land_use_and_atlas_steps():
    settings = load_config(SEATTLE_ACTIVITYSIM_BEAM)
    settings.land_use_enabled = False
    settings.vehicle_ownership_model_enabled = False
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True

    plan = _build_plan(settings, include_postprocessing=False)

    step_names = [step.step_name for step in plan.step_runs]
    assert any(name.startswith("activitysim_") for name in step_names)
    assert any(name.startswith("beam_") for name in step_names)
    assert all(not name.startswith("urbansim_") for name in step_names)
    assert all(not name.startswith("atlas_") for name in step_names)


def test_static_execution_plan_expands_land_use_and_atlas_subyears():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True

    plan = _build_plan(settings, include_postprocessing=False)

    first_year_steps = [step for step in plan.step_runs if step.year == 2017]
    assert [step.step_name for step in first_year_steps[:2]] == [
        "urbansim_run",
        "urbansim_postprocess",
    ]
    assert "urbansim_preprocess" not in [step.step_name for step in plan.step_runs]

    atlas_preprocess_runs = [
        step for step in first_year_steps if step.step_name == "atlas_preprocess"
    ]
    assert [step.atlas_year for step in atlas_preprocess_runs] == _atlas_sub_years(
        2017,
        plan.metadata["years"][0]["forecast_year"],
    )


def test_static_execution_plan_uses_settings_aware_atlas_scenario_contracts():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = False
    settings.traffic_assignment_enabled = False
    settings.atlas.scenario = "zev_mandate"
    settings.atlas.adscen = "zev_mandate"

    plan = _build_plan(settings, include_postprocessing=False)

    atlas_static_artifacts = {
        artifact.canonical_key
        for artifact in plan.artifacts
        if artifact.canonical_key.startswith("adopt/")
    }

    assert any(key.startswith("adopt/zev_mandate/") for key in atlas_static_artifacts)
    assert all(not key.startswith("adopt/baseline/") for key in atlas_static_artifacts)
    assert all(not key.startswith("adopt/ess_cons/") for key in atlas_static_artifacts)


def test_static_execution_plan_filters_future_atlas_adopt_snapshots_by_subyear():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = False
    settings.traffic_assignment_enabled = False
    settings.atlas.scenario = "baseline"
    settings.atlas.adscen = "baseline"

    plan = _build_plan(settings, include_postprocessing=False)

    atlas_preprocess_2021 = next(
        step
        for step in plan.step_runs
        if step.step_name == "atlas_preprocess"
        and step.year == 2017
        and step.atlas_year == 2021
    )
    preprocess_outputs_2021 = {
        artifact.canonical_key
        for artifact in plan.artifacts
        if artifact.producer_step_run_id == atlas_preprocess_2021.id
    }

    assert "adopt/baseline/new_vehicles_biannual_values_2017" in preprocess_outputs_2021
    assert "adopt/baseline/new_vehicles_biannual_values_2019" in preprocess_outputs_2021
    assert "adopt/baseline/new_vehicles_biannual_values_2021" in preprocess_outputs_2021
    assert (
        "adopt/baseline/new_vehicles_biannual_values_2023"
        not in preprocess_outputs_2021
    )
    assert "adopt/baseline/used_vehicles_2023" not in preprocess_outputs_2021

    atlas_run_2021 = next(
        step
        for step in plan.step_runs
        if step.step_name == "atlas_run"
        and step.year == 2017
        and step.atlas_year == 2021
    )
    consumed_artifact_ids = {
        edge.source
        for edge in plan.edges
        if edge.kind == "consumes" and edge.target == atlas_run_2021.id
    }
    run_inputs_2021 = {
        artifact.canonical_key
        for artifact in plan.artifacts
        if artifact.id in consumed_artifact_ids
    }

    assert "adopt/baseline/new_vehicles_biannual_values_2021" in run_inputs_2021
    assert "adopt/baseline/new_vehicles_biannual_values_2023" not in run_inputs_2021
    assert "adopt/baseline/used_vehicles_2023" not in run_inputs_2021


def test_static_execution_plan_threads_atlas_vehicles2_from_atlas_postprocess():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True

    plan = _build_plan(settings, include_postprocessing=False)

    vehicles2_artifacts = [
        artifact
        for artifact in plan.artifacts
        if artifact.canonical_key == "atlas_vehicles2_output"
    ]
    assert vehicles2_artifacts
    assert all(
        artifact.producer_step_run_id is not None for artifact in vehicles2_artifacts
    )
    assert all(
        "external" not in artifact.instance_key for artifact in vehicles2_artifacts
    )


def test_static_execution_plan_coalesces_final_skims_omx_external_artifact():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True

    plan = _build_plan(settings, include_postprocessing=False)

    final_skims_artifacts = [
        artifact
        for artifact in plan.artifacts
        if artifact.canonical_key == FINAL_SKIMS_OMX and artifact.external
    ]

    assert len(final_skims_artifacts) == 1
    artifact = final_skims_artifacts[0]
    assert artifact.instance_key == f"external:{FINAL_SKIMS_OMX}"
    assert artifact.year is None
    assert artifact.forecast_year is None

    consuming_edges = [
        edge
        for edge in plan.edges
        if edge.source == artifact.id and edge.kind == "consumes"
    ]
    consuming_step_names = {
        next(step.step_name for step in plan.step_runs if step.id == edge.target)
        for edge in consuming_edges
    }
    assert "urbansim_run" in consuming_step_names
    assert "urbansim_preprocess" not in consuming_step_names


def test_static_execution_plan_uses_final_skims_omx_not_generic_omx_fallback():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True

    plan = _build_plan(settings, include_postprocessing=False)

    generic_omx_artifacts = [
        artifact
        for artifact in plan.artifacts
        if artifact.canonical_key == OMX_SKIMS and artifact.external
    ]
    assert generic_omx_artifacts == []

    final_skims_artifact = next(
        artifact
        for artifact in plan.artifacts
        if artifact.canonical_key == FINAL_SKIMS_OMX and artifact.external
    )
    assert final_skims_artifact.instance_key == f"external:{FINAL_SKIMS_OMX}"

    consuming_edges = [
        edge
        for edge in plan.edges
        if edge.source == final_skims_artifact.id and edge.kind == "consumes"
    ]
    consuming_step_names = {
        next(step.step_name for step in plan.step_runs if step.id == edge.target)
        for edge in consuming_edges
    }
    assert "activitysim_preprocess" in consuming_step_names


def test_static_execution_plan_distinguishes_usim_semantic_roles_from_path_hints():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = False

    plan = _build_plan(settings, include_postprocessing=False)
    steps_by_id = {step.id: step for step in plan.step_runs}

    base_artifact = next(
        artifact
        for artifact in plan.artifacts
        if artifact.canonical_key == "usim_datastore_base_h5"
    )
    current_handoff_artifact = next(
        artifact
        for artifact in plan.artifacts
        if artifact.canonical_key == "usim_datastore_h5"
        and artifact.producer_step_run_id is not None
        and steps_by_id[artifact.producer_step_run_id].step_name
        == "activitysim_postprocess"
    )
    forecast_output_artifact = next(
        artifact
        for artifact in plan.artifacts
        if artifact.canonical_key == "usim_datastore_h5"
        and artifact.producer_step_run_id is not None
        and steps_by_id[artifact.producer_step_run_id].step_name == "urbansim_run"
    )

    assert base_artifact.path_role == "semantic_base_datastore"
    assert current_handoff_artifact.path_role == "current_mutable_datastore"
    assert forecast_output_artifact.path_role == "forecast_output_datastore"

    assert base_artifact.resolved_path_hint is not None
    assert (
        current_handoff_artifact.resolved_path_hint == base_artifact.resolved_path_hint
    )
    assert base_artifact.producer_step_run_id is None
    assert forecast_output_artifact.resolved_path_hint is not None
    assert (
        forecast_output_artifact.resolved_path_hint != base_artifact.resolved_path_hint
    )

    assert "same physical input-slot path" in str(current_handoff_artifact.path_notes)


def test_static_execution_plan_renders_mermaid_without_contract_gaps():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True

    plan = _build_plan(settings, include_postprocessing=False)

    assert plan.contract_gaps == []

    mermaid = render_plan_mermaid(plan)
    assert mermaid.startswith("flowchart TD")
    assert "urbansim_preprocess" not in mermaid
    assert "atlas_run" in mermaid


def test_static_execution_plan_renders_html_wrapper():
    plan = build_static_execution_plan_from_file(SEATTLE_ACTIVITYSIM_BEAM)

    html = render_plan_html(plan)

    assert "<!doctype html>" in html
    assert "PILATES Workflow Lineage" in html
    assert "https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/+esm" in html
    assert "drag nodes to reposition boxes" in html
    assert "Ordered Layout" in html
    assert "Relax Artifacts" in html
    assert "name: 'preset'" in html
    assert "lockStepNodes" in html
    assert "unlockStepNodes" in html
    assert "nodeRepulsion: 36000" in html
    assert "idealEdgeLength: 240" in html
    assert '"kind": "step"' in html
    assert "activitysim_preprocess" in html
    assert "year=2018" in html


def test_renderers_can_hide_terminal_artifacts():
    plan = build_static_execution_plan_from_file(SEATTLE_ACTIVITYSIM_BEAM)

    html = render_plan_html(plan, hide_terminal_artifacts=True)
    mermaid = render_plan_mermaid(plan, hide_terminal_artifacts=True)

    assert "Render Filter" in html
    assert "hide terminal artifacts" in html
    assert "Rendered artifacts:" in html
    assert "postprocessing" in mermaid
    assert "usim_input_next" not in mermaid


def test_hide_terminal_artifacts_keeps_only_valid_edges():
    from pilates.workflows.lineage_render import _filtered_plan_for_render

    plan = build_static_execution_plan_from_file(SEATTLE_ACTIVITYSIM_BEAM)
    filtered = _filtered_plan_for_render(plan, hide_terminal_artifacts=True)
    node_ids = {step.id for step in filtered.step_runs} | {
        artifact.id for artifact in filtered.artifacts
    }

    assert filtered.artifacts
    assert all(
        edge.source in node_ids and edge.target in node_ids for edge in filtered.edges
    )


def test_planned_artifacts_render_as_distinct_instances_for_reused_canonical_keys():
    plan = build_static_execution_plan_from_file(SEATTLE_ACTIVITYSIM_BEAM)

    zarr_instances = {
        artifact.instance_key
        for artifact in plan.artifacts
        if artifact.canonical_key == "zarr_skims"
    }

    assert "external:y2018:i0:zarr_skims" in zarr_instances
    assert "beam_postprocess:y2018:i0:zarr_skims" in zarr_instances


def test_static_execution_plan_honors_after_final_iteration_full_skim_schedule():
    settings = load_config(SEATTLE_ACTIVITYSIM_BEAM)
    settings.run.supply_demand_iters = 3
    settings.beam.full_skim = FullSkimsCreatorConfig(
        run_schedule="after_final_iteration"
    )

    plan = _build_plan(settings, include_postprocessing=False)

    assert [step.step_name for step in plan.step_runs].count("beam_full_skim") == 1
    last_step_names = [step.step_name for step in plan.step_runs[-3:]]
    assert last_step_names == [
        "beam_run",
        "beam_postprocess",
        "beam_full_skim",
    ]


def test_static_execution_plan_supports_beam_only_run():
    settings = load_config(SEATTLE_ACTIVITYSIM_BEAM)
    settings.run.models.activity_demand = None
    settings.activity_demand_enabled = False
    settings.run.models.land_use = None
    settings.land_use_enabled = False
    settings.run.models.vehicle_ownership = None
    settings.vehicle_ownership_model_enabled = False
    settings.run.models.travel = "beam"
    settings.traffic_assignment_enabled = True

    plan = _build_plan(settings, include_postprocessing=False)

    step_names = [step.step_name for step in plan.step_runs]
    assert any(name.startswith("beam_") for name in step_names)
    assert all(not name.startswith("activitysim_") for name in step_names)
    assert all(not name.startswith("urbansim_") for name in step_names)
    assert all(not name.startswith("atlas_") for name in step_names)
