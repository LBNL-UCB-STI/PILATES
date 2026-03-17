from __future__ import annotations

from pilates.config.models import FullSkimsCreatorConfig, load_config
from pilates.workflows.lineage_render import (
    render_plan_html,
    render_plan_json,
    render_plan_mermaid,
)
from pilates.workflows.planning import (
    build_static_execution_plan,
    build_static_execution_plan_from_file,
)


def test_static_execution_plan_builds_activitysim_beam_sequence_from_config_file():
    plan = build_static_execution_plan_from_file("settings-seattle-consist-local.yaml")

    assert plan.metadata["years"] == [{"year": 2018, "forecast_year": 2018}]
    assert [step.step_name for step in plan.step_runs[:7]] == [
        "activitysim_preprocess",
        "activitysim_compile",
        "activitysim_run",
        "activitysim_postprocess",
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
    ]
    assert plan.step_runs[-1].step_name == "postprocessing"

    compile_artifacts = [
        artifact
        for artifact in plan.artifacts
        if artifact.producer_step_run_id == plan.step_runs[1].id
    ]
    assert any(artifact.artifact_key == "zarr_skims" for artifact in compile_artifacts)
    assert any(
        artifact.instance_key == "activitysim_compile:y2018:i0:zarr_skims"
        for artifact in compile_artifacts
    )

    json_payload = render_plan_json(plan)
    assert '"step_name": "activitysim_run"' in json_payload
    assert '"kind": "depends_on"' in json_payload


def test_static_execution_plan_expands_land_use_and_atlas_subyears():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True

    plan = build_static_execution_plan(settings, include_postprocessing=False)

    first_year_steps = [step for step in plan.step_runs if step.year == 2017]
    assert [step.step_name for step in first_year_steps[:3]] == [
        "urbansim_preprocess",
        "urbansim_run",
        "urbansim_postprocess",
    ]

    atlas_preprocess_runs = [
        step
        for step in first_year_steps
        if step.step_name == "atlas_preprocess"
    ]
    assert [step.atlas_year for step in atlas_preprocess_runs] == [2017, 2019, 2021, 2023]


def test_static_execution_plan_marks_underdeclared_contracts_and_renders_mermaid():
    settings = load_config("scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True

    plan = build_static_execution_plan(settings, include_postprocessing=False)

    gap_kinds = {(gap.kind, gap.message) for gap in plan.contract_gaps}
    assert any(kind == "underdeclared_inputs" for kind, _message in gap_kinds)

    mermaid = render_plan_mermaid(plan)
    assert mermaid.startswith("flowchart TD")
    assert "urbansim_preprocess" in mermaid
    assert "No declared input contract is available for this step." in mermaid


def test_static_execution_plan_renders_html_wrapper():
    plan = build_static_execution_plan_from_file("settings-seattle-consist-local.yaml")

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
    plan = build_static_execution_plan_from_file("settings-seattle-consist-local.yaml")

    html = render_plan_html(plan, hide_terminal_artifacts=True)
    mermaid = render_plan_mermaid(plan, hide_terminal_artifacts=True)

    assert "Render Filter" in html
    assert "hide terminal artifacts" in html
    assert "Rendered artifacts:" in html
    assert "postprocessing" in mermaid
    assert "usim_input_next" not in mermaid


def test_hide_terminal_artifacts_keeps_only_valid_edges():
    from pilates.workflows.lineage_render import _filtered_plan_for_render

    plan = build_static_execution_plan_from_file("settings-seattle-consist-local.yaml")
    filtered = _filtered_plan_for_render(plan, hide_terminal_artifacts=True)
    node_ids = {step.id for step in filtered.step_runs} | {
        artifact.id for artifact in filtered.artifacts
    }

    assert filtered.artifacts
    assert all(edge.source in node_ids and edge.target in node_ids for edge in filtered.edges)


def test_planned_artifacts_render_as_distinct_instances_for_reused_canonical_keys():
    plan = build_static_execution_plan_from_file("settings-seattle-consist-local.yaml")

    zarr_instances = {
        artifact.instance_key
        for artifact in plan.artifacts
        if artifact.canonical_key == "zarr_skims"
    }

    assert "activitysim_compile:y2018:i0:zarr_skims" in zarr_instances
    assert "beam_postprocess:y2018:i0:zarr_skims" in zarr_instances


def test_static_execution_plan_honors_after_final_iteration_full_skim_schedule():
    settings = load_config("settings-seattle-consist-local.yaml")
    settings.run.supply_demand_iters = 3
    settings.beam.full_skim = FullSkimsCreatorConfig(run_schedule="after_final_iteration")

    plan = build_static_execution_plan(settings, include_postprocessing=False)

    assert [step.step_name for step in plan.step_runs].count("beam_full_skim") == 1
    last_step_names = [step.step_name for step in plan.step_runs[-3:]]
    assert last_step_names == [
        "beam_run",
        "beam_postprocess",
        "beam_full_skim",
    ]
