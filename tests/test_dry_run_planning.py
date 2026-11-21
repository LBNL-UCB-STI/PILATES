from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from pilates.config.models import PilatesConfig, load_config
import pilates.workflows.orchestrator  # noqa: F401 - warm import before timing
from pilates.workflows.dry_run import (
    ExecutionPlan,
    plan_pilates,
    render_plan_html,
    render_plan_json,
    render_plan_mermaid,
)

SEATTLE_ACTIVITYSIM_BEAM = "scenarios/seattle/settings-new-asim-seattle-null.yaml"
SFBAY_FULL_USIM = "scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml"

_BEAM_ONLY_YAML = """\
run:
  region: sfbay
  scenario: base
  start_year: 2017
  end_year: 2017
  output_directory: /tmp/pilates_dry_run
  output_run_name: beam_only
  models:
    travel: beam
shared:
  geography:
    FIPS: {}
    local_crs: "EPSG:26910"
  skims:
    fname: skims.omx
  database:
    path: /tmp/pilates_dry_run/provenance.duckdb
infrastructure: {}
beam:
  config: sfbay-pilates-base-calibrated.conf
  local_input_folder: pilates/beam/input
  local_mutable_data_folder: pilates/beam/mutable
  local_output_folder: pilates/beam/output
  scenario_folder: urbansim
  router_directory: pilates/beam/r5
  skims_shapefile: taz_sfbay.geojson
  skim_zone_source_id_col: taz
  skim_zone_geoid_col: GEOID
postprocessing:
  enabled: true
  output_folder: postprocessing
  mep_output_folder: mep
"""


def _make_beam_only_settings(output_dir: Path | None = None) -> PilatesConfig:
    settings = PilatesConfig.model_validate(yaml.safe_load(_BEAM_ONLY_YAML))
    if output_dir is not None:
        _set_output_dir(settings, output_dir)
    return settings


def _set_output_dir(settings, target: Path) -> None:
    settings.run.output_directory = str(target)


def test_plan_pilates_multi_year_full_stack_and_subyears(tmp_path: Path):
    settings = load_config(SFBAY_FULL_USIM)
    settings.run.start_year = 2017
    settings.run.end_year = 2019
    settings.run.travel_model_freq = 2
    settings.run.max_year_intervals = 2
    settings.run.models.land_use = "urbansim"
    settings.run.models.vehicle_ownership = "atlas"
    settings.run.models.activity_demand = "activitysim"
    settings.run.models.travel = "beam"
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True
    missing_out = tmp_path / "must_not_be_created"
    _set_output_dir(settings, missing_out)

    t0 = time.perf_counter()
    plan = plan_pilates(settings, include_postprocessing=True)
    elapsed = time.perf_counter() - t0

    assert isinstance(plan, ExecutionPlan)
    assert elapsed < 0.5, f"plan_pilates took {elapsed:.3f}s (>= 0.5s)"
    assert not missing_out.exists()

    step_names = [step.step_name for step in plan.steps]
    assert "urbansim_run" in step_names
    assert "urbansim_postprocess" in step_names
    assert "atlas_preprocess" in step_names
    assert "atlas_run" in step_names
    assert "atlas_postprocess" in step_names
    assert "activitysim_preprocess" in step_names
    assert "activitysim_run" in step_names
    assert "activitysim_postprocess" in step_names
    assert "beam_preprocess" in step_names
    assert "beam_run" in step_names
    assert "beam_postprocess" in step_names
    if settings.postprocessing is not None:
        assert "postprocessing" in step_names

    # Multi-year: both 2017 and 2019 intervals should be planned
    years = {step.year for step in plan.steps}
    assert 2017 in years
    assert 2019 in years

    # Verify dataflow edges connect producers to consumers
    assert any(
        e.source_step.startswith("urbansim_run_")
        and e.target_step.startswith("urbansim_postprocess_")
        for e in plan.edges
    )
    assert any(
        e.source_step.startswith("activitysim_run_")
        and e.target_step.startswith("activitysim_postprocess_")
        for e in plan.edges
    )
    assert any(
        e.source_step.startswith("beam_run_")
        and e.target_step.startswith("beam_postprocess_")
        for e in plan.edges
    )


def test_plan_pilates_activitysim_only(tmp_path: Path):
    settings = load_config(SEATTLE_ACTIVITYSIM_BEAM)
    settings.run.models.land_use = None
    settings.run.models.vehicle_ownership = None
    settings.run.models.activity_demand = "activitysim"
    settings.run.models.travel = None
    settings.land_use_enabled = False
    settings.vehicle_ownership_model_enabled = False
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = False
    settings.run.supply_demand_iters = 1
    missing_out = tmp_path / "asim_only_out"
    _set_output_dir(settings, missing_out)

    t0 = time.perf_counter()
    plan = plan_pilates(settings, include_postprocessing=False)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5
    assert not missing_out.exists()

    step_names = [step.step_name for step in plan.steps]
    assert step_names == [
        "activitysim_preprocess",
        "activitysim_run",
        "activitysim_postprocess",
    ]


def test_plan_pilates_beam_only_with_and_without_postprocessing(tmp_path: Path):
    missing_out = tmp_path / "beam_only_out"
    settings = _make_beam_only_settings(missing_out)

    t0 = time.perf_counter()
    plan_with_post = plan_pilates(settings, include_postprocessing=True)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5
    assert not missing_out.exists()
    assert [s.step_name for s in plan_with_post.steps] == [
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
        "postprocessing",
    ]

    had_postprocessing = settings.postprocessing is not None
    plan_no_post = plan_pilates(settings, include_postprocessing=False)
    assert (settings.postprocessing is not None) == had_postprocessing
    assert [s.step_name for s in plan_no_post.steps] == [
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
    ]


def test_plan_renderers_json_mermaid_and_html():
    settings = _make_beam_only_settings()
    plan = plan_pilates(settings, include_postprocessing=True)

    json_text = render_plan_json(plan)
    parsed = json.loads(json_text)
    assert len(parsed["steps"]) == len(plan.steps)
    assert len(parsed["edges"]) == len(plan.edges)

    mermaid_full = render_plan_mermaid(plan, hide_terminal_artifacts=False)
    assert mermaid_full.startswith("flowchart TD")
    assert "beam_run_y2017_i0" in mermaid_full
    assert "term_1" in mermaid_full

    mermaid_compact = render_plan_mermaid(plan, hide_terminal_artifacts=True)
    assert "term_1" not in mermaid_compact

    html_text = render_plan_html(plan, hide_terminal_artifacts=True)
    assert "<!DOCTYPE html>" in html_text
    assert "mermaid" in html_text
    assert "beam_run_y2017_i0" in html_text


def test_render_workflow_lineage_cli_outputs_clean_json(tmp_path: Path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(_BEAM_ONLY_YAML, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "pilates/scripts/render_workflow_lineage.py",
            "-c",
            str(config_path),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(proc.stdout)
    assert [step["step_name"] for step in parsed["steps"]] == [
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
        "postprocessing",
    ]
