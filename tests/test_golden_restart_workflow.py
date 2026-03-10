"""
Golden restart workflow contract test.

This focuses on a single realistic resume point:

1. land use and vehicle ownership already completed
2. required local ActivitySim restart inputs are lost from scratch
3. archive rehydration repairs the local workspace
4. supply-demand resumes without bootstrap
"""

from __future__ import annotations

import shutil
from pathlib import Path

import run as run_module
from pilates.activitysim.preprocessor import required_asim_config_dirs
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage

pytest_plugins = ("tests.test_golden_stub_workflow",)


def _write_file(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_golden_restart_rehydrates_local_workspace_and_resumes_supply_demand(
    golden_stub_env,
    tmp_path,
):
    settings = golden_stub_env["settings"]
    workspace = golden_stub_env["workspace"]
    state = golden_stub_env["state"]
    scenario = golden_stub_env["scenario"]
    coupler = golden_stub_env["coupler"]

    state.set_data_initialized(True)

    asim_configs_root = Path(workspace.get_asim_mutable_configs_dir())
    for dirname in required_asim_config_dirs(settings.activitysim.main_configs_dir):
        _write_file(
            asim_configs_root / dirname / "settings.yaml",
            f"{dirname}: true\n",
        )

    outputs_holder_year = StepOutputsHolder()
    usim_inputs = run_land_use_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        outputs_holder_year=outputs_holder_year,
    )
    run_vehicle_ownership_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )

    archive_run_dir = tmp_path / "archive-run"
    shutil.copytree(Path(workspace.full_path), archive_run_dir)

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    local_households = Path(workspace.get_asim_mutable_data_dir()) / "households.csv"
    local_persons = Path(workspace.get_asim_mutable_data_dir()) / "persons.csv"
    local_land_use = Path(workspace.get_asim_mutable_data_dir()) / "land_use.csv"
    local_mp_settings = asim_configs_root / "configs_mp" / "settings.yaml"
    local_compile_settings = asim_configs_root / "configs_sh_compile" / "settings.yaml"

    for path in (
        local_households,
        local_persons,
        local_land_use,
        local_mp_settings,
        local_compile_settings,
    ):
        path.unlink()

    missing_before = run_module._find_missing_restart_local_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    assert {item["key"] for item in missing_before} >= {
        "activitysim_input_households.csv",
        "activitysim_input_persons.csv",
        "activitysim_input_land_use.csv",
        "activitysim_config_settings_yaml_configs_mp",
        "activitysim_config_settings_yaml_configs_sh_compile",
    }

    summary = run_module._rehydrate_full_local_run_from_archive(
        local_run_dir=str(workspace.full_path),
        archive_run_dir=str(archive_run_dir),
    )

    assert summary["copied"] >= 5
    assert local_households.exists()
    assert local_persons.exists()
    assert local_land_use.exists()
    assert local_mp_settings.exists()
    assert local_compile_settings.exists()

    missing_after = run_module._find_missing_restart_local_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    assert missing_after == []

    manifest_dir = tmp_path / "golden_restart_manifests"

    def _build_manifest_path(_workspace, year, iteration):
        manifest_dir.mkdir(parents=True, exist_ok=True)
        return manifest_dir / f"manifest_{year}_{iteration}.yaml"

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )

    manifest_path = manifest_dir / f"manifest_{state.forecast_year}_0.yaml"
    assert manifest_path.exists()
