from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
)
from pilates.workflows.stages.land_use import run_land_use_stage as _run_land_use_stage
from pilates.workflows.steps import StepOutputsHolder
from tests.workflow_contract_harness import CouplerStub, build_runtime_context

pytest_plugins = ("tests.test_stage_contracts",)


def run_land_use_stage(
    *, context=None, settings=None, state=None, workspace=None, **kwargs
):
    context = context or build_runtime_context(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return _run_land_use_stage(context=context, **kwargs)


def _land_use_manifest_path(*, workspace, year: int) -> Path:
    return Path(workspace.full_path) / ".workflow" / f"land_use_year_{year}.yaml"


def test_land_use_stage_persists_year_scoped_run_ids_and_restores_from_manifest(
    stage_env, monkeypatch
):
    scenario = stage_env["scenario"]
    state = stage_env["state"]
    settings = stage_env["settings"]
    workspace = stage_env["workspace"]

    original_run = scenario.run
    run_counter = {"value": 0}

    def _run_with_run_id(**kwargs):
        original_run(**kwargs)
        step_meta = getattr(kwargs["fn"], "__consist_step__", None)
        model_name = getattr(step_meta, "model", "unknown_step")
        year = kwargs.get("year", "unknown_year")
        run_id = f"{model_name}__y{year}__n{run_counter['value']}"
        run_counter["value"] += 1
        return SimpleNamespace(cache_hit=False, run=SimpleNamespace(id=run_id))

    monkeypatch.setattr(scenario, "run", _run_with_run_id)

    year_a = int(state.forecast_year)
    year_b = year_a + 1

    first_outputs = StepOutputsHolder()
    first_usim_inputs = run_land_use_stage(
        scenario=scenario,
        coupler=stage_env["coupler"],
        year=year_a,
        outputs_holder_year=first_outputs,
        context=stage_env["context"],
    )

    second_outputs = StepOutputsHolder()
    run_land_use_stage(
        scenario=scenario,
        coupler=stage_env["coupler"],
        year=year_b,
        outputs_holder_year=second_outputs,
        context=stage_env["context"],
    )

    manifest_path_a = _land_use_manifest_path(workspace=workspace, year=year_a)
    manifest_path_b = _land_use_manifest_path(workspace=workspace, year=year_b)
    assert manifest_path_a.exists()
    assert manifest_path_b.exists()

    manifest_a = yaml.safe_load(manifest_path_a.read_text(encoding="utf-8"))
    manifest_b = yaml.safe_load(manifest_path_b.read_text(encoding="utf-8"))
    expected_steps = {"urbansim_preprocess", "urbansim_run", "urbansim_postprocess"}
    assert set(manifest_a.keys()) == expected_steps
    assert set(manifest_b.keys()) == expected_steps

    run_ids_a = {step: manifest_a[step]["run_id"] for step in expected_steps}
    run_ids_b = {step: manifest_b[step]["run_id"] for step in expected_steps}
    assert all(run_ids_a.values())
    assert all(run_ids_b.values())
    assert set(run_ids_a.values()).isdisjoint(set(run_ids_b.values()))

    class _FailOnRunScenario:
        def run(self, **_kwargs):
            raise AssertionError("land_use manifest restore should skip scenario.run")

    restored_coupler = CouplerStub()
    restored_outputs = StepOutputsHolder()
    restored_usim_inputs = run_land_use_stage(
        scenario=_FailOnRunScenario(),
        coupler=restored_coupler,
        year=year_a,
        outputs_holder_year=restored_outputs,
        context=stage_env["context"],
    )

    assert restored_outputs.urbansim_preprocess is not None
    assert restored_outputs.urbansim_run is not None
    assert restored_outputs.urbansim_postprocess is not None
    assert restored_coupler.get(USIM_DATASTORE_H5) is not None
    assert restored_coupler.get(USIM_DATASTORE_BASE_H5) is not None
    assert (
        restored_usim_inputs[USIM_DATASTORE_CURRENT_H5]
        == first_usim_inputs[USIM_DATASTORE_CURRENT_H5]
    )

    manifest_a_after_restore = yaml.safe_load(
        manifest_path_a.read_text(encoding="utf-8")
    )
    assert {
        step: manifest_a_after_restore[step]["run_id"] for step in expected_steps
    } == run_ids_a
