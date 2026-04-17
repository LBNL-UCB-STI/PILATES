from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.workflows.stages.postprocessing import run_postprocessing_stage
from tests.workflow_contract_harness import CouplerStub


def test_postprocessing_stage_persists_year_scoped_run_id_and_reruns_without_restorable_outputs(
    tmp_path,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace = SimpleNamespace(full_path=str(workspace_root))

    state = SimpleNamespace(year=2018, current_year=2018)
    settings = SimpleNamespace(run=SimpleNamespace(consist_code_identity=None))
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=SimpleNamespace(profile=SimpleNamespace()),
    )

    run_ids = {"value": 0}

    class ScenarioStub:
        def run(self, **_kwargs):
            run_ids["value"] += 1
            return SimpleNamespace(
                cache_hit=False,
                run=SimpleNamespace(id=f"postprocess-run-{run_ids['value']}"),
            )

    coupler = CouplerStub()

    run_postprocessing_stage(
        scenario=ScenarioStub(),
        coupler=coupler,
        year=2018,
        context=context,
    )

    manifest_path = Path(workspace_root) / ".workflow" / "postprocessing_year_2018.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    assert manifest.get("postprocessing", {}).get("run_id") == "postprocess-run-1"

    class ScenarioSecondRun:
        def run(self, **_kwargs):
            run_ids["value"] += 1
            return SimpleNamespace(
                cache_hit=False,
                run=SimpleNamespace(id=f"postprocess-run-{run_ids['value']}"),
            )

    run_postprocessing_stage(
        scenario=ScenarioSecondRun(),
        coupler=coupler,
        year=2018,
        context=context,
    )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    assert manifest.get("postprocessing", {}).get("run_id") == "postprocess-run-2"
