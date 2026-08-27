"""Integration coverage for the HPC BEAM-preprocess acceptance entry point."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.beam.preprocessor import BeamPreprocessor
from pilates.runtime import beam_preprocess_acceptance as acceptance
from pilates.workflows.artifact_keys import ATLAS_VEHICLES2_OUTPUT
from pilates.workflows.steps import beam as beam_steps
from pilates.workflows.steps.shared import BeamPreprocessOutputs


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="sfbay", models=SimpleNamespace(traffic_assignment=None, travel=None), consist_hashing_strategy="full"),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="", skim_zone_geoid_col="TAZ", admission=None, local_mutable_data_folder="beam-input", local_output_folder="beam-output"),
        activitysim=SimpleNamespace(file_format="parquet"),
        shared=SimpleNamespace(geography=SimpleNamespace(zones=None)),
        vehicle_ownership_model_enabled=True,
    )


def test_main_runs_real_native_step_cold_then_fresh_and_retains_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real Tracker/resolver/execute_step run hydrates a distinct fresh workspace."""

    settings = _settings()
    evidence_root = tmp_path / "evidence"
    manifest = evidence_root / "submitted-input-manifest.json"
    manifest.parent.mkdir()
    source_root = tmp_path / "sources"
    beam_root = source_root / "beam-tree"
    beam_root.mkdir(parents=True)
    (beam_root / "beam.conf").write_text("beam {}\n", encoding="utf-8")
    inputs: dict[str, str] = {}
    for key, name in (("plans_beam_in", "plans.parquet"), ("households_beam_in", "households.parquet"), ("persons_beam_in", "persons.parquet"), (ATLAS_VEHICLES2_OUTPUT, "vehicles2_2019.csv.gz")):
        path = source_root / name
        path.write_text(f"{key}\n", encoding="utf-8")
        inputs[key] = str(path)
    manifest.write_text(json.dumps({"beam_input_root": str(beam_root), "inputs": inputs, "cohort": {"year": 2019, "iteration": 0}}), encoding="utf-8")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("run: {}\n", encoding="utf-8")
    monkeypatch.setattr(acceptance, "load_config", lambda _path: settings)
    monkeypatch.setattr(acceptance.WorkflowState, "from_settings", lambda _settings: SimpleNamespace(year=2019, current_inner_iter=0, full_settings=settings))
    monkeypatch.setattr(beam_steps, "build_enabled_workflow_surface", lambda _settings: SimpleNamespace(profile=SimpleNamespace(vehicle_ownership_model_enabled=True)))
    monkeypatch.setenv("PILATES_DISABLE_BEAM_CONFIG_ADAPTER", "1")
    monkeypatch.setattr(beam_steps, "enqueue_archive_copy", lambda **_kwargs: None)
    monkeypatch.setattr("pilates.workflows.step_consist_meta.build_step_consist_kwargs", lambda **_kwargs: {"config": {"model": "beam"}, "identity_inputs": []})
    calls: list[Path] = []

    def fake_preprocess(self, workspace, *, beam_preprocess_inputs, beam_preprocess_context):
        del self, beam_preprocess_context
        mutable = Path(workspace.get_beam_mutable_data_dir())
        calls.append(mutable)
        prepared = {}
        for key, source in beam_preprocess_inputs.items():
            destination = mutable / "prepared" / Path(source).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
            prepared[key] = destination
        return BeamPreprocessOutputs(beam_mutable_data_dir=mutable, prepared_inputs=prepared)

    monkeypatch.setattr(BeamPreprocessor, "preprocess", fake_preprocess)
    assert acceptance.main(["--settings", str(settings_path), "--manifest", str(manifest), "--evidence-root", str(evidence_root)]) == 0
    assert len(calls) == 1
    cold = json.loads((evidence_root / "phases" / "cold.json").read_text())
    fresh = json.loads((evidence_root / "phases" / "fresh.json").read_text())
    verdict = json.loads((evidence_root / "semantic-validation.json").read_text())
    assert cold["cache_hit"] is False and fresh["cache_hit"] is True
    assert cold["cohort"] == {"year": 2019, "iteration": 0}
    assert fresh["body_executions_after"] == cold["body_executions_after"]
    assert ATLAS_VEHICLES2_OUTPUT in cold["selected_roles"]
    assert verdict["valid"] is True
    assert json.loads((evidence_root / "effective-input-manifest.json").read_text())["cohort"]["year"] == 2019


def test_semantic_validation_rejects_symmetric_missing_outputs() -> None:
    """Equal missing output maps are not accepted as equivalent products."""

    phase = {"selected_roles": {}, "source_bindings": {}, "input_identities": {}, "config_identity": "same", "workspace_root": "cold", "declared_outputs": {"plans": {"present": False, "type": "missing"}}, "body_executions_before": 0, "body_executions_after": 1}
    fresh = {**phase, "workspace_root": "fresh", "body_executions_before": 1, "body_executions_after": 1}
    verdict = acceptance._validate(phase, fresh)
    assert verdict["valid"] is False
    assert verdict["declared_outputs_present"] is False
