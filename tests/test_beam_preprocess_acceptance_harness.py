"""Integration coverage for the HPC BEAM-preprocess acceptance entry point."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.beam.preprocessor import BeamPreprocessor
from pilates.config import load_config
from pilates.runtime import beam_preprocess_acceptance as acceptance
from pilates.workflows.artifact_keys import ATLAS_VEHICLES2_OUTPUT
from pilates.workflows.steps import beam as beam_steps
from pilates.workflows.steps.shared import BeamPreprocessOutputs
from workflow_state import WorkflowState


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ACCEPTANCE_SETTINGS = (
    _PROJECT_ROOT
    / "scenarios/sfbay/settings-sfbay-consist-beam-preprocess-hpc-2019-acceptance.yaml"
)


def test_acceptance_settings_name_the_workflow_and_forecast_years_separately() -> None:
    """The reviewed 2019 ATLAS artifact belongs to the 2017→2019 interval."""

    state = WorkflowState.from_settings(load_config(str(_ACCEPTANCE_SETTINGS)))

    assert state.year == 2017
    assert state.forecast_year == 2019
    assert state.current_inner_iter == 0


def test_acceptance_state_rejects_a_correct_workflow_year_with_wrong_forecast_year() -> (
    None
):
    """The 2017 workflow state must not silently consume a 2019 artifact for 2021."""

    state = SimpleNamespace(year=2017, forecast_year=2021, current_inner_iter=0)

    with pytest.raises(
        ValueError,
        match="workflow_year=2017, forecast_year=2019, iteration=0",
    ):
        acceptance._validate_acceptance_state(state)


def _acceptance_inputs(tmp_path: Path) -> tuple[Path, Path]:
    settings = load_config(str(_ACCEPTANCE_SETTINGS))
    evidence_root = tmp_path / "evidence"
    manifest = tmp_path / "submitted-input-manifest.json"
    source_root = tmp_path / "sources"
    beam_root = source_root / "beam-tree"
    config_path = beam_root / settings.beam.config
    config_path.parent.mkdir(parents=True)
    config_path.write_text("beam {}\n", encoding="utf-8")
    common_root = source_root / "common"
    common_root.mkdir(parents=True)
    for name in ("akka.conf", "metrics.conf", "matsim.conf"):
        (common_root / name).write_text("config {}\n", encoding="utf-8")
    inputs: dict[str, str] = {}
    for key, name in (
        ("plans_beam_in", "plans.parquet"),
        ("households_beam_in", "households.parquet"),
        ("persons_beam_in", "persons.parquet"),
        (ATLAS_VEHICLES2_OUTPUT, "vehicles2_2019.csv"),
    ):
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{key}\n", encoding="utf-8")
        inputs[key] = str(path)
    manifest.write_text(
        json.dumps(
            {
                "beam_input_root": str(beam_root),
                "inputs": inputs,
                "cohort": {
                    "workflow_year": 2017,
                    "forecast_year": 2019,
                    "iteration": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, evidence_root


def test_stage_beam_input_tree_preserves_sibling_common_configuration(
    tmp_path: Path,
) -> None:
    """A staged SFBay tree keeps the shared HOCON include root reachable."""

    production_root = tmp_path / "production"
    source_root = production_root / "sfbay"
    (source_root / "scenarios").mkdir(parents=True)
    (source_root / "scenarios" / "acceptance.conf").write_text(
        'include "../common/akka.conf"\n', encoding="utf-8"
    )
    common_root = production_root / "common"
    for name in ("akka.conf", "metrics.conf", "matsim.conf"):
        (common_root / name).parent.mkdir(parents=True, exist_ok=True)
        (common_root / name).write_text("config {}\n", encoding="utf-8")
    mutable_root = tmp_path / "workspace" / "beam" / "input"
    workspace = SimpleNamespace(get_beam_mutable_data_dir=lambda: str(mutable_root))
    settings = SimpleNamespace(run=SimpleNamespace(region="sfbay"))

    acceptance._stage_beam_input_tree(
        source_root=source_root,
        settings=settings,
        workspace=workspace,
    )

    assert (mutable_root / "sfbay" / "scenarios" / "acceptance.conf").is_file()
    assert (mutable_root / "common" / "akka.conf").is_file()
    assert (mutable_root / "common" / "metrics.conf").is_file()
    assert (mutable_root / "common" / "matsim.conf").is_file()


def test_stage_beam_input_tree_rejects_missing_sibling_common_configuration(
    tmp_path: Path,
) -> None:
    """The harness fails before hashing when a submitted BEAM tree is incomplete."""

    source_root = tmp_path / "production" / "sfbay"
    source_root.mkdir(parents=True)
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "workspace" / "beam" / "input")
    )
    settings = SimpleNamespace(run=SimpleNamespace(region="sfbay"))

    with pytest.raises(ValueError, match="shared common configuration"):
        acceptance._stage_beam_input_tree(
            source_root=source_root,
            settings=settings,
            workspace=workspace,
        )


def test_main_runs_real_native_step_cold_then_fresh_and_retains_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A real state/adapter/Tracker run hydrates a distinct fresh workspace."""

    manifest, evidence_root = _acceptance_inputs(tmp_path)
    monkeypatch.delenv("PILATES_DISABLE_BEAM_CONFIG_ADAPTER", raising=False)
    monkeypatch.setattr(beam_steps, "enqueue_archive_copy", lambda **_kwargs: None)
    calls: list[Path] = []

    def fake_preprocess(
        self: BeamPreprocessor,
        workspace: object,
        *,
        beam_preprocess_inputs: dict[str, Path],
        beam_preprocess_context: object,
    ) -> BeamPreprocessOutputs:
        del self, beam_preprocess_context
        mutable = Path(workspace.get_beam_mutable_data_dir())
        calls.append(mutable)
        prepared = {}
        for key, source in beam_preprocess_inputs.items():
            destination = mutable / "prepared" / Path(source).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                Path(source).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            prepared[key] = destination
        return BeamPreprocessOutputs(
            beam_mutable_data_dir=mutable,
            prepared_inputs=prepared,
        )

    monkeypatch.setattr(BeamPreprocessor, "preprocess", fake_preprocess)
    assert (
        acceptance.main(
            [
                "--settings",
                str(_ACCEPTANCE_SETTINGS),
                "--manifest",
                str(manifest),
                "--evidence-root",
                str(evidence_root),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    cold = json.loads((evidence_root / "phases" / "cold.json").read_text())
    fresh = json.loads((evidence_root / "phases" / "fresh.json").read_text())
    verdict = json.loads((evidence_root / "semantic-validation.json").read_text())
    assert cold["cache_hit"] is False and fresh["cache_hit"] is True
    assert cold["cohort"] == {
        "workflow_year": 2017,
        "forecast_year": 2019,
        "iteration": 0,
    }
    assert fresh["body_executions_after"] == cold["body_executions_after"]
    assert ATLAS_VEHICLES2_OUTPUT in cold["selected_roles"]
    assert cold["persisted_run"]["binding_kind"] == "ordinary-binding"
    assert cold["persisted_run"]["cache_outcome"] == "miss"
    assert fresh["persisted_run"]["cache_outcome"] == "hit"
    assert cold["requested_run_id"] != fresh["requested_run_id"]
    assert fresh["source_run_id"] == cold["requested_run_id"]
    assert fresh["persisted_run"]["execution_run_id"] == cold["requested_run_id"]
    assert fresh["persisted_run"]["source_run_id"] == cold["requested_run_id"]
    assert cold["persisted_run"]["identity"]["input_identity"]["mode"] == "action-v2"
    assert cold["persisted_run"]["identity"]["config_adapter"]["name"] == "beam"
    assert cold["persisted_run"]["identity"]["config_adapter"]["hash"]
    assert (
        cold["persisted_run"]["artifacts"]["action_inputs"]
        == fresh["persisted_run"]["artifacts"]["action_inputs"]
    )
    assert (
        cold["persisted_run"]["artifacts"]["outputs"]
        == fresh["persisted_run"]["artifacts"]["outputs"]
    )
    assert (
        cold["persisted_run"]["requested_input_staging"]["normalized_input_paths"]
        == fresh["persisted_run"]["requested_input_staging"]["normalized_input_paths"]
    )
    assert (evidence_root / "persisted-runs" / "cold.json").is_file()
    assert (evidence_root / "persisted-runs" / "fresh.json").is_file()
    fresh_snapshot = json.loads(
        (evidence_root / "persisted-runs" / "fresh.json").read_text(encoding="utf-8")
    )
    assert fresh_snapshot["run"]["id"] == fresh["requested_run_id"]
    assert fresh_snapshot["run"]["meta"]["cache_source"] == cold["requested_run_id"]
    assert verdict["persisted_cache_relationship_valid"] is True
    assert verdict["persisted_identity_valid"] is True
    assert verdict["persisted_artifact_links_valid"] is True
    assert verdict["persisted_staging_valid"] is True
    assert verdict["fresh_hydration_destinations_valid"] is True
    assert verdict["valid"] is True
    assert (
        json.loads((evidence_root / "effective-input-manifest.json").read_text())[
            "cohort"
        ]["forecast_year"]
        == 2019
    )
    progress = capsys.readouterr().out
    expected_milestones = (
        "acceptance driver started",
        "acceptance manifest validated",
        "acceptance settings and state validated",
        "acceptance tracker created",
        "acceptance input artifacts logged",
        "cold acceptance phase started",
        "cold acceptance phase completed",
        "fresh acceptance phase started",
        "fresh acceptance phase completed",
        "acceptance semantic validation completed",
    )
    assert [progress.index(milestone) for milestone in expected_milestones] == sorted(
        progress.index(milestone) for milestone in expected_milestones
    )


@pytest.mark.parametrize(
    ("cold_hit", "fresh_hit", "expected_message", "expected_calls"),
    [
        (
            True,
            True,
            "cold beam_preprocess acceptance invocation unexpectedly hit cache",
            1,
        ),
        (False, False, "fresh beam_preprocess acceptance invocation missed cache", 2),
    ],
)
def test_main_fails_closed_on_unexpected_cache_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cold_hit: bool,
    fresh_hit: bool,
    expected_message: str,
    expected_calls: int,
) -> None:
    """The driver stops at the first invalid cold/fresh cache selector outcome."""

    manifest, evidence_root = _acceptance_inputs(tmp_path)
    cache_outcomes = iter((cold_hit, fresh_hit))
    calls: list[str] = []

    def fake_run_phase(*, phase: str, **_kwargs: object) -> acceptance.PhaseExecution:
        calls.append(phase)
        cache_hit = next(cache_outcomes)
        return acceptance.PhaseExecution(
            cache_hit=cache_hit,
            run_id=f"{phase}-run",
            source_run_id="cold-run" if cache_hit else None,
            declared_outputs={},
            selected_roles={},
            source_bindings={},
            input_identities={},
            config_identity="config",
            snapshot_reference=str(evidence_root / "provenance.duckdb"),
            requested_run_id=f"{phase}-run",
        )

    monkeypatch.setattr(acceptance, "run_phase", fake_run_phase)
    with pytest.raises(RuntimeError, match=expected_message):
        acceptance.main(
            [
                "--settings",
                str(_ACCEPTANCE_SETTINGS),
                "--manifest",
                str(manifest),
                "--evidence-root",
                str(evidence_root),
            ]
        )
    assert calls == ["cold", "fresh"][:expected_calls]


def test_semantic_validation_rejects_symmetric_missing_outputs() -> None:
    """Equal missing output maps are not accepted as equivalent products."""

    persisted_run = {
        "requested_run_id": "cold-run",
        "execution_run_id": "cold-run",
        "source_run_id": None,
        "cache_outcome": "miss",
        "binding_kind": "ordinary-binding",
        "identity": {},
        "artifacts": {},
        "requested_input_staging": {"normalized_input_paths": {}},
        "materialized_outputs": {"normalized_paths": {}},
    }
    phase = {
        "selected_roles": {},
        "source_bindings": {},
        "input_identities": {},
        "config_identity": "same",
        "workspace_root": "cold",
        "declared_outputs": {"plans": {"present": False, "type": "missing"}},
        "body_executions_before": 0,
        "body_executions_after": 1,
        "persisted_run": persisted_run,
    }
    fresh = {
        **phase,
        "workspace_root": "fresh",
        "body_executions_before": 1,
        "body_executions_after": 1,
        "persisted_run": {
            **persisted_run,
            "requested_run_id": "fresh-run",
            "execution_run_id": "cold-run",
            "source_run_id": "cold-run",
            "cache_outcome": "hit",
        },
    }
    verdict = acceptance._validate(phase, fresh)
    assert verdict["valid"] is False
    assert verdict["declared_outputs_present"] is False
