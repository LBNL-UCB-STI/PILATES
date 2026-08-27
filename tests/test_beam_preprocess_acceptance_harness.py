"""Entry-point coverage for the HPC BEAM-preprocess acceptance harness."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


def test_main_records_cold_and_fresh_semantic_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The real CLI assembles both phases and preserves reviewer-readable evidence."""

    from pilates.runtime import beam_preprocess_acceptance as acceptance

    settings = tmp_path / "settings.yaml"
    settings.write_text("run: {}\n", encoding="utf-8")
    beam_input_root = tmp_path / "beam-input"
    beam_input_root.mkdir()
    (beam_input_root / "beam.conf").write_text("beam {}\n", encoding="utf-8")
    inputs = {}
    for key in ("plans_beam_in", "households_beam_in", "persons_beam_in"):
        path = tmp_path / f"{key}.parquet"
        path.write_text(f"{key}\n", encoding="utf-8")
        inputs[key] = str(path)
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps({"beam_input_root": str(beam_input_root), "inputs": inputs}),
        encoding="utf-8",
    )

    calls: list[tuple[str, Path]] = []

    def fake_run_phase(*, phase, workspace_root, **_kwargs):
        calls.append((phase, workspace_root))
        output = workspace_root / ".pilates-consist-outputs" / "beam_preprocess" / "plans_beam_in.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("plans\n", encoding="utf-8")
        return acceptance.PhaseExecution(
            cache_hit=phase == "fresh",
            run_id=f"{phase}-run",
            source_run_id="cold-run" if phase == "fresh" else None,
            declared_outputs={"plans_beam_in": output},
            selected_roles={"plans_beam_in": "plans_beam_in"},
            source_bindings={"plans_beam_in": "seed-input"},
            input_identities={"plans_beam_in": "input-sha"},
            config_identity="config-sha",
            snapshot_reference="provenance.duckdb",
        )

    monkeypatch.setattr(acceptance, "run_phase", fake_run_phase)
    fake_tracker = SimpleNamespace(
        start_run=lambda *_args: nullcontext(),
        log_artifact=lambda path, **_kwargs: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(acceptance, "load_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(
        acceptance.WorkflowState,
        "from_settings",
        lambda _settings: SimpleNamespace(),
    )
    monkeypatch.setattr(acceptance.cr, "create_tracker", lambda **_kwargs: fake_tracker)

    evidence_root = tmp_path / "evidence"
    assert acceptance.main(
        [
            "--settings",
            str(settings),
            "--manifest",
            str(manifest),
            "--evidence-root",
            str(evidence_root),
        ]
    ) == 0

    assert [phase for phase, _ in calls] == ["cold", "fresh"]
    assert calls[0][1] != calls[1][1]
    cold = json.loads((evidence_root / "phases" / "cold.json").read_text())
    fresh = json.loads((evidence_root / "phases" / "fresh.json").read_text())
    verdict = json.loads((evidence_root / "semantic-validation.json").read_text())
    assert cold["cache_hit"] is False
    assert fresh["cache_hit"] is True
    assert fresh["source_run_id"] == "cold-run"
    assert verdict["valid"] is True
    assert verdict["expected_workspace_differences"]
