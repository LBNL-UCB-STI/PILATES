"""ResolvedBinding V1 bridge contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from consist import AdmissionEvidence, ResolvedBinding, Tracker, define_step
from consist.core.resolved_binding import ArtifactIdentity

from pilates.workflows.binding import build_resolved_binding


def _tracked_artifact(tmp_path: Path):
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    source = tmp_path / "payload.txt"
    source.write_text("selected\n", encoding="utf-8")
    with tracker.start_run("seed", "test"):
        return tracker.log_artifact(source, key="payload", direction="input")


def _step_identity(step_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"{step_name}__y2030__i1__phase_test",
        step_contract_identity="sha256:step-v1:" + "0" * 64,
    )


def test_build_resolved_binding_freezes_tracked_named_inputs_without_diagnostics_in_identity(
    tmp_path: Path,
) -> None:
    artifact = _tracked_artifact(tmp_path)

    @define_step(model="example")
    def consume(payload: Path, *, settings: object) -> None:
        del payload, settings

    first = build_resolved_binding(
        step_name="consume",
        function=consume,
        selected_artifacts={"payload": artifact},
        logical_destinations={"payload": Path("inputs/payload.txt")},
        selection_diagnostics={"selection": "preferred"},
        step_identity=_step_identity("consume"),
    )
    second = build_resolved_binding(
        step_name="consume",
        function=consume,
        selected_artifacts={"payload": artifact},
        logical_destinations={"payload": Path("inputs/payload.txt")},
        selection_diagnostics={"selection": "fallback"},
        step_identity=_step_identity("consume"),
    )

    assert isinstance(first, ResolvedBinding)
    assert first.identity_json() == second.identity_json()
    assert first.identity_digest() == second.identity_digest()
    assert first.evidence_json() != second.evidence_json()


def test_build_resolved_binding_rejects_invalid_v1_inputs(tmp_path: Path) -> None:
    artifact = _tracked_artifact(tmp_path)

    def consume(payload: Path) -> None:
        del payload

    with pytest.raises(ValueError, match="relative"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"payload": artifact},
            logical_destinations={"payload": tmp_path / "absolute.txt"},
            selection_diagnostics={},
            step_identity=_step_identity("consume"),
        )
    with pytest.raises(TypeError, match="tracked Artifact"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"payload": tmp_path / "untracked.txt"},
            logical_destinations={"payload": Path("inputs/payload.txt")},
            selection_diagnostics={},
            step_identity=_step_identity("consume"),
        )
    with pytest.raises(ValueError, match="named callable parameter"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"missing": artifact},
            logical_destinations={"missing": Path("inputs/payload.txt")},
            selection_diagnostics={},
            step_identity=_step_identity("consume"),
        )

    def consume_pair(first: Path, second: Path) -> None:
        del first, second

    with pytest.raises(ValueError, match="unique"):
        build_resolved_binding(
            step_name="consume_pair",
            function=consume_pair,
            selected_artifacts={"first": artifact, "second": artifact},
            logical_destinations={
                "first": Path("inputs/payload.txt"),
                "second": Path("inputs/payload.txt"),
            },
            selection_diagnostics={},
            step_identity=_step_identity("consume_pair"),
        )


def test_build_resolved_binding_rejects_admission_identity_mismatch(
    tmp_path: Path,
) -> None:
    artifact = _tracked_artifact(tmp_path)

    def consume(payload: Path) -> None:
        del payload

    with pytest.raises(ValueError, match="observed identity"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"payload": artifact},
            logical_destinations={"payload": Path("inputs/payload.txt")},
            selection_diagnostics={},
            step_identity=_step_identity("consume"),
            admission_evidence={
                "payload": AdmissionEvidence(
                    observed_identity=ArtifactIdentity.parse("sha256:file:" + "0" * 64),
                    expected_identity=None,
                    expected_source=None,
                )
            },
        )


def test_build_resolved_binding_requires_consist_preflight_identity(
    tmp_path: Path,
) -> None:
    artifact = _tracked_artifact(tmp_path)

    def consume(payload: Path) -> None:
        del payload

    with pytest.raises(TypeError, match="step_identity"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"payload": artifact},
            logical_destinations={"payload": Path("inputs/payload.txt")},
            selection_diagnostics={},
        )
