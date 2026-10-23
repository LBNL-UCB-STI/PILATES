from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.beam.launch_paths import BeamLaunchPathReference
from pilates.config.models import PriorRunExpectation


def _reference(path: Path) -> BeamLaunchPathReference:
    return BeamLaunchPathReference(
        config_key="beam.warmStart.initialLinkstatsFilePath",
        raw_value='${beam.inputDirectory}"/_pilates/linkstats/warmstart.csv.gz"',
        canonical_value=str(path),
        configured_path=path,
        execution_path=path,
        physical_target_path=path.resolve(),
        container_path="/app/input/seattle/_pilates/linkstats/warmstart.csv.gz",
    )


class _Tracker:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.meta: dict[str, object] = {}

    def log_meta(self, **kwargs: object) -> None:
        self.meta.update(kwargs)


def _settings(mode: str | None) -> SimpleNamespace:
    linkstats = (
        None
        if mode is None
        else SimpleNamespace(
            mode=mode,
            expectation=PriorRunExpectation(
                expected_run_id="baseline-run",
                artifact_key="linkstats_warmstart",
                expected_bytes_path=None,
            ),
        )
    )
    return SimpleNamespace(
        beam=SimpleNamespace(admission=SimpleNamespace(linkstats=linkstats))
    )


def test_linkstats_admission_config_returns_the_configured_entry() -> None:
    from pilates.beam.admission import _linkstats_admission_config

    settings = _settings("strict")

    assert _linkstats_admission_config(settings) is settings.beam.admission.linkstats


def test_strict_linkstats_admission_writes_report_before_rejecting(
    monkeypatch, tmp_path: Path
) -> None:
    from pilates.beam import admission

    staged = tmp_path / "beam" / "input" / "seattle" / "warmstart.csv.gz"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"observed")
    tracker = _Tracker(tmp_path)
    captured: dict[str, object] = {}

    def _check(received_tracker, *, reference, expected_run_id, expected_bytes_path):
        captured.update(
            tracker=received_tracker,
            reference=reference,
            expected_run_id=expected_run_id,
            expected_bytes_path=expected_bytes_path,
        )
        return SimpleNamespace(
            outcome="mismatched",
            canonical_json=lambda: '{"outcome":"mismatched"}',
        )

    monkeypatch.setattr(admission.consist, "check_admission_reference", _check)

    with pytest.raises(admission.BeamInputAdmissionError, match="mismatched"):
        admission.preflight_staged_linkstats_admission(
            tracker=tracker,
            metadata_logger=tracker,
            existing_admission_reports={},
            settings=_settings("strict"),
            launch_reference=_reference(staged),
            report_dir=tmp_path / "run-a",
        )

    assert captured["tracker"] is tracker
    runtime_reference = captured["reference"]
    assert runtime_reference.execution_path == staged
    assert runtime_reference.consumer_path.startswith("/app/input/")
    report_path = tmp_path / "run-a" / "admission" / "beam-linkstats-warmstart.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "outcome": "mismatched"
    }
    assert tracker.meta["admission_reports"] == {
        "beam_linkstats_warmstart": {"outcome": "mismatched"}
    }


def test_warn_linkstats_admission_records_and_continues(
    monkeypatch, tmp_path: Path
) -> None:
    from pilates.beam import admission

    staged = tmp_path / "warmstart.csv.gz"
    staged.write_bytes(b"observed")
    tracker = _Tracker(tmp_path)
    monkeypatch.setattr(
        admission.consist,
        "check_admission_reference",
        lambda *_args, **_kwargs: SimpleNamespace(
            outcome="unverified",
            canonical_json=lambda: '{"outcome":"unverified"}',
        ),
    )

    report = admission.preflight_staged_linkstats_admission(
        tracker=tracker,
        metadata_logger=tracker,
        existing_admission_reports={},
        settings=_settings("warn"),
        launch_reference=_reference(staged),
        report_dir=tmp_path / "run-a",
    )

    assert report.outcome == "unverified"
    assert (tmp_path / "run-a" / "admission" / "beam-linkstats-warmstart.json").exists()


def test_linkstats_admission_merges_existing_child_run_reports(
    monkeypatch, tmp_path: Path
) -> None:
    from pilates.beam import admission

    staged = tmp_path / "warmstart.csv.gz"
    staged.write_bytes(b"observed")
    tracker = _Tracker(tmp_path)
    metadata_logger = _Tracker(tmp_path)
    monkeypatch.setattr(
        admission.consist,
        "check_admission_reference",
        lambda *_args, **_kwargs: SimpleNamespace(
            outcome="verified",
            canonical_json=lambda: '{"outcome":"verified"}',
        ),
    )

    admission.preflight_staged_linkstats_admission(
        tracker=tracker,
        metadata_logger=metadata_logger,
        existing_admission_reports={"other_input": {"outcome": "verified"}},
        settings=_settings("warn"),
        launch_reference=_reference(staged),
        report_dir=tmp_path / "run-a",
    )

    assert metadata_logger.meta["admission_reports"] == {
        "other_input": {"outcome": "verified"},
        "beam_linkstats_warmstart": {"outcome": "verified"},
    }


def test_linkstats_admission_is_disabled_without_an_expectation(
    monkeypatch, tmp_path: Path
) -> None:
    from pilates.beam import admission

    staged = tmp_path / "warmstart.csv.gz"
    staged.write_bytes(b"observed")
    monkeypatch.setattr(
        admission.consist,
        "check_admission_reference",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not check")),
    )

    report = admission.preflight_staged_linkstats_admission(
        tracker=_Tracker(tmp_path),
        metadata_logger=_Tracker(tmp_path),
        existing_admission_reports={},
        settings=_settings(None),
        launch_reference=_reference(staged),
        report_dir=tmp_path / "run-a",
    )

    assert report is None


def test_linkstats_admission_reports_are_isolated_by_run_directory(
    monkeypatch, tmp_path: Path
) -> None:
    from pilates.beam import admission

    staged = tmp_path / "warmstart.csv.gz"
    staged.write_bytes(b"observed")
    monkeypatch.setattr(
        admission.consist,
        "check_admission_reference",
        lambda *_args, **_kwargs: SimpleNamespace(
            outcome="verified",
            canonical_json=lambda: '{"outcome":"verified"}',
        ),
    )
    tracker = _Tracker(tmp_path / "tracker-root")

    for run_name in ("run-a", "run-b"):
        admission.preflight_staged_linkstats_admission(
            tracker=tracker,
            metadata_logger=tracker,
            existing_admission_reports={},
            settings=_settings("warn"),
            launch_reference=_reference(staged),
            report_dir=tmp_path / "outputs" / run_name,
        )

    assert (
        tmp_path / "outputs" / "run-a" / "admission" / "beam-linkstats-warmstart.json"
    ).exists()
    assert (
        tmp_path / "outputs" / "run-b" / "admission" / "beam-linkstats-warmstart.json"
    ).exists()
    assert not (tmp_path / "tracker-root" / "admission").exists()


def test_declared_digest_linkstats_admission_verifies_and_reads_sidecar(
    tmp_path: Path,
) -> None:
    import hashlib
    from pilates.beam import admission
    from pilates.config.models import BeamLinkstatsAdmissionConfig

    staged = tmp_path / "warmstart.parquet"
    content = b"PAR1-converged-linkstats-bytes"
    staged.write_bytes(content)
    digest = "sha256:file:" + hashlib.sha256(content).hexdigest()

    sidecar = tmp_path / "warmstart.parquet.provenance.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "pilates.linkstats_provenance.v1",
                "artifact_name": "init.linkstats.parquet",
                "format": "parquet",
                "identity": digest,
                "size_bytes": len(content),
                "source": {
                    "beam_run_label": "beam-fc07-v260419-it3",
                    "source_uri": "https://drive.google.com/file/d/10ubIO7oamqNbzvRlDfIWcQ8xSWcee9s5/view?usp=share_link",
                },
            }
        ),
        encoding="utf-8",
    )

    linkstats_cfg = BeamLinkstatsAdmissionConfig(
        mode="strict",
        expectation={
            "kind": "declared_digest",
            "identity": digest,
            "source_label": "beam-fc07-v260419-it3",
            "source_uri": "https://drive.google.com/file/d/10ubIO7oamqNbzvRlDfIWcQ8xSWcee9s5/view?usp=share_link",
        },
    )
    settings = SimpleNamespace(
        beam=SimpleNamespace(admission=SimpleNamespace(linkstats=linkstats_cfg))
    )
    tracker = _Tracker(tmp_path)

    report = admission.preflight_staged_linkstats_admission(
        tracker=tracker,
        metadata_logger=tracker,
        existing_admission_reports={},
        settings=settings,
        launch_reference=_reference(staged),
        report_dir=tmp_path / "run-declared",
    )

    assert report is not None
    assert report.outcome == "verified"
    report_path = (
        tmp_path / "run-declared" / "admission" / "beam-linkstats-warmstart.json"
    )
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["outcome"] == "verified"
    assert (
        saved["sidecar_provenance"]["source"]["beam_run_label"]
        == "beam-fc07-v260419-it3"
    )


def test_declared_digest_linkstats_admission_rejects_mismatched_sidecar(
    tmp_path: Path,
) -> None:
    import hashlib
    from pilates.beam import admission
    from pilates.config.models import BeamLinkstatsAdmissionConfig

    staged = tmp_path / "warmstart.parquet"
    content = b"PAR1-converged-linkstats-bytes"
    staged.write_bytes(content)
    digest = "sha256:file:" + hashlib.sha256(content).hexdigest()

    sidecar = tmp_path / "warmstart.parquet.provenance.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "pilates.linkstats_provenance.v1",
                "artifact_name": "init.linkstats.parquet",
                "format": "parquet",
                "identity": "sha256:file:" + ("0" * 64),
                "size_bytes": len(content),
            }
        ),
        encoding="utf-8",
    )

    linkstats_cfg = BeamLinkstatsAdmissionConfig(
        mode="strict",
        expectation={
            "kind": "declared_digest",
            "identity": digest,
        },
    )
    settings = SimpleNamespace(
        beam=SimpleNamespace(admission=SimpleNamespace(linkstats=linkstats_cfg))
    )
    tracker = _Tracker(tmp_path)

    with pytest.raises(admission.BeamInputAdmissionError, match="mismatched"):
        admission.preflight_staged_linkstats_admission(
            tracker=tracker,
            metadata_logger=tracker,
            existing_admission_reports={},
            settings=settings,
            launch_reference=_reference(staged),
            report_dir=tmp_path / "run-declared-bad",
        )


def test_declared_digest_linkstats_admission_flags_unreadable_colocated_sidecar(
    tmp_path: Path,
) -> None:
    import hashlib
    from pilates.beam import admission
    from pilates.config.models import BeamLinkstatsAdmissionConfig

    staged = tmp_path / "warmstart.parquet"
    content = b"PAR1-converged-linkstats-bytes"
    staged.write_bytes(content)
    digest = "sha256:file:" + hashlib.sha256(content).hexdigest()

    sidecar = tmp_path / "warmstart.parquet.provenance.json"
    sidecar.write_text("{corrupted-json", encoding="utf-8")

    linkstats_cfg = BeamLinkstatsAdmissionConfig(
        mode="warn",
        expectation={
            "kind": "declared_digest",
            "identity": digest,
        },
    )
    settings = SimpleNamespace(
        beam=SimpleNamespace(admission=SimpleNamespace(linkstats=linkstats_cfg))
    )
    tracker = _Tracker(tmp_path)

    report = admission.preflight_staged_linkstats_admission(
        tracker=tracker,
        metadata_logger=tracker,
        existing_admission_reports={},
        settings=settings,
        launch_reference=_reference(staged),
        report_dir=tmp_path / "run-declared-corrupt-sidecar",
    )
    assert report is not None
    assert report.outcome == "unverified"
    assert "sidecar_unreadable" in report.observations


def test_declared_digest_linkstats_admission_preserves_missing_outcome_with_corrupt_sidecar(
    tmp_path: Path,
) -> None:
    import hashlib
    from pilates.beam import admission
    from pilates.config.models import BeamLinkstatsAdmissionConfig

    staged = tmp_path / "missing_warmstart.parquet"
    digest = "sha256:file:" + hashlib.sha256(b"bytes").hexdigest()

    sidecar = tmp_path / "missing_warmstart.parquet.provenance.json"
    sidecar.write_text("{corrupted-json", encoding="utf-8")

    linkstats_cfg = BeamLinkstatsAdmissionConfig(
        mode="warn",
        expectation={
            "kind": "declared_digest",
            "identity": digest,
        },
    )
    settings = SimpleNamespace(
        beam=SimpleNamespace(admission=SimpleNamespace(linkstats=linkstats_cfg))
    )
    tracker = _Tracker(tmp_path)

    report = admission.preflight_staged_linkstats_admission(
        tracker=tracker,
        metadata_logger=tracker,
        existing_admission_reports={},
        settings=settings,
        launch_reference=_reference(staged),
        report_dir=tmp_path / "run-declared-missing-file",
    )
    assert report is not None
    assert report.outcome == "unreadable"
    assert report.observations == ("file_unreadable", "sidecar_unreadable")


def test_beam_identity_config_uses_portable_admission_fingerprint(
    tmp_path: Path,
) -> None:
    from pilates.config.models import BeamLinkstatsAdmissionConfig
    from pilates.utils.consist_config import build_beam_identity_config

    cfg_host_a = BeamLinkstatsAdmissionConfig(
        mode="strict",
        expectation={
            "kind": "prior_run",
            "expected_run_id": "run_1",
            "artifact_key": "linkstats",
            "expected_direction": "output",
            "expected_bytes_path": str(tmp_path / "host_a" / "linkstats.parquet"),
        },
    )
    cfg_host_b = BeamLinkstatsAdmissionConfig(
        mode="strict",
        expectation={
            "kind": "prior_run",
            "expected_run_id": "run_1",
            "artifact_key": "linkstats",
            "expected_direction": "output",
            "expected_bytes_path": str(tmp_path / "host_b" / "linkstats.parquet"),
        },
    )

    def _make_settings(linkstats_cfg: BeamLinkstatsAdmissionConfig) -> SimpleNamespace:
        return SimpleNamespace(
            beam=SimpleNamespace(
                config="sfbay.conf",
                sample=0.1,
                replanning_portion=0.4,
                memory="16g",
                discard_plans_every_year=False,
                max_plans_memory=0,
                router_directory="r5",
                scenario_folder="sfbay",
                admission=SimpleNamespace(linkstats=linkstats_cfg),
            ),
            defs=SimpleNamespace(),
        )

    ident_a = build_beam_identity_config(_make_settings(cfg_host_a))
    ident_b = build_beam_identity_config(_make_settings(cfg_host_b))
    assert ident_a["linkstats_admission"] == ident_b["linkstats_admission"]
    assert "expected_bytes_path" not in ident_a["linkstats_admission"]
    assert ident_a["linkstats_admission"]["direction"] == "output"
