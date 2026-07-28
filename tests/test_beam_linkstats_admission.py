from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.beam.launch_paths import BeamLaunchPathReference


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
            expected_run_id="baseline-run",
            artifact_key="linkstats_warmstart",
            expected_bytes_path=None,
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
        settings=_settings("warn"),
        launch_reference=_reference(staged),
        report_dir=tmp_path / "run-a",
    )

    assert report.outcome == "unverified"
    assert (tmp_path / "run-a" / "admission" / "beam-linkstats-warmstart.json").exists()


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
