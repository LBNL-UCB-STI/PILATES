from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from consist.models.artifact import Artifact
from consist.models.run import Run

from pilates.runtime import run_publishers
from pilates.runtime.run_notifications import RunNotificationContext
from pilates.runtime.run_publishers import (
    GoogleSheetWebhookRunEventPublisher,
    LocalJsonlRunEventPublisher,
    RunEvent,
    RunEventPublisher,
    RunPublisherSettings,
    SheetWebhookPayload,
    SummaryHtmlRunEventPublisher,
    register_consist_run_publishers,
)


def _run(
    run_id: str,
    *,
    model_name: str = "demo",
    parent_run_id: str | None = None,
    tags: list[str] | None = None,
    year: int | None = None,
    iteration: int | None = None,
    stage: str | None = None,
    phase: str | None = None,
    meta: dict[str, object] | None = None,
) -> Run:
    return Run(
        id=run_id,
        model_name=model_name,
        parent_run_id=parent_run_id,
        tags=tags or [],
        year=year,
        iteration=iteration,
        stage=stage,
        phase=phase,
        meta=meta or {},
    )


def _artifact(key: str) -> Artifact:
    return Artifact(key=key, container_uri=f"./{key}.csv", driver="csv")


class FakePublisher(RunEventPublisher):
    name = "fake"

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def publish(self, event: RunEvent) -> None:
        self.events.append(event)


class FakeTracker:
    def __init__(self) -> None:
        self.start_hooks: list[Callable[[Run], None]] = []
        self.complete_hooks: list[Callable[[Run, list[Artifact]], None]] = []
        self.failed_hooks: list[Callable[[Run, Exception], None]] = []

    def on_run_start(self, callback: Callable[[Run], None]) -> Callable[[Run], None]:
        self.start_hooks.append(callback)
        return callback

    def on_run_complete(
        self, callback: Callable[[Run, list[Artifact]], None]
    ) -> Callable[[Run, list[Artifact]], None]:
        self.complete_hooks.append(callback)
        return callback

    def on_run_failed(
        self, callback: Callable[[Run, Exception], None]
    ) -> Callable[[Run, Exception], None]:
        self.failed_hooks.append(callback)
        return callback

    def emit_start(self, run: Run) -> None:
        for hook in self.start_hooks:
            hook(run)

    def emit_complete(self, run: Run, outputs: list[Artifact]) -> None:
        for hook in self.complete_hooks:
            hook(run, outputs)

    def emit_failed(self, run: Run, error: Exception) -> None:
        for hook in self.failed_hooks:
            hook(run, error)


class FakeResponse:
    def raise_for_status(self) -> None:
        return None


class RecordedPost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SheetWebhookPayload, float]] = []

    def __call__(
        self,
        url: str,
        *,
        json: SheetWebhookPayload,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((url, json, timeout))
        return FakeResponse()


def _context(tmp_path: Path) -> RunNotificationContext:
    return RunNotificationContext(
        run_name="pilates-run--sfbay--baseline",
        scenario_id="baseline",
        seed=42,
        archive_run_dir=str(tmp_path / "archive"),
        local_run_dir=str(tmp_path / "local"),
        settings_file="settings.yaml",
        submit_user="zaneedell",
        slurm_job_id="22338190",
        slurm_job_name="ABC.2026.05.04",
        slurm_partition="lr7",
        slurm_node_list="n0114.lr7",
    )


def test_run_publisher_settings_parse_google_sheet_webhook() -> None:
    settings = RunPublisherSettings.from_env(
        {
            "PILATES_RUN_EVENT_LOG": "0",
            "PILATES_RUN_SUMMARY_HTML": "false",
            "PILATES_RUN_PUBLISHERS_INCLUDE_INTERNAL": "yes",
            "PILATES_GSHEET_PUBLISH": "1",
            "PILATES_GSHEET_WEBHOOK_URL": " https://script.google.com/macros/s/demo/exec ",
            "PILATES_GSHEET_TIMEOUT_SECONDS": "0",
            "PILATES_GSHEET_SECRET": " token ",
        }
    )

    assert settings.local_jsonl_enabled is False
    assert settings.summary_html_enabled is False
    assert settings.include_internal is True
    assert settings.google_sheet.enabled is True
    assert settings.google_sheet.webhook_url == "https://script.google.com/macros/s/demo/exec"
    assert settings.google_sheet.timeout_seconds == 0.25
    assert settings.google_sheet.secret == "token"


def test_register_consist_run_publishers_writes_structured_events(tmp_path: Path) -> None:
    tracker = FakeTracker()
    publisher = FakePublisher()

    registered = register_consist_run_publishers(
        tracker,
        settings=RunPublisherSettings(local_jsonl_enabled=False, summary_html_enabled=False),
        publishers=[publisher],
        context=_context(tmp_path),
    )

    assert registered is not None
    tracker.emit_start(_run("scenario", tags=["scenario_header"]))
    tracker.emit_complete(
        _run(
            "pilates-run--sfbay--baseline__step_func__y2030__i2__phase_run_abcdef0",
            model_name="beam_run",
            parent_run_id="scenario",
            year=2030,
            iteration=2,
            stage="traffic_assignment",
            phase="run",
            meta={"cache_hit": True},
        ),
        [_artifact("linkstats"), _artifact("skims")],
    )
    tracker.emit_start(_run("workspace_setup"))

    assert [event.event_type for event in publisher.events] == ["start", "complete"]
    complete = publisher.events[1]
    assert complete.run_kind == "step"
    assert complete.display_id == "y2030 | i2 | run_abcdef0"
    assert complete.result == "cache hit"
    assert complete.output_count == 2
    assert complete.output_keys == ("linkstats", "skims")
    assert complete.scenario_id == "baseline"
    assert complete.submit_user == "zaneedell"


def test_run_publisher_always_reports_internal_failures(tmp_path: Path) -> None:
    tracker = FakeTracker()
    publisher = FakePublisher()
    register_consist_run_publishers(
        tracker,
        settings=RunPublisherSettings(local_jsonl_enabled=False, summary_html_enabled=False),
        publishers=[publisher],
        context=_context(tmp_path),
    )

    tracker.emit_failed(_run("workspace_setup"), RuntimeError("boom"))

    assert publisher.events[0].event_type == "failed"
    assert publisher.events[0].run_id == "workspace_setup"
    assert publisher.events[0].error == "boom"


def test_local_jsonl_and_summary_html_publishers_write_archive_files(
    tmp_path: Path,
) -> None:
    tracker = FakeTracker()
    context = _context(tmp_path)
    event_log = tmp_path / "archive" / ".pilates" / "run_events.jsonl"
    summary = tmp_path / "archive" / ".pilates" / "run_summary.html"

    register_consist_run_publishers(
        tracker,
        settings=RunPublisherSettings(local_jsonl_enabled=False, summary_html_enabled=False),
        publishers=[
            LocalJsonlRunEventPublisher(event_log),
            SummaryHtmlRunEventPublisher(summary),
        ],
        context=context,
    )

    tracker.emit_complete(
        _run("step", model_name="activitysim_run", parent_run_id="scenario"),
        [_artifact("persons")],
    )

    rows = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert rows[0]["event_type"] == "complete"
    assert rows[0]["model"] == "activitysim_run"
    assert rows[0]["result"] == "executed"
    assert rows[0]["output_keys"] == ["persons"]
    html = summary.read_text()
    assert "pilates-run--sfbay--baseline" in html
    assert "activitysim_run" in html
    assert "executed" in html


def test_google_sheet_webhook_publisher_posts_row_and_event(monkeypatch) -> None:
    recorded = RecordedPost()
    monkeypatch.setattr(run_publishers.requests, "post", recorded)
    publisher = GoogleSheetWebhookRunEventPublisher(
        "https://script.google.com/macros/s/demo/exec",
        timeout_seconds=4.0,
        secret="shared-secret",
    )
    event = RunEvent(
        event_type="complete",
        event_time="2026-05-04T12:00:00+00:00",
        run_kind="step",
        run_id="step-a",
        display_id="step-a",
        model="beam_run",
        status="completed",
        result="executed",
        scenario_id="baseline",
        year=2030,
        iteration=2,
        stage="traffic_assignment",
        phase="run",
        parent_run_id="scenario",
        run_name="pilates-run",
        submit_user="zaneedell",
        slurm_job_id="22338190",
        slurm_job_name="ABC",
        slurm_partition="lr7",
        slurm_node_list="n0114.lr7",
        hostname=None,
        archive_run_dir="/global/scratch/run",
        local_run_dir="/local/run",
        settings_file="settings.yaml",
        started_at="2026-05-04T11:59:00+00:00",
        ended_at="2026-05-04T12:00:00+00:00",
        duration_seconds=60.0,
        output_count=1,
        output_keys=("linkstats",),
        error=None,
        config_hash="config",
        input_hash="input",
        git_hash="git",
        signature="sig",
    )

    publisher.publish(event)

    url, payload, timeout = recorded.calls[0]
    assert url == "https://script.google.com/macros/s/demo/exec"
    assert timeout == 4.0
    assert payload["kind"] == "pilates_run_event"
    assert payload["secret"] == "shared-secret"
    assert payload["event"]["run_id"] == "step-a"
    assert payload["row"][0] == "2026-05-04T12:00:00+00:00"
