from __future__ import annotations

from datetime import timedelta
from typing import Callable

import pytest
from consist.models.artifact import Artifact
from consist.models.run import Run

from pilates.runtime import run_notifications
from pilates.runtime.run_notifications import (
    GoogleChatPayload,
    IncomingWebhookGoogleChatBackend,
    IncomingWebhookSlackBackend,
    ProviderWebhookSettings,
    RunNotificationBackend,
    RunNotificationContext,
    RunNotificationMessage,
    RunNotificationSettings,
    SlackPayload,
    register_consist_run_notification_hooks,
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
    return Artifact(key=key, container_uri=f"./{key}.txt", driver="other")


class FakeBackend(RunNotificationBackend):
    name = "fake"

    def __init__(self) -> None:
        self.messages: list[RunNotificationMessage] = []

    def send(self, message: RunNotificationMessage) -> None:
        self.messages.append(message)


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
        self.calls: list[tuple[str, SlackPayload | GoogleChatPayload, float]] = []

    def __call__(
        self,
        url: str,
        *,
        json: SlackPayload | GoogleChatPayload,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((url, json, timeout))
        return FakeResponse()


def _enabled_settings(
    *,
    include_internal: bool = False,
    max_error_chars: int = 500,
) -> RunNotificationSettings:
    return RunNotificationSettings(
        include_internal=include_internal,
        max_error_chars=max_error_chars,
        slack=ProviderWebhookSettings(enabled=False),
        google_chat=ProviderWebhookSettings(enabled=False),
    )


def test_run_notification_settings_are_disabled_by_default() -> None:
    settings = RunNotificationSettings.from_env({})

    assert settings.slack.enabled is False
    assert settings.google_chat.enabled is False
    assert "SLACK_NOTIFICATIONS" in str(settings.slack.disabled_reason)
    assert "GCHAT_NOTIFICATIONS" in str(settings.google_chat.disabled_reason)


def test_run_notification_settings_require_provider_webhooks() -> None:
    settings = RunNotificationSettings.from_env(
        {
            "PILATES_SLACK_NOTIFICATIONS": "1",
            "PILATES_GCHAT_NOTIFICATIONS": "1",
        }
    )

    assert settings.slack.enabled is False
    assert settings.google_chat.enabled is False
    assert "SLACK_WEBHOOK_URL" in str(settings.slack.disabled_reason)
    assert "GCHAT_WEBHOOK_URL" in str(settings.google_chat.disabled_reason)


def test_run_notification_settings_parse_provider_values() -> None:
    settings = RunNotificationSettings.from_env(
        {
            "PILATES_SLACK_NOTIFICATIONS": "true",
            "PILATES_SLACK_WEBHOOK_URL": " https://hooks.slack.com/services/T/B/C ",
            "PILATES_SLACK_TIMEOUT_SECONDS": "120",
            "PILATES_GCHAT_NOTIFICATIONS": "yes",
            "PILATES_GCHAT_WEBHOOK_URL": " https://chat.googleapis.com/v1/spaces/S/messages?key=K&token=T ",
            "PILATES_GCHAT_TIMEOUT_SECONDS": "0",
            "PILATES_RUN_NOTIFICATIONS_INCLUDE_INTERNAL": "yes",
            "PILATES_RUN_NOTIFICATIONS_MAX_ERROR_CHARS": "40",
        }
    )

    assert settings.slack.enabled is True
    assert settings.slack.webhook_url == "https://hooks.slack.com/services/T/B/C"
    assert settings.slack.timeout_seconds == 60.0
    assert settings.google_chat.enabled is True
    assert (
        settings.google_chat.webhook_url
        == "https://chat.googleapis.com/v1/spaces/S/messages?key=K&token=T"
    )
    assert settings.google_chat.timeout_seconds == 0.25
    assert settings.include_internal is True
    assert settings.max_error_chars == 80


def test_run_notification_context_reads_user_and_slurm_env() -> None:
    context = RunNotificationContext.from_env(
        env={
            "USER": "zaneedell",
            "SLURM_JOB_ID": "22337428",
            "SLURM_JOB_NAME": "NGSWDCQD",
            "SLURM_JOB_PARTITION": "lr7",
            "SLURM_JOB_NODELIST": "n0114.lr7",
            "HOSTNAME": "submit-host",
        },
        run_name="pilates-run",
        scenario_id="base",
    )

    assert context.run_name == "pilates-run"
    assert context.scenario_id == "base"
    assert context.submit_user == "zaneedell"
    assert context.slurm_job_id == "22337428"
    assert context.slurm_job_name == "NGSWDCQD"
    assert context.slurm_partition == "lr7"
    assert context.slurm_node_list == "n0114.lr7"


def test_notifier_filters_to_scenario_headers_and_child_runs() -> None:
    backend = FakeBackend()
    notifier = run_notifications.ConsistRunNotifier(
        settings=_enabled_settings(),
        backends=[backend],
        context=RunNotificationContext(scenario_id="scenario-a"),
    )

    notifier.on_run_start(_run("scenario", tags=["scenario_header"]))
    notifier.on_run_start(_run("step", parent_run_id="scenario"))
    notifier.on_run_start(_run("workspace_setup"))

    assert [message.fallback_text for message in backend.messages] == [
        "PILATES run started: scenario",
        "PILATES step started: step",
    ]


def test_notifier_verbose_mode_includes_internal_runs() -> None:
    backend = FakeBackend()
    notifier = run_notifications.ConsistRunNotifier(
        settings=_enabled_settings(include_internal=True),
        backends=[backend],
    )

    notifier.on_run_start(_run("workspace_setup"))

    assert backend.messages[0].fallback_text == "PILATES step started: workspace_setup"


def test_notifier_message_includes_run_metadata_outputs_and_archive() -> None:
    backend = FakeBackend()
    notifier = run_notifications.ConsistRunNotifier(
        settings=_enabled_settings(),
        backends=[backend],
        context=RunNotificationContext(
            run_name="pilates-run--sfbay--baseline",
            scenario_id="sfbay-baseline",
            seed=42,
            archive_run_dir="/global/scratch/run",
            submit_user="zaneedell",
            slurm_job_id="22337428",
            slurm_job_name="NGSWDCQD",
            slurm_partition="lr7",
            slurm_node_list="n0114.lr7",
        ),
    )
    run = _run(
        "beam_run__y2020__i0",
        model_name="beam_run",
        parent_run_id="scenario",
        year=2020,
        iteration=0,
        stage="supply_demand",
        phase="traffic_assignment",
        meta={"cache_hit": True},
    )
    run.ended_at = run.started_at + timedelta(seconds=125)

    notifier.on_run_complete(run, outputs=[_artifact("a"), _artifact("b")])

    message = backend.messages[0]
    assert message.fallback_text == "PILATES step completed: beam_run__y2020__i0"
    assert message.thread_key == "pilates-run--sfbay--baseline"
    assert "cache: hit" in message.markdown_text
    assert "outputs: 2" in message.markdown_text
    assert "duration: 2m 5s" in message.markdown_text
    assert "archive: `/global/scratch/run`" in message.markdown_text
    assert "scenario_id: `sfbay-baseline`" in message.markdown_text
    assert "user: `zaneedell`" in message.markdown_text
    assert "slurm_job: `22337428 (NGSWDCQD)`" in message.markdown_text
    assert "partition: `lr7`" in message.markdown_text
    assert "nodes: `n0114.lr7`" in message.markdown_text


def test_notifier_truncates_long_failure_errors() -> None:
    backend = FakeBackend()
    notifier = run_notifications.ConsistRunNotifier(
        settings=_enabled_settings(max_error_chars=80),
        backends=[backend],
    )

    notifier.on_run_failed(
        _run("failed_step", parent_run_id="scenario"),
        RuntimeError("x" * 200),
    )

    text = backend.messages[0].markdown_text
    assert "error: " in text
    assert "..." in text
    assert len(text) < 260


def test_register_consist_run_notification_hooks_uses_fake_backend() -> None:
    tracker = FakeTracker()
    backend = FakeBackend()

    notifier = register_consist_run_notification_hooks(
        tracker,
        settings=_enabled_settings(),
        backends=[backend],
        context=RunNotificationContext(scenario_id="scenario-a"),
    )

    assert notifier is not None
    tracker.emit_start(_run("scenario", tags=["scenario_header"]))
    tracker.emit_complete(_run("step", parent_run_id="scenario"), [_artifact("out")])
    tracker.emit_failed(_run("bad_step", parent_run_id="scenario"), ValueError("bad"))

    assert [message.fallback_text for message in backend.messages] == [
        "PILATES run started: scenario",
        "PILATES step completed: step",
        "PILATES step failed: bad_step",
    ]


def test_register_consist_run_notification_hooks_noops_when_disabled() -> None:
    tracker = FakeTracker()

    notifier = register_consist_run_notification_hooks(
        tracker,
        settings=_enabled_settings(),
    )

    assert notifier is None
    assert tracker.start_hooks == []


def test_register_consist_run_notification_hooks_with_real_consist_scenario(tmp_path) -> None:
    consist = pytest.importorskip("consist")

    backend = FakeBackend()
    tracker = consist.Tracker(
        run_dir=tmp_path / "runs",
        db_path=tmp_path / "consist.duckdb",
        allow_external_paths=True,
    )
    register_consist_run_notification_hooks(
        tracker,
        settings=_enabled_settings(),
        backends=[backend],
        context=RunNotificationContext(run_name="smoke_scenario", scenario_id="smoke"),
    )

    with tracker.scenario(
        "smoke_scenario",
        model="pilates_orchestrator",
        tags=["pilates_simulation"],
    ) as scenario:
        scenario.run(
            fn=lambda: None,
            name="smoke_step",
            model="smoke_model",
            outputs=[],
        )

    messages = [message.fallback_text for message in backend.messages]
    assert messages[0] == "PILATES run started: smoke_scenario"
    assert any(
        message.startswith("PILATES step started: smoke_scenario_smoke_step")
        for message in messages
    )
    assert any(
        message.startswith("PILATES step completed: smoke_scenario_smoke_step")
        for message in messages
    )
    assert messages[-1] == "PILATES run completed: smoke_scenario"


def test_slack_backend_posts_block_payload(monkeypatch) -> None:
    recorded = RecordedPost()
    monkeypatch.setattr(run_notifications.requests, "post", recorded)
    backend = IncomingWebhookSlackBackend(
        "https://hooks.slack.com/services/T/B/C",
        timeout_seconds=3.0,
    )

    backend.send(
        RunNotificationMessage(
            title="PILATES run started",
            run_id="run-a",
            lines=("run_id: `run-a`",),
            thread_key="run-a",
        )
    )

    url, payload, timeout = recorded.calls[0]
    assert url == "https://hooks.slack.com/services/T/B/C"
    assert timeout == 3.0
    assert payload["text"] == "PILATES run started: run-a"
    assert "blocks" in payload


def test_google_chat_backend_posts_threaded_payload(monkeypatch) -> None:
    recorded = RecordedPost()
    monkeypatch.setattr(run_notifications.requests, "post", recorded)
    backend = IncomingWebhookGoogleChatBackend(
        "https://chat.googleapis.com/v1/spaces/S/messages?key=K&token=T",
        timeout_seconds=4.0,
    )

    backend.send(
        RunNotificationMessage(
            title="PILATES run started",
            run_id="run-a",
            lines=("run_id: `run-a`",),
            thread_key="run-a",
        )
    )

    url, payload, timeout = recorded.calls[0]
    assert url == (
        "https://chat.googleapis.com/v1/spaces/S/messages?"
        "key=K&token=T&messageReplyOption=REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
    )
    assert timeout == 4.0
    assert payload == {
        "text": "PILATES run started\n- run_id: run-a",
        "thread": {"threadKey": "run-a"},
    }
