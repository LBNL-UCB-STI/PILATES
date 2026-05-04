"""Structured run event publishers for PILATES Consist runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol, Sequence, Union

import requests
from consist.models.artifact import Artifact
from consist.models.run import Run

from pilates.runtime.run_notifications import ConsistHookTracker, RunNotificationContext

logger = logging.getLogger(__name__)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}
_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_EVENT_LOG_RELATIVE_PATH = ".pilates/run_events.jsonl"
_DEFAULT_SUMMARY_HTML_RELATIVE_PATH = ".pilates/run_summary.html"

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, list["JsonValue"], dict[str, "JsonValue"]]
RunEventPayload = dict[str, JsonValue]
SheetRow = list[JsonScalar]
SheetWebhookPayload = dict[str, Union[str, SheetRow, RunEventPayload]]


class RunEventPublisher(Protocol):
    """Destination for structured run event records."""

    name: str

    def publish(self, event: "RunEvent") -> None:
        """Publish one run event."""


@dataclass(frozen=True)
class SheetWebhookSettings:
    """Google Sheets webhook publishing settings."""

    enabled: bool = False
    webhook_url: Optional[str] = None
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    secret: Optional[str] = None
    disabled_reason: Optional[str] = "PILATES_GSHEET_PUBLISH is not enabled"


@dataclass(frozen=True)
class RunPublisherSettings:
    """Environment-derived settings for structured run event publishing."""

    include_internal: bool = False
    local_jsonl_enabled: bool = True
    summary_html_enabled: bool = True
    event_log_path: Optional[str] = None
    summary_html_path: Optional[str] = None
    google_sheet: SheetWebhookSettings = field(default_factory=SheetWebhookSettings)

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> "RunPublisherSettings":
        values = os.environ if env is None else env
        include_internal = _parse_bool(
            values.get("PILATES_RUN_PUBLISHERS_INCLUDE_INTERNAL"),
            default=False,
        )
        local_jsonl_enabled = _parse_bool(
            values.get("PILATES_RUN_EVENT_LOG"),
            default=True,
        )
        summary_html_enabled = _parse_bool(
            values.get("PILATES_RUN_SUMMARY_HTML"),
            default=True,
        )
        return cls(
            include_internal=include_internal,
            local_jsonl_enabled=local_jsonl_enabled,
            summary_html_enabled=summary_html_enabled,
            event_log_path=_clean_optional(values.get("PILATES_RUN_EVENT_LOG_PATH")),
            summary_html_path=_clean_optional(
                values.get("PILATES_RUN_SUMMARY_HTML_PATH")
            ),
            google_sheet=_sheet_settings_from_env(values),
        )


@dataclass(frozen=True)
class RunEvent:
    """Structured event emitted from a Consist run hook."""

    event_type: str
    event_time: str
    run_kind: str
    run_id: str
    display_id: str
    model: str
    status: str
    result: Optional[str]
    scenario_id: Optional[str]
    year: Optional[int]
    iteration: Optional[int]
    stage: Optional[str]
    phase: Optional[str]
    parent_run_id: Optional[str]
    run_name: Optional[str]
    submit_user: Optional[str]
    slurm_job_id: Optional[str]
    slurm_job_name: Optional[str]
    slurm_partition: Optional[str]
    slurm_node_list: Optional[str]
    hostname: Optional[str]
    archive_run_dir: Optional[str]
    local_run_dir: Optional[str]
    settings_file: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    duration_seconds: Optional[float]
    output_count: Optional[int]
    output_keys: tuple[str, ...]
    error: Optional[str]
    config_hash: Optional[str]
    input_hash: Optional[str]
    git_hash: Optional[str]
    signature: Optional[str]

    def to_payload(self) -> RunEventPayload:
        return {
            "event_type": self.event_type,
            "event_time": self.event_time,
            "run_kind": self.run_kind,
            "run_id": self.run_id,
            "display_id": self.display_id,
            "model": self.model,
            "status": self.status,
            "result": self.result,
            "scenario_id": self.scenario_id,
            "year": self.year,
            "iteration": self.iteration,
            "stage": self.stage,
            "phase": self.phase,
            "parent_run_id": self.parent_run_id,
            "run_name": self.run_name,
            "submit_user": self.submit_user,
            "slurm_job_id": self.slurm_job_id,
            "slurm_job_name": self.slurm_job_name,
            "slurm_partition": self.slurm_partition,
            "slurm_node_list": self.slurm_node_list,
            "hostname": self.hostname,
            "archive_run_dir": self.archive_run_dir,
            "local_run_dir": self.local_run_dir,
            "settings_file": self.settings_file,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "output_count": self.output_count,
            "output_keys": list(self.output_keys),
            "error": self.error,
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "git_hash": self.git_hash,
            "signature": self.signature,
        }

    def to_sheet_row(self) -> SheetRow:
        return [
            self.event_time,
            self.event_type,
            self.run_kind,
            self.run_name,
            self.display_id,
            self.model,
            self.status,
            self.result,
            self.scenario_id,
            self.year,
            self.iteration,
            self.stage,
            self.phase,
            self.submit_user,
            self.slurm_job_id,
            self.slurm_job_name,
            self.slurm_partition,
            self.slurm_node_list or self.hostname,
            self.duration_seconds,
            self.output_count,
            self.archive_run_dir,
            self.error,
        ]


@dataclass(frozen=True)
class LocalJsonlRunEventPublisher:
    """Append structured run events into the archived run directory."""

    path: Path
    name: str = "local_jsonl"

    def publish(self, event: RunEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_payload(), sort_keys=True) + "\n")


class SummaryHtmlRunEventPublisher:
    """Write a small static HTML run summary from observed hook events."""

    name = "summary_html"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._events: list[RunEvent] = []

    def publish(self, event: RunEvent) -> None:
        self._events.append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(_render_summary_html(self._events), encoding="utf-8")


@dataclass(frozen=True)
class GoogleSheetWebhookRunEventPublisher:
    """Publish run event rows to a user-managed Google Sheets webhook endpoint."""

    webhook_url: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    secret: Optional[str] = None
    name: str = "google_sheet_webhook"

    def publish(self, event: RunEvent) -> None:
        payload: SheetWebhookPayload = {
            "kind": "pilates_run_event",
            "row": event.to_sheet_row(),
            "event": event.to_payload(),
        }
        if self.secret:
            payload["secret"] = self.secret
        response = requests.post(
            self.webhook_url,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()


class ConsistRunPublisher:
    """Consist run hook callbacks that publish structured run events."""

    def __init__(
        self,
        *,
        settings: RunPublisherSettings,
        publishers: Sequence[RunEventPublisher],
        context: Optional[RunNotificationContext] = None,
    ) -> None:
        self.settings = settings
        self.publishers = tuple(publishers)
        self.context = context or RunNotificationContext()

    def on_run_start(self, run: Run) -> None:
        if not self._should_publish(run):
            return
        self._publish(self._build_event("start", run, outputs=(), error=None))

    def on_run_complete(self, run: Run, outputs: Sequence[Artifact]) -> None:
        if not self._should_publish(run):
            return
        self._publish(self._build_event("complete", run, outputs=outputs, error=None))

    def on_run_failed(self, run: Run, error: Exception) -> None:
        if not self._should_publish(run):
            return
        self._publish(self._build_event("failed", run, outputs=(), error=error))

    def _should_publish(self, run: Run) -> bool:
        if self.settings.include_internal:
            return True
        return _is_scenario_header(run) or bool(run.parent_run_id)

    def _publish(self, event: RunEvent) -> None:
        for publisher in self.publishers:
            try:
                publisher.publish(event)
            except Exception as exc:
                logger.warning(
                    "PILATES %s run event publishing failed: %s",
                    publisher.name,
                    exc,
                )

    def _build_event(
        self,
        event_type: str,
        run: Run,
        *,
        outputs: Sequence[Artifact],
        error: Optional[Exception],
    ) -> RunEvent:
        return RunEvent(
            event_type=event_type,
            event_time=_now_iso(),
            run_kind="scenario" if _is_scenario_header(run) else "step",
            run_id=run.id,
            display_id=_display_run_id(run, context=self.context),
            model=run.model_name,
            status=run.status,
            result=_result_for_event(run, event_type=event_type),
            scenario_id=self.context.scenario_id,
            year=run.year,
            iteration=run.iteration,
            stage=run.stage,
            phase=run.phase,
            parent_run_id=run.parent_run_id,
            run_name=self.context.run_name,
            submit_user=self.context.submit_user,
            slurm_job_id=self.context.slurm_job_id,
            slurm_job_name=self.context.slurm_job_name,
            slurm_partition=self.context.slurm_partition,
            slurm_node_list=self.context.slurm_node_list,
            hostname=self.context.hostname,
            archive_run_dir=self.context.archive_run_dir,
            local_run_dir=self.context.local_run_dir,
            settings_file=self.context.settings_file,
            started_at=_datetime_iso(run.started_at),
            ended_at=_datetime_iso(run.ended_at),
            duration_seconds=run.duration_seconds,
            output_count=len(outputs) if event_type == "complete" else None,
            output_keys=tuple(artifact.key for artifact in outputs),
            error=str(error) if error is not None else None,
            config_hash=run.config_hash,
            input_hash=run.input_hash,
            git_hash=run.git_hash,
            signature=run.signature,
        )


def register_consist_run_publishers(
    tracker: ConsistHookTracker,
    *,
    context: Optional[RunNotificationContext] = None,
    settings: Optional[RunPublisherSettings] = None,
    publishers: Optional[Sequence[RunEventPublisher]] = None,
) -> Optional[ConsistRunPublisher]:
    """Register structured run event publishers on a Consist tracker."""
    resolved_context = context or RunNotificationContext()
    resolved_settings = settings or RunPublisherSettings.from_env()
    resolved_publishers = (
        tuple(publishers)
        if publishers is not None
        else _build_publishers(resolved_settings, resolved_context)
    )
    if not resolved_publishers:
        logger.info(
            "PILATES run event publishers disabled: %s",
            _disabled_summary(resolved_settings),
        )
        return None

    run_publisher = ConsistRunPublisher(
        settings=resolved_settings,
        publishers=resolved_publishers,
        context=resolved_context,
    )
    try:
        tracker.on_run_start(run_publisher.on_run_start)
        tracker.on_run_complete(run_publisher.on_run_complete)
        tracker.on_run_failed(run_publisher.on_run_failed)
    except AttributeError as exc:
        logger.warning(
            "PILATES run event publishers disabled: tracker lacks Consist hooks: %s",
            exc,
        )
        return None
    logger.info(
        "PILATES run event publishers enabled for: %s",
        ", ".join(publisher.name for publisher in resolved_publishers),
    )
    return run_publisher


def _build_publishers(
    settings: RunPublisherSettings,
    context: RunNotificationContext,
) -> tuple[RunEventPublisher, ...]:
    publishers: list[RunEventPublisher] = []
    if settings.local_jsonl_enabled:
        event_log_path = _resolve_output_path(
            configured_path=settings.event_log_path,
            archive_run_dir=context.archive_run_dir,
            relative_default=_DEFAULT_EVENT_LOG_RELATIVE_PATH,
        )
        if event_log_path is not None:
            publishers.append(LocalJsonlRunEventPublisher(event_log_path))
    if settings.summary_html_enabled:
        summary_html_path = _resolve_output_path(
            configured_path=settings.summary_html_path,
            archive_run_dir=context.archive_run_dir,
            relative_default=_DEFAULT_SUMMARY_HTML_RELATIVE_PATH,
        )
        if summary_html_path is not None:
            publishers.append(SummaryHtmlRunEventPublisher(summary_html_path))
    if settings.google_sheet.enabled and settings.google_sheet.webhook_url:
        publishers.append(
            GoogleSheetWebhookRunEventPublisher(
                settings.google_sheet.webhook_url,
                timeout_seconds=settings.google_sheet.timeout_seconds,
                secret=settings.google_sheet.secret,
            )
        )
    return tuple(publishers)


def _sheet_settings_from_env(env: Mapping[str, str]) -> SheetWebhookSettings:
    enabled = _parse_bool(env.get("PILATES_GSHEET_PUBLISH"), default=False)
    webhook_url = _clean_optional(env.get("PILATES_GSHEET_WEBHOOK_URL"))
    timeout_seconds = _parse_float(
        env.get("PILATES_GSHEET_TIMEOUT_SECONDS"),
        default=_DEFAULT_TIMEOUT_SECONDS,
        minimum=0.25,
        maximum=60.0,
    )
    secret = _clean_optional(env.get("PILATES_GSHEET_SECRET"))
    if not enabled:
        return SheetWebhookSettings(
            enabled=False,
            webhook_url=webhook_url,
            timeout_seconds=timeout_seconds,
            secret=secret,
            disabled_reason="PILATES_GSHEET_PUBLISH is not enabled",
        )
    if not webhook_url:
        return SheetWebhookSettings(
            enabled=False,
            timeout_seconds=timeout_seconds,
            secret=secret,
            disabled_reason=(
                "PILATES_GSHEET_PUBLISH is enabled but "
                "PILATES_GSHEET_WEBHOOK_URL is not set"
            ),
        )
    return SheetWebhookSettings(
        enabled=True,
        webhook_url=webhook_url,
        timeout_seconds=timeout_seconds,
        secret=secret,
        disabled_reason=None,
    )


def _resolve_output_path(
    *,
    configured_path: Optional[str],
    archive_run_dir: Optional[str],
    relative_default: str,
) -> Optional[Path]:
    if configured_path:
        path = Path(configured_path).expanduser()
        return path if path.is_absolute() else path.resolve()
    if not archive_run_dir:
        return None
    return Path(archive_run_dir).expanduser() / relative_default


def _disabled_summary(settings: RunPublisherSettings) -> str:
    reasons: list[str] = []
    if not settings.local_jsonl_enabled:
        reasons.append("PILATES_RUN_EVENT_LOG is disabled")
    if not settings.summary_html_enabled:
        reasons.append("PILATES_RUN_SUMMARY_HTML is disabled")
    if settings.google_sheet.disabled_reason:
        reasons.append(settings.google_sheet.disabled_reason)
    return "; ".join(reasons) or "no run event publishers configured"


def _render_summary_html(events: Sequence[RunEvent]) -> str:
    run_title = _first_present(event.run_name for event in events) or "PILATES run"
    rows = "\n".join(_event_table_row(event) for event in events)
    generated_at = _now_iso()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(run_title)} run summary</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    .meta {{ color: #566573; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d5d8dc; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f7; position: sticky; top: 0; }}
    code {{ white-space: nowrap; }}
    .failed {{ background: #fdecea; }}
    .complete {{ background: #edf7ed; }}
  </style>
</head>
<body>
  <h1>{escape(run_title)}</h1>
  <div class="meta">Generated {escape(generated_at)} from Consist run hooks.</div>
  <table>
    <thead>
      <tr>
        <th>Time</th>
        <th>Event</th>
        <th>Kind</th>
        <th>Step</th>
        <th>When</th>
        <th>Status</th>
        <th>Result</th>
        <th>Duration</th>
        <th>Outputs</th>
        <th>Error</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
"""


def _event_table_row(event: RunEvent) -> str:
    css_class = "failed" if event.event_type == "failed" else ""
    if event.event_type == "complete":
        css_class = "complete"
    when = " | ".join(
        part
        for part in (
            f"year {event.year}" if event.year is not None else "",
            f"iter {event.iteration}" if event.iteration is not None else "",
            event.phase if event.phase is not None else "",
        )
        if part
    )
    outputs = "" if event.output_count is None else str(event.output_count)
    duration = "" if event.duration_seconds is None else f"{event.duration_seconds:.1f}s"
    return (
        f'      <tr class="{escape(css_class)}">'
        f"<td>{escape(event.event_time)}</td>"
        f"<td>{escape(event.event_type)}</td>"
        f"<td>{escape(event.run_kind)}</td>"
        f"<td><code>{escape(event.model)}</code><br><small>{escape(event.display_id)}</small></td>"
        f"<td>{escape(when)}</td>"
        f"<td>{escape(event.status)}</td>"
        f"<td>{escape(event.result if event.result is not None else '')}</td>"
        f"<td>{escape(duration)}</td>"
        f"<td>{escape(outputs)}</td>"
        f"<td>{escape(event.error if event.error is not None else '')}</td>"
        "</tr>"
    )


def _first_present(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def _is_scenario_header(run: Run) -> bool:
    return "scenario_header" in run.tags


def _result_for_event(run: Run, *, event_type: str) -> Optional[str]:
    if bool(run.meta.get("cache_hit")):
        return "cache hit"
    if event_type == "complete":
        return "executed"
    return None


def _display_run_id(run: Run, *, context: RunNotificationContext) -> str:
    if _is_scenario_header(run):
        return context.run_name or run.id
    display_id = run.id
    if context.run_name and display_id.startswith(context.run_name):
        display_id = display_id[len(context.run_name) :].lstrip("_")
    if display_id.startswith("step_func__"):
        display_id = display_id[len("step_func__") :]
    display_id = display_id.replace("__phase_", "__")
    display_id = display_id.replace("__", " | ")
    return display_id or run.id


def _datetime_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_bool(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _parse_float(
    value: Optional[str],
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None:
        parsed = default
    else:
        try:
            parsed = float(value)
        except ValueError:
            parsed = default
    return min(max(parsed, minimum), maximum)


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
