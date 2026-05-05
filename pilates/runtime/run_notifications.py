"""Run notification hooks for PILATES Consist runs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from consist.models.artifact import Artifact
from consist.models.run import Run

logger = logging.getLogger(__name__)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}
_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_MAX_ERROR_CHARS = 500
_GCHAT_REPLY_OPTION = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"

SlackTextObject = dict[str, str]
SlackBlock = dict[str, Union[str, SlackTextObject]]
SlackPayload = dict[str, Union[str, list[SlackBlock]]]
GoogleChatThread = dict[str, str]
GoogleChatPayload = dict[str, Union[str, GoogleChatThread]]


class ConsistHookTracker(Protocol):
    """Consist tracker hook surface used by run notifications."""

    def on_run_start(self, callback: Callable[[Run], None]) -> Callable[[Run], None]:
        """Register a run-start callback."""

    def on_run_complete(
        self, callback: Callable[[Run, list[Artifact]], None]
    ) -> Callable[[Run, list[Artifact]], None]:
        """Register a run-complete callback."""

    def on_run_failed(
        self, callback: Callable[[Run, Exception], None]
    ) -> Callable[[Run, Exception], None]:
        """Register a run-failed callback."""


class RunNotificationBackend(Protocol):
    """Delivery backend for provider-specific notification messages."""

    name: str

    def send(self, message: "RunNotificationMessage") -> None:
        """Send a prepared run notification message."""


@dataclass(frozen=True)
class ProviderWebhookSettings:
    """Provider-specific webhook settings."""

    enabled: bool
    webhook_url: Optional[str] = None
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    disabled_reason: Optional[str] = None


@dataclass(frozen=True)
class RunNotificationSettings:
    """Environment-derived run notification settings."""

    include_internal: bool = False
    max_error_chars: int = _DEFAULT_MAX_ERROR_CHARS
    slack: ProviderWebhookSettings = field(
        default_factory=lambda: ProviderWebhookSettings(enabled=False)
    )
    google_chat: ProviderWebhookSettings = field(
        default_factory=lambda: ProviderWebhookSettings(enabled=False)
    )

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> "RunNotificationSettings":
        values = os.environ if env is None else env
        include_internal = _parse_bool(
            values.get("PILATES_RUN_NOTIFICATIONS_INCLUDE_INTERNAL"),
            default=_parse_bool(
                values.get("PILATES_SLACK_INCLUDE_INTERNAL"),
                default=_parse_bool(
                values.get("PILATES_GCHAT_INCLUDE_INTERNAL"),
                    default=_parse_bool(
                        values.get("PILATES_SLACK_VERBOSE"),
                        default=False,
                    ),
                ),
            ),
        )
        max_error_chars = int(
            _parse_float(
                values.get("PILATES_RUN_NOTIFICATIONS_MAX_ERROR_CHARS")
                or values.get("PILATES_SLACK_MAX_ERROR_CHARS")
                or values.get("PILATES_GCHAT_MAX_ERROR_CHARS"),
                default=float(_DEFAULT_MAX_ERROR_CHARS),
                minimum=80.0,
                maximum=4000.0,
            )
        )
        return cls(
            include_internal=include_internal,
            max_error_chars=max_error_chars,
            slack=_provider_from_env(
                values,
                enabled_var="PILATES_SLACK_NOTIFICATIONS",
                webhook_var="PILATES_SLACK_WEBHOOK_URL",
                timeout_var="PILATES_SLACK_TIMEOUT_SECONDS",
            ),
            google_chat=_provider_from_env(
                values,
                enabled_var="PILATES_GCHAT_NOTIFICATIONS",
                webhook_var="PILATES_GCHAT_WEBHOOK_URL",
                timeout_var="PILATES_GCHAT_TIMEOUT_SECONDS",
            ),
        )


@dataclass(frozen=True)
class RunNotificationContext:
    """Run-level context included in notification messages."""

    run_name: Optional[str] = None
    scenario_id: Optional[str] = None
    seed: Optional[int] = None
    archive_run_dir: Optional[str] = None
    local_run_dir: Optional[str] = None
    settings_file: Optional[str] = None
    submit_user: Optional[str] = None
    slurm_job_id: Optional[str] = None
    slurm_job_name: Optional[str] = None
    slurm_partition: Optional[str] = None
    slurm_node_list: Optional[str] = None
    hostname: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        *,
        env: Optional[Mapping[str, str]] = None,
        run_name: Optional[str] = None,
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
        archive_run_dir: Optional[str] = None,
        local_run_dir: Optional[str] = None,
        settings_file: Optional[str] = None,
    ) -> "RunNotificationContext":
        values = os.environ if env is None else env
        return cls(
            run_name=run_name,
            scenario_id=scenario_id,
            seed=seed,
            archive_run_dir=archive_run_dir,
            local_run_dir=local_run_dir,
            settings_file=settings_file,
            submit_user=_first_env_value(values, ("SLURM_JOB_USER", "USER", "LOGNAME")),
            slurm_job_id=_first_env_value(values, ("SLURM_JOB_ID", "SLURM_JOBID")),
            slurm_job_name=_first_env_value(values, ("SLURM_JOB_NAME",)),
            slurm_partition=_first_env_value(
                values, ("SLURM_JOB_PARTITION", "SLURM_JOB_PARTITION_NAME")
            ),
            slurm_node_list=_first_env_value(
                values, ("SLURM_JOB_NODELIST", "SLURM_NODELIST")
            ),
            hostname=_first_env_value(values, ("SLURMD_NODENAME", "HOSTNAME")),
        )


@dataclass(frozen=True)
class RunNotificationMessage:
    """Provider-neutral notification text built from a Consist run event."""

    title: str
    run_id: str
    lines: tuple[str, ...]
    thread_key: str

    @property
    def fallback_text(self) -> str:
        return f"{self.title}: {self.run_id}"

    @property
    def markdown_text(self) -> str:
        body = "\n".join(f"- {line}" for line in self.lines)
        return f"*{self.title}*\n{body}"

    @property
    def plain_text(self) -> str:
        body = "\n".join(f"- {_strip_backticks(line)}" for line in self.lines)
        return f"*{self.title}*\n{body}"


@dataclass(frozen=True)
class IncomingWebhookSlackBackend:
    """Slack Incoming Webhook delivery backend."""

    webhook_url: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    name: str = "slack"

    def send(self, message: RunNotificationMessage) -> None:
        response = requests.post(
            self.webhook_url,
            json=self._payload(message),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def _payload(self, message: RunNotificationMessage) -> SlackPayload:
        return {
            "text": message.fallback_text,
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{message.title}*"},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "\n".join(f"- {line}" for line in message.lines),
                    },
                },
            ],
        }


@dataclass(frozen=True)
class IncomingWebhookGoogleChatBackend:
    """Google Chat Incoming Webhook delivery backend with thread-key support."""

    webhook_url: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    name: str = "google_chat"

    def send(self, message: RunNotificationMessage) -> None:
        response = requests.post(
            self._url_with_thread_reply_option(),
            json=self._payload(message),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def _payload(self, message: RunNotificationMessage) -> GoogleChatPayload:
        return {
            "text": message.plain_text,
            "thread": {"threadKey": message.thread_key},
        }

    def _url_with_thread_reply_option(self) -> str:
        return _url_with_query_param(
            self.webhook_url,
            "messageReplyOption",
            _GCHAT_REPLY_OPTION,
        )


class ConsistRunNotifier:
    """Consist run hook callbacks that post compact PILATES run updates."""

    def __init__(
        self,
        *,
        settings: RunNotificationSettings,
        backends: Sequence[RunNotificationBackend],
        context: Optional[RunNotificationContext] = None,
    ) -> None:
        self.settings = settings
        self.backends = tuple(backends)
        self.context = context or RunNotificationContext()

    def on_run_start(self, run: Run) -> None:
        if not self._should_notify(run):
            return
        label = self._run_label(run)
        self._send(
            self._build_message(
                title=f"{label} started",
                run=run,
                lines=self._run_lines(run, event="start"),
            )
        )

    def on_run_complete(self, run: Run, outputs: Sequence[Artifact]) -> None:
        if not self._should_notify(run):
            return
        label = self._run_label(run)
        self._send(
            self._build_message(
                title=f"{label} completed",
                run=run,
                lines=self._run_lines(run, event="complete", output_count=len(outputs)),
            )
        )

    def on_run_failed(self, run: Run, error: Exception) -> None:
        if not self._should_notify(run):
            return
        label = self._run_label(run)
        error_text = self._truncate(str(error) or type(error).__name__)
        self._send(
            self._build_message(
                title=f"{label} failed",
                run=run,
                lines=(
                    *self._run_lines(run, event="failed"),
                    f"error: {error_text}",
                ),
            )
        )

    def _send(self, message: RunNotificationMessage) -> None:
        for backend in self.backends:
            try:
                backend.send(message)
            except Exception as exc:
                logger.warning(
                    "PILATES %s notification failed: %s",
                    backend.name,
                    _safe_request_error(exc),
                )

    def _should_notify(self, run: Run) -> bool:
        if self.settings.include_internal:
            return True
        return _is_scenario_header(run) or bool(run.parent_run_id)

    def _run_label(self, run: Run) -> str:
        if _is_scenario_header(run):
            return "PILATES run"
        return "PILATES step"

    def _run_lines(
        self,
        run: Run,
        *,
        event: str,
        output_count: Optional[int] = None,
    ) -> tuple[str, ...]:
        is_scenario = _is_scenario_header(run)
        lines = (
            self._scenario_lines(run, event=event)
            if is_scenario
            else self._step_lines(run, event=event)
        )
        cache_status = _cache_status(run, event=event)
        if cache_status:
            lines.append(f"Result: {cache_status}")
        duration = run.duration_seconds
        if duration is not None:
            lines.append(f"Duration: {_format_duration(duration)}")
        if output_count is not None:
            lines.append(f"Outputs: {output_count}")

        if not is_scenario:
            return tuple(lines)

        if self.context.run_name and self.context.run_name != _string_value(
            run.id, ""
        ):
            lines.append(f"Consist run: `{_string_value(run.id, '<unknown>')}`")
        if self.context.scenario_id:
            lines.append(f"Scenario: `{self.context.scenario_id}`")
        if self.context.seed is not None:
            lines.append(f"Seed: `{self.context.seed}`")
        if self.context.submit_user:
            lines.append(f"User: `{self.context.submit_user}`")
        slurm_job = _format_slurm_job(self.context)
        if slurm_job:
            lines.append(f"Slurm job: `{slurm_job}`")
        cluster_parts = _cluster_parts(self.context)
        if cluster_parts:
            lines.append(f"Cluster: {' | '.join(f'`{part}`' for part in cluster_parts)}")
        node_label = _node_label(self.context)
        if node_label and not cluster_parts:
            lines.append(f"Nodes: `{node_label}`")
        if self.context.archive_run_dir:
            lines.append(f"Archive: `{self.context.archive_run_dir}`")
        return tuple(lines)

    def _scenario_lines(self, run: Run, *, event: str) -> list[str]:
        del event
        return [
            f"Run: `{self.context.run_name or _string_value(run.id, '<unknown>')}`",
            f"Model: `{_string_value(run.model_name, '<unknown>')}`",
        ]

    def _step_lines(self, run: Run, *, event: str) -> list[str]:
        del event
        lines = [
            f"Step: `{_string_value(run.model_name, '<unknown>')}`",
        ]
        timing_parts = _step_timing_parts(run)
        if timing_parts:
            lines.append(f"When: {' | '.join(f'`{part}`' for part in timing_parts)}")
        if run.stage is not None:
            lines.append(f"Stage: `{run.stage}`")
        display_id = _display_run_id(run, context=self.context)
        if display_id:
            lines.append(f"ID: `{display_id}`")
        return lines

    def _build_message(
        self,
        *,
        title: str,
        run: Run,
        lines: Sequence[str],
    ) -> RunNotificationMessage:
        run_id = _string_value(run.id, "<unknown>")
        return RunNotificationMessage(
            title=title,
            run_id=_display_run_id(run, context=self.context) or run_id,
            lines=tuple(lines),
            thread_key=_thread_key(context=self.context, run=run),
        )

    def _truncate(self, text: str) -> str:
        limit = self.settings.max_error_chars
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."


def register_consist_run_notification_hooks(
    tracker: ConsistHookTracker,
    *,
    context: Optional[RunNotificationContext] = None,
    settings: Optional[RunNotificationSettings] = None,
    backends: Optional[Sequence[RunNotificationBackend]] = None,
) -> Optional[ConsistRunNotifier]:
    """Register run notification callbacks on a Consist tracker when enabled."""
    resolved_settings = settings or RunNotificationSettings.from_env()
    resolved_backends = (
        tuple(backends) if backends is not None else _build_backends(resolved_settings)
    )
    if not resolved_backends:
        logger.info(
            "PILATES run notifications disabled: %s",
            _disabled_summary(resolved_settings),
        )
        return None

    notifier = ConsistRunNotifier(
        settings=resolved_settings,
        backends=resolved_backends,
        context=context,
    )
    try:
        tracker.on_run_start(notifier.on_run_start)
        tracker.on_run_complete(notifier.on_run_complete)
        tracker.on_run_failed(notifier.on_run_failed)
    except AttributeError as exc:
        logger.warning(
            "PILATES run notifications disabled: tracker lacks Consist hooks: %s",
            exc,
        )
        return None
    logger.info(
        "PILATES run notifications enabled for: %s",
        ", ".join(backend.name for backend in resolved_backends),
    )
    return notifier


def _build_backends(
    settings: RunNotificationSettings,
) -> tuple[RunNotificationBackend, ...]:
    backends: list[RunNotificationBackend] = []
    if settings.slack.enabled and settings.slack.webhook_url:
        backends.append(
            IncomingWebhookSlackBackend(
                settings.slack.webhook_url,
                timeout_seconds=settings.slack.timeout_seconds,
            )
        )
    if settings.google_chat.enabled and settings.google_chat.webhook_url:
        backends.append(
            IncomingWebhookGoogleChatBackend(
                settings.google_chat.webhook_url,
                timeout_seconds=settings.google_chat.timeout_seconds,
            )
        )
    return tuple(backends)


def _provider_from_env(
    env: Mapping[str, str],
    *,
    enabled_var: str,
    webhook_var: str,
    timeout_var: str,
) -> ProviderWebhookSettings:
    requested = _parse_bool(env.get(enabled_var), default=False)
    webhook_url = _clean_optional(env.get(webhook_var))
    timeout_seconds = _parse_float(
        env.get(timeout_var),
        default=_DEFAULT_TIMEOUT_SECONDS,
        minimum=0.25,
        maximum=60.0,
    )
    if not requested:
        return ProviderWebhookSettings(
            enabled=False,
            webhook_url=webhook_url,
            timeout_seconds=timeout_seconds,
            disabled_reason=f"{enabled_var} is not enabled",
        )
    if not webhook_url:
        return ProviderWebhookSettings(
            enabled=False,
            timeout_seconds=timeout_seconds,
            disabled_reason=f"{enabled_var} is enabled but {webhook_var} is not set",
        )
    return ProviderWebhookSettings(
        enabled=True,
        webhook_url=webhook_url,
        timeout_seconds=timeout_seconds,
        disabled_reason=None,
    )


def _disabled_summary(settings: RunNotificationSettings) -> str:
    reasons = [
        reason
        for reason in (
            settings.slack.disabled_reason,
            settings.google_chat.disabled_reason,
        )
        if reason
    ]
    return "; ".join(reasons) or "no notification backends enabled"


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
        except (TypeError, ValueError):
            parsed = default
    return min(max(parsed, minimum), maximum)


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _first_env_value(env: Mapping[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key not in env:
            continue
        cleaned = _clean_optional(env[key])
        if cleaned:
            return cleaned
    return None


def _is_scenario_header(run: Run) -> bool:
    return "scenario_header" in run.tags


def _format_slurm_job(context: RunNotificationContext) -> Optional[str]:
    if context.slurm_job_id and context.slurm_job_name:
        return f"{context.slurm_job_id} ({context.slurm_job_name})"
    return context.slurm_job_id or context.slurm_job_name


def _cluster_parts(context: RunNotificationContext) -> tuple[str, ...]:
    parts: list[str] = []
    if context.slurm_partition:
        parts.append(context.slurm_partition)
    node_label = _node_label(context)
    if node_label:
        parts.append(node_label)
    return tuple(parts)


def _node_label(context: RunNotificationContext) -> Optional[str]:
    return context.slurm_node_list or context.hostname


def _step_timing_parts(run: Run) -> tuple[str, ...]:
    parts: list[str] = []
    if run.year is not None:
        parts.append(f"year {run.year}")
    if run.iteration is not None:
        parts.append(f"iter {run.iteration}")
    if run.phase is not None:
        parts.append(str(run.phase))
    return tuple(parts)


def _display_run_id(run: Run, *, context: RunNotificationContext) -> str:
    run_id = _string_value(run.id, "")
    if not run_id:
        return ""
    if _is_scenario_header(run):
        return context.run_name or run_id

    display_id = run_id
    if context.run_name and display_id.startswith(context.run_name):
        display_id = display_id[len(context.run_name) :].lstrip("_")
    if display_id.startswith("step_func__"):
        display_id = display_id[len("step_func__") :]
    display_id = display_id.replace("__phase_", "__")
    display_id = display_id.replace("__", " | ")
    return display_id or run_id


def _cache_status(run: Run, *, event: str) -> Optional[str]:
    meta: Mapping[str, object] = run.meta
    if bool(meta.get("cache_hit")):
        return "cache hit"
    if event == "complete":
        return "executed"
    return None


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(remainder)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _string_value(value: Optional[object], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _strip_backticks(text: str) -> str:
    return text.replace("`", "")


def _thread_key(*, context: RunNotificationContext, run: Run) -> str:
    if context.run_name:
        return _sanitize_thread_key(context.run_name)
    if run.parent_run_id:
        return _sanitize_thread_key(run.parent_run_id)
    return _sanitize_thread_key(run.id)


def _sanitize_thread_key(value: str) -> str:
    sanitized = "".join(
        char if char.isalnum() or char in "-_." else "-" for char in value
    )
    return sanitized[:512] or "pilates-run"


def _url_with_query_param(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    query_pairs = [
        (existing_key, existing_value)
        for existing_key, existing_value in parse_qsl(parts.query)
        if existing_key != key
    ]
    query_pairs.append((key, value))
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query_pairs),
            parts.fragment,
        )
    )


def _safe_request_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is not None:
            reason = response.reason or "HTTP error"
            return f"HTTP {response.status_code} {reason}"
        return type(exc).__name__
    if isinstance(exc, requests.RequestException):
        return type(exc).__name__
    return str(exc)
