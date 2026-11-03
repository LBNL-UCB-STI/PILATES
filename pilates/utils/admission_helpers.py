"""Shared helpers for persisting and extracting PILATES input-admission reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import consist


@runtime_checkable
class AdmissionMetadataLogger(Protocol):
    """Minimal active-run metadata capability required for admission evidence."""

    def log_meta(self, **metadata: object) -> None: ...


def extract_admission_reports(
    run: object | None,
    *,
    default: Mapping[str, object] | None = None,
) -> Mapping[str, object] | None:
    """Return the active run's ``admission_reports`` mapping when present."""
    try:
        meta = run.meta  # type: ignore[attr-defined]
    except AttributeError:
        return default
    if isinstance(meta, Mapping):
        reports = meta.get("admission_reports")
        if isinstance(reports, Mapping):
            return reports
    return default


def resolve_tracker_report_dir(
    tracker: object | None,
    *,
    fallback: Path | str,
) -> Path:
    """Resolve the admission report directory from a tracker's ``run_dir`` or fallback."""
    try:
        run_dir = tracker.run_dir  # type: ignore[attr-defined]
    except AttributeError:
        run_dir = None
    return Path(run_dir) if run_dir else Path(fallback)


def persist_admission_report(
    *,
    report: consist.admission.AdmissionReport,
    report_dir: Path | str,
    report_filename: str,
    role: str,
    metadata_logger: AdmissionMetadataLogger | None,
    existing_admission_reports: Mapping[str, object] | None = None,
    extra_payload: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Write deterministic admission evidence to disk and merge into run metadata."""
    canonical_report = report.canonical_json()
    payload: dict[str, Any] = json.loads(canonical_report)
    has_extra = False
    if extra_payload is not None:
        for key, value in extra_payload.items():
            if value is not None:
                payload[key] = dict(value)
                has_extra = True
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if has_extra
        else canonical_report
    )
    report_path = Path(report_dir) / "admission" / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(serialized + "\n", encoding="utf-8")

    if metadata_logger is not None:
        admission_reports = dict(existing_admission_reports or {})
        admission_reports[role] = payload
        metadata_logger.log_meta(admission_reports=admission_reports)
    return payload


__all__ = [
    "AdmissionMetadataLogger",
    "extract_admission_reports",
    "persist_admission_report",
    "resolve_tracker_report_dir",
]
