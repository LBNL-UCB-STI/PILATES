"""PILATES policy for admitting one staged BEAM warm-start input."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import consist

from pilates.beam.launch_paths import BeamLaunchPathReference


logger = logging.getLogger(__name__)
_LINKSTATS_ROLE = "beam_linkstats_warmstart"


class BeamInputAdmissionError(RuntimeError):
    """Raised when configured BEAM input-admission policy rejects a file."""


def preflight_staged_linkstats_admission(
    *,
    tracker: Any,
    settings: Any,
    launch_reference: BeamLaunchPathReference,
    report_dir: Path,
) -> consist.AdmissionReport | None:
    """Check staged linkstats against an explicit baseline when configured.

    An absent expectation is deliberately a no-op: it preserves ordinary BEAM
    warm-start behavior and does not invent an inbound-admission target.
    """
    beam_config = getattr(settings, "beam", None)
    admission_config = getattr(beam_config, "admission", None)
    linkstats_config = getattr(admission_config, "linkstats", None)
    if linkstats_config is None:
        return None

    runtime_reference = consist.AdmissionReference(
        artifact_key=str(linkstats_config.artifact_key),
        input_role=_LINKSTATS_ROLE,
        config_key=launch_reference.config_key,
        config_reference_key=launch_reference.config_key,
        raw_config_value=launch_reference.raw_value,
        canonical_value=launch_reference.canonical_value,
        configured_path=launch_reference.configured_path,
        execution_path=launch_reference.execution_path,
        consumer_path=launch_reference.container_path,
    )
    report = consist.check_admission_reference(
        tracker,
        reference=runtime_reference,
        expected_run_id=str(linkstats_config.expected_run_id),
        expected_bytes_path=linkstats_config.expected_bytes_path,
    )
    _persist_admission_report(report_dir=report_dir, tracker=tracker, report=report)

    if report.outcome == "verified":
        return report
    if linkstats_config.mode == "strict":
        raise BeamInputAdmissionError(
            "BEAM linkstats admission rejected before execution: "
            f"{report.outcome}"
        )
    logger.warning(
        "BEAM linkstats admission recorded %s and will continue because mode=warn",
        report.outcome,
    )
    return report


def reject_or_warn_for_missing_staged_linkstats(*, settings: Any) -> None:
    """Apply configured policy when preprocessing did not stage a warm-start file."""
    beam_config = getattr(settings, "beam", None)
    admission_config = getattr(beam_config, "admission", None)
    linkstats_config = getattr(admission_config, "linkstats", None)
    if linkstats_config is None:
        return

    message = (
        "Configured BEAM linkstats admission requires a staged warm-start "
        "linkstats file, but BEAM preprocess did not publish linkstats_warmstart."
    )
    if linkstats_config.mode == "strict":
        raise BeamInputAdmissionError(message)
    logger.warning("%s Continuing because mode=warn.", message)


def _persist_admission_report(
    *, report_dir: Path, tracker: Any, report: consist.AdmissionReport
) -> None:
    """Write deterministic sidecar and run metadata before policy can reject."""
    payload = json.loads(report.canonical_json())
    report_path = Path(report_dir) / "admission" / "beam-linkstats-warmstart.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.canonical_json() + "\n", encoding="utf-8")
    tracker.log_meta(admission_reports={_LINKSTATS_ROLE: payload})
