"""PILATES policy for admitting one staged BEAM warm-start input."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Protocol

import consist

from pilates.beam.launch_paths import BeamLaunchPathReference
from pilates.config import PilatesConfig
from pilates.config.models import BeamLinkstatsAdmissionConfig


logger = logging.getLogger(__name__)
_LINKSTATS_ROLE = "beam_linkstats_warmstart"


class BeamInputAdmissionError(RuntimeError):
    """Raised when configured BEAM input-admission policy rejects a file."""


class AdmissionMetadataTracker(Protocol):
    """Minimal tracker capability used to retain a preflight report."""

    def log_meta(self, **metadata: object) -> None: ...


def _linkstats_admission_config(
    settings: PilatesConfig,
) -> BeamLinkstatsAdmissionConfig | None:
    """Return the optional typed admission policy for staged linkstats."""
    beam_config = settings.beam
    if beam_config is None or beam_config.admission is None:
        return None
    return beam_config.admission.linkstats


def has_staged_linkstats_admission_policy(*, settings: PilatesConfig) -> bool:
    """Return whether BEAM must evaluate staged-linkstats admission policy."""
    return _linkstats_admission_config(settings) is not None


def preflight_staged_linkstats_admission(
    *,
    tracker: consist.Tracker,
    metadata_logger: AdmissionMetadataTracker,
    existing_admission_reports: Mapping[str, object],
    settings: PilatesConfig,
    launch_reference: BeamLaunchPathReference,
    report_dir: Path,
) -> consist.AdmissionReport | None:
    """Check staged linkstats against an explicit baseline when configured.

    An absent expectation is deliberately a no-op: it preserves ordinary BEAM
    warm-start behavior and does not invent an inbound-admission target.
    """
    linkstats_config = _linkstats_admission_config(settings)
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
    _persist_admission_report(
        report_dir=report_dir,
        metadata_logger=metadata_logger,
        existing_admission_reports=existing_admission_reports,
        report=report,
    )

    if report.outcome == "verified":
        return report
    if linkstats_config.mode == "strict":
        raise BeamInputAdmissionError(
            f"BEAM linkstats admission rejected before execution: {report.outcome}"
        )
    logger.warning(
        "BEAM linkstats admission recorded %s and will continue because mode=warn",
        report.outcome,
    )
    return report


def reject_or_warn_for_missing_staged_linkstats(*, settings: PilatesConfig) -> None:
    """Apply configured policy when preprocessing did not stage a warm-start file."""
    linkstats_config = _linkstats_admission_config(settings)
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
    *,
    report_dir: Path,
    metadata_logger: AdmissionMetadataTracker,
    existing_admission_reports: Mapping[str, object],
    report: consist.AdmissionReport,
) -> None:
    """Write deterministic sidecar and run metadata before policy can reject."""
    payload = json.loads(report.canonical_json())
    report_path = Path(report_dir) / "admission" / "beam-linkstats-warmstart.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.canonical_json() + "\n", encoding="utf-8")
    admission_reports = dict(existing_admission_reports)
    admission_reports[_LINKSTATS_ROLE] = payload
    metadata_logger.log_meta(admission_reports=admission_reports)
