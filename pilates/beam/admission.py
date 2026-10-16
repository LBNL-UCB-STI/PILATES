"""PILATES policy for admitting one staged BEAM warm-start input."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Protocol

import consist

from pilates.beam.launch_paths import BeamLaunchPathReference
from pilates.config import PilatesConfig
from pilates.config.models import (
    BeamLinkstatsAdmissionConfig,
    DeclaredDigestExpectation,
    PriorRunExpectation,
)


logger = logging.getLogger(__name__)
_LINKSTATS_ROLE = "beam_linkstats_warmstart"
LINKSTATS_PROVENANCE_SCHEMA_VERSION = "pilates.linkstats_provenance.v1"


class BeamInputAdmissionError(RuntimeError):
    """Raised when configured BEAM input-admission policy rejects a file."""


class AdmissionMetadataTracker(Protocol):
    """Minimal tracker capability used to retain a preflight report."""

    def log_meta(self, **metadata: object) -> None: ...


def linkstats_provenance_sidecar_path(linkstats_path: Path | str) -> Path:
    """Return the co-located provenance sidecar path for ``linkstats_path``."""
    p = Path(linkstats_path)
    return p.parent / f"{p.name}.provenance.json"


def read_linkstats_provenance_sidecar(sidecar_path: Path | str) -> dict[str, object]:
    """Read and validate a ``pilates.linkstats_provenance.v1`` JSON sidecar."""
    p = Path(sidecar_path)
    if not p.is_file():
        raise FileNotFoundError(f"Linkstats provenance sidecar does not exist: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Linkstats provenance sidecar must be a JSON object: {p}")
    schema_version = raw.get("schema_version")
    if schema_version != LINKSTATS_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported linkstats provenance schema_version {schema_version!r} in {p}; "
            f"expected {LINKSTATS_PROVENANCE_SCHEMA_VERSION!r}."
        )
    return raw


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


def _check_declared_digest_linkstats_expectation(
    *,
    runtime_reference: consist.AdmissionReference,
    expectation: DeclaredDigestExpectation,
) -> tuple[consist.AdmissionReport, dict[str, object] | None]:
    """Evaluate a DeclaredDigestExpectation and optional provenance sidecar."""
    report = consist.check_admission_reference_expected_identity(
        reference=runtime_reference,
        expectation=consist.DeclaredDigestExpectation(
            identity=consist.FileIdentity.parse(expectation.identity),
            source_label=expectation.source_label,
            source_uri=expectation.source_uri,
        ),
    )

    sidecar_path: Path | None = None
    explicit_sidecar = expectation.provenance_sidecar_path is not None
    if explicit_sidecar:
        sidecar_path = Path(expectation.provenance_sidecar_path).expanduser()
    else:
        default_sidecar = linkstats_provenance_sidecar_path(
            runtime_reference.execution_path
        )
        if default_sidecar.is_file():
            sidecar_path = default_sidecar

    if sidecar_path is None:
        return report, None

    try:
        sidecar_payload = read_linkstats_provenance_sidecar(sidecar_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sidecar_label = "Configured" if explicit_sidecar else "Co-located"
        report = report.model_copy(
            update={
                "outcome": (
                    "unverified" if report.outcome == "verified" else report.outcome
                ),
                "observations": (*report.observations, "sidecar_unreadable"),
                "reason": (
                    f"{sidecar_label} provenance sidecar could not be read: {exc}"
                    if report.outcome == "verified"
                    else report.reason
                ),
            }
        )
        return report, None

    sidecar_identity = sidecar_payload.get("identity")
    if sidecar_identity != expectation.identity or (
        report.outcome == "verified"
        and report.observed_artifact_id is not None
        and sidecar_identity != report.observed_artifact_id
    ):
        report = report.model_copy(
            update={
                "outcome": (
                    "mismatched" if report.outcome == "verified" else report.outcome
                ),
                "observations": (*report.observations, "sidecar_identity_mismatched"),
                "reason": (
                    "Provenance sidecar identity does not match expected or observed linkstats digest."
                    if report.outcome == "verified"
                    else report.reason
                ),
            }
        )
    return report, sidecar_payload


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

    expectation = getattr(linkstats_config, "expectation", None)
    sidecar_payload: dict[str, object] | None = None

    if isinstance(expectation, DeclaredDigestExpectation):
        runtime_reference = consist.AdmissionReference(
            artifact_key=None,
            input_role=_LINKSTATS_ROLE,
            config_key=launch_reference.config_key,
            config_reference_key=launch_reference.config_key,
            raw_config_value=launch_reference.raw_value,
            canonical_value=launch_reference.canonical_value,
            configured_path=launch_reference.configured_path,
            execution_path=launch_reference.execution_path,
            consumer_path=launch_reference.container_path,
        )
        report, sidecar_payload = _check_declared_digest_linkstats_expectation(
            runtime_reference=runtime_reference,
            expectation=expectation,
        )
    else:
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
        source_tracker_db = (
            expectation.source_tracker_db
            if isinstance(expectation, PriorRunExpectation)
            else None
        )
        expected_direction = (
            expectation.expected_direction
            if isinstance(expectation, PriorRunExpectation)
            else "any"
        )
        if source_tracker_db is not None or expected_direction != "any":
            report = consist.check_admission_reference_prior_run_expectation(
                tracker=tracker,
                reference=runtime_reference,
                expectation=consist.PriorRunExpectation(
                    expected_run_id=str(linkstats_config.expected_run_id),
                    artifact_key=str(linkstats_config.artifact_key),
                    expected_direction=expected_direction,
                    source_tracker_db=source_tracker_db,
                    expected_bytes_path=linkstats_config.expected_bytes_path,
                    input_role=_LINKSTATS_ROLE,
                ),
            )
        else:
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
        sidecar_provenance=sidecar_payload,
    )

    if report.outcome == "verified":
        return report
    observations = tuple(getattr(report, "observations", ()) or ())
    obs_suffix = f" ({', '.join(observations)})" if observations else ""
    if linkstats_config.mode == "strict":
        raise BeamInputAdmissionError(
            "BEAM linkstats admission rejected before execution: "
            f"{report.outcome}{obs_suffix}"
        )
    logger.warning(
        "BEAM linkstats admission recorded %s%s and will continue because mode=warn",
        report.outcome,
        obs_suffix,
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
    sidecar_provenance: Mapping[str, object] | None = None,
) -> None:
    """Write deterministic sidecar and run metadata before policy can reject."""
    payload = json.loads(report.canonical_json())
    if sidecar_provenance is not None:
        payload["sidecar_provenance"] = dict(sidecar_provenance)
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    report_path = Path(report_dir) / "admission" / "beam-linkstats-warmstart.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(serialized + "\n", encoding="utf-8")
    admission_reports = dict(existing_admission_reports)
    admission_reports[_LINKSTATS_ROLE] = payload
    metadata_logger.log_meta(admission_reports=admission_reports)
