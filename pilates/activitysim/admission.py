"""PILATES policy for admitting ActivitySim bootstrap OMX skims."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

import consist

from pilates.beam.skim_provenance import (
    inspect_skim_artifact,
    read_skim_provenance_sidecar,
    skim_provenance_sidecar_path,
)
from pilates.config import PilatesConfig
from pilates.config.models import (
    AdmissionPolicyConfig,
    DeclaredDigestExpectation,
    PriorRunExpectation,
)


logger = logging.getLogger(__name__)
_ASIM_INITIAL_SKIMS_ROLE = "asim_omx_skims"


class ActivitySimInputAdmissionError(RuntimeError):
    """Raised when configured ActivitySim input-admission policy rejects a skim file."""


class AdmissionMetadataLogger(Protocol):
    """Minimal active-run metadata capability required for admission evidence."""

    def log_meta(self, **metadata: object) -> None: ...


def _initial_skims_admission_config(
    settings: PilatesConfig,
) -> AdmissionPolicyConfig | None:
    """Return the optional typed admission policy for initial ActivitySim skims."""
    try:
        asim_config = settings.activitysim
    except AttributeError:
        return None
    if asim_config is None:
        return None
    try:
        admission = asim_config.admission
    except AttributeError:
        return None
    if admission is None:
        return None
    return admission.initial_skims


def has_initial_skims_admission_policy(settings: PilatesConfig) -> bool:
    """Return whether ActivitySim has an active initial_skims admission policy."""
    return _initial_skims_admission_config(settings) is not None


def _check_declared_digest_expectation(
    *,
    execution_path: Path,
    expectation: DeclaredDigestExpectation,
) -> tuple[consist.AdmissionReport, dict[str, Any] | None]:
    """Evaluate a DeclaredDigestExpectation and optional provenance sidecar."""
    reference = consist.AdmissionReference(
        artifact_key=None,
        input_role=_ASIM_INITIAL_SKIMS_ROLE,
        execution_path=execution_path,
    )
    report = consist.check_admission_reference_expected_identity(
        reference=reference,
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
        default_sidecar = skim_provenance_sidecar_path(execution_path)
        if default_sidecar.is_file():
            sidecar_path = default_sidecar

    if sidecar_path is None:
        return report, None

    try:
        sidecar_payload = read_skim_provenance_sidecar(sidecar_path)
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
                    "Provenance sidecar identity does not match expected or observed skim digest."
                    if report.outcome == "verified"
                    else report.reason
                ),
            }
        )
    return report, sidecar_payload


def resolve_bootstrap_activitysim_skims_path(
    *,
    settings: PilatesConfig,
    workspace_path: Path,
    explicit_skims_path: Path | None = None,
) -> Path:
    """Resolve the candidate OMX skims path that ActivitySim bootstrap will consume."""
    if explicit_skims_path is not None:
        return Path(explicit_skims_path)
    asim_cfg = settings.activitysim
    asim_mutable = (
        Path(workspace_path) / asim_cfg.local_mutable_data_folder / "skims.omx"
        if asim_cfg is not None
        else None
    )
    if asim_mutable is not None and asim_mutable.is_file():
        return asim_mutable
    beam_cfg = settings.beam
    skims_fname = (
        settings.shared.skims.get("fname", "skims.omx")
        if isinstance(settings.shared.skims, Mapping)
        else getattr(settings.shared.skims, "fname", "skims.omx")
    )
    if beam_cfg is not None:
        beam_skims = (
            Path(workspace_path)
            / beam_cfg.local_mutable_data_folder
            / settings.run.region
            / skims_fname
        )
        if beam_skims.is_file() or asim_mutable is None:
            return beam_skims
    assert asim_mutable is not None
    return asim_mutable


def preflight_activitysim_initial_skims_admission(
    *,
    settings: PilatesConfig,
    metadata_logger: AdmissionMetadataLogger | None = None,
    tracker: consist.Tracker | None = None,
    workspace_path: Path,
    report_dir: Path,
    existing_admission_reports: Mapping[str, object] | None = None,
    explicit_skims_path: Path | None = None,
) -> consist.AdmissionReport | None:
    """Check initial ActivitySim OMX skims against configured admission policy."""
    policy = _initial_skims_admission_config(settings)
    if policy is None:
        return None

    execution_path = resolve_bootstrap_activitysim_skims_path(
        settings=settings,
        workspace_path=workspace_path,
        explicit_skims_path=explicit_skims_path,
    )

    expectation = policy.expectation
    sidecar_payload: dict[str, Any] | None = None
    if isinstance(expectation, DeclaredDigestExpectation):
        report, sidecar_payload = _check_declared_digest_expectation(
            execution_path=execution_path,
            expectation=expectation,
        )
    elif isinstance(expectation, PriorRunExpectation):
        consist_exp = consist.PriorRunExpectation(
            expected_run_id=expectation.expected_run_id,
            artifact_key=expectation.artifact_key,
            expected_direction=expectation.direction or "any",
            source_tracker_db=expectation.source_tracker_db,
            expected_bytes_path=expectation.expected_bytes_path,
            input_role=_ASIM_INITIAL_SKIMS_ROLE,
        )
        reference = consist.AdmissionReference(
            artifact_key=expectation.artifact_key,
            input_role=_ASIM_INITIAL_SKIMS_ROLE,
            execution_path=execution_path,
        )
        report = consist.check_admission_reference_prior_run_expectation(
            tracker=tracker,
            reference=reference,
            expectation=consist_exp,
        )
        default_sidecar = skim_provenance_sidecar_path(execution_path)
        if default_sidecar.is_file():
            try:
                sidecar_payload = read_skim_provenance_sidecar(default_sidecar)
            except (OSError, ValueError, json.JSONDecodeError):
                sidecar_payload = None
    else:
        raise TypeError(
            f"Unsupported ActivitySim skim expectation: {type(expectation)!r}"
        )

    skim_inspection: dict[str, Any] | None = None
    if execution_path.exists():
        try:
            skim_inspection = inspect_skim_artifact(execution_path)
        except Exception as exc:
            logger.debug(
                "Unable to inspect skim structure for %s: %s", execution_path, exc
            )

    _persist_admission_report(
        report_dir=report_dir,
        metadata_logger=metadata_logger,
        report=report,
        skim_inspection=skim_inspection,
        sidecar_provenance=sidecar_payload,
        existing_admission_reports=existing_admission_reports,
    )

    if report.outcome == "verified":
        return report
    if policy.mode == "strict":
        raise ActivitySimInputAdmissionError(
            "ActivitySim initial skims admission rejected before execution: "
            f"{report.outcome} ({', '.join(report.observations)})"
        )
    logger.warning(
        "ActivitySim initial skims admission recorded %s (%s) and will continue "
        "because mode=warn",
        report.outcome,
        ", ".join(report.observations),
    )
    return report


def _persist_admission_report(
    *,
    report_dir: Path,
    metadata_logger: AdmissionMetadataLogger | None,
    report: consist.AdmissionReport,
    skim_inspection: Mapping[str, Any] | None,
    sidecar_provenance: Mapping[str, Any] | None,
    existing_admission_reports: Mapping[str, object] | None,
) -> None:
    """Write deterministic admission evidence and merge it into active-run metadata."""
    payload = json.loads(report.canonical_json())
    if skim_inspection is not None:
        payload["skim_inspection"] = dict(skim_inspection)
    if sidecar_provenance is not None:
        payload["sidecar_provenance"] = dict(sidecar_provenance)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    report_path = Path(report_dir) / "admission" / "asim-omx-skims.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(serialized + "\n", encoding="utf-8")
    if metadata_logger is not None:
        admission_reports = dict(existing_admission_reports or {})
        admission_reports[_ASIM_INITIAL_SKIMS_ROLE] = payload
        metadata_logger.log_meta(admission_reports=admission_reports)
