"""PILATES policy for admitting ActivitySim bootstrap OMX skims."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Protocol

import consist
from consist.core.persistence import DatabaseManager
from consist.models.artifact import Artifact
from consist.models.run import Run, RunArtifactLink
from sqlmodel import col, select

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
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


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


def _sha256_file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:file:{digest.hexdigest()}"


def _resolve_artifact_expected_identity(
    artifact: Artifact,
    *,
    expected_bytes_path: str | None,
    candidate_path: Path,
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve a canonical sha256:file:<hex> identity from a persisted Artifact."""
    meta = artifact.meta if isinstance(artifact.meta, Mapping) else {}
    explicit_identity = meta.get("admission_file_identity")
    if isinstance(explicit_identity, str) and explicit_identity.startswith(
        "sha256:file:"
    ):
        return explicit_identity, ()

    semantics = meta.get("hash_semantics")
    if (
        isinstance(semantics, Mapping)
        and semantics.get("algorithm") == "sha256"
        and semantics.get("kind") == "file"
        and semantics.get("digest_contract") == "raw_file_bytes"
        and artifact.hash
        and _SHA256_HEX.fullmatch(artifact.hash)
    ):
        return f"sha256:file:{artifact.hash}", ()

    if expected_bytes_path is not None:
        eb_path = Path(expected_bytes_path).expanduser()
        if not eb_path.is_file():
            return None, ("expected_bytes_unreadable",)
        try:
            if (
                candidate_path.exists()
                and eb_path.resolve() == candidate_path.resolve()
            ):
                return None, ("expected_bytes_aliases_candidate",)
            eb_identity = _sha256_file_identity(eb_path)
        except OSError:
            return None, ("expected_bytes_unreadable",)
        if (
            artifact.hash
            and _SHA256_HEX.fullmatch(artifact.hash)
            and eb_identity != f"sha256:file:{artifact.hash}"
        ):
            return None, ("expected_bytes_mismatched_artifact_hash",)
        return eb_identity, ()

    if artifact.hash and _SHA256_HEX.fullmatch(artifact.hash):
        return f"sha256:file:{artifact.hash}", ()

    return None, ("expected_artifact_identity_unverified",)


def _check_prior_run_expectation(
    *,
    tracker: consist.Tracker | None,
    execution_path: Path,
    expectation: PriorRunExpectation,
) -> consist.AdmissionReport:
    """Evaluate a PriorRunExpectation against same-DB or external source_tracker_db."""
    if (
        expectation.source_tracker_db is None
        and expectation.direction == "input"
        and tracker is not None
    ):
        reference = consist.AdmissionReference(
            artifact_key=str(expectation.artifact_key),
            input_role=_ASIM_INITIAL_SKIMS_ROLE,
            execution_path=execution_path,
        )
        return consist.check_admission_reference(
            tracker,
            reference=reference,
            expected_run_id=str(expectation.expected_run_id),
            expected_bytes_path=expectation.expected_bytes_path,
        )

    db: DatabaseManager | None = None
    if expectation.source_tracker_db is not None:
        db_path = Path(expectation.source_tracker_db).expanduser()
        if not db_path.is_file():
            observed_id: str | None = None
            if execution_path.is_file():
                try:
                    observed_id = _sha256_file_identity(execution_path)
                except OSError:
                    pass
            return consist.AdmissionReport(
                outcome="unverified",
                input_role=_ASIM_INITIAL_SKIMS_ROLE,
                artifact_key=expectation.artifact_key,
                execution_path=str(execution_path),
                physical_target_path=str(execution_path.resolve(strict=False)),
                expected_source="prior_run",
                expected_run_id=expectation.expected_run_id,
                observed_artifact_id=observed_id,
                observations=("source_tracker_db_unreadable",),
                reason=f"Configured source_tracker_db does not exist: {db_path}",
            )
        db = DatabaseManager(str(db_path))
    elif tracker is not None:
        db = tracker.db

    if db is None:
        raise RuntimeError(
            "ActivitySim prior-run skim admission requires a Consist tracker database "
            "or an explicit source_tracker_db."
        )

    if not execution_path.is_file():
        return consist.AdmissionReport(
            outcome="unreadable",
            input_role=_ASIM_INITIAL_SKIMS_ROLE,
            artifact_key=expectation.artifact_key,
            execution_path=str(execution_path),
            physical_target_path=str(execution_path.resolve(strict=False)),
            expected_source="prior_run",
            expected_run_id=expectation.expected_run_id,
            observations=("file_unreadable",),
            reason="Admission requires a readable regular file.",
        )

    try:
        observed_identity = _sha256_file_identity(execution_path)
    except OSError as exc:
        return consist.AdmissionReport(
            outcome="unreadable",
            input_role=_ASIM_INITIAL_SKIMS_ROLE,
            artifact_key=expectation.artifact_key,
            execution_path=str(execution_path),
            physical_target_path=str(execution_path.resolve(strict=False)),
            expected_source="prior_run",
            expected_run_id=expectation.expected_run_id,
            observations=("file_unreadable",),
            reason=str(exc),
        )

    with db.session_scope() as session:
        run = session.get(Run, expectation.expected_run_id)
        if run is None:
            return consist.AdmissionReport(
                outcome="unverified",
                input_role=_ASIM_INITIAL_SKIMS_ROLE,
                artifact_key=expectation.artifact_key,
                execution_path=str(execution_path),
                physical_target_path=str(execution_path.resolve(strict=False)),
                expected_source="prior_run",
                expected_run_id=expectation.expected_run_id,
                observed_artifact_id=observed_identity,
                observations=("expected_prior_run_absent",),
            )
        if run.status != "completed":
            return consist.AdmissionReport(
                outcome="unverified",
                input_role=_ASIM_INITIAL_SKIMS_ROLE,
                artifact_key=expectation.artifact_key,
                execution_path=str(execution_path),
                physical_target_path=str(execution_path.resolve(strict=False)),
                expected_source="prior_run",
                expected_run_id=expectation.expected_run_id,
                observed_artifact_id=observed_identity,
                observations=("expected_prior_run_not_completed",),
            )
        statement = (
            select(Artifact)
            .join(
                RunArtifactLink,
                Artifact.id == RunArtifactLink.artifact_id,  # type: ignore[arg-type]
            )
            .where(RunArtifactLink.run_id == expectation.expected_run_id)
            .where(Artifact.key == expectation.artifact_key)
        )
        if expectation.direction in ("input", "output"):
            statement = statement.where(
                RunArtifactLink.direction == expectation.direction
            )
        statement = statement.order_by(col(Artifact.id))
        matches = list(session.exec(statement).all())

    if not matches:
        absent_code = (
            "expected_output_role_absent"
            if expectation.direction == "output"
            else "expected_input_role_absent"
        )
        return consist.AdmissionReport(
            outcome="unverified",
            input_role=_ASIM_INITIAL_SKIMS_ROLE,
            artifact_key=expectation.artifact_key,
            execution_path=str(execution_path),
            physical_target_path=str(execution_path.resolve(strict=False)),
            expected_source="prior_run",
            expected_run_id=expectation.expected_run_id,
            observed_artifact_id=observed_identity,
            observations=(absent_code,),
        )
    if len(matches) != 1:
        ambiguous_code = (
            "expected_output_role_ambiguous"
            if expectation.direction == "output"
            else "expected_input_role_ambiguous"
        )
        return consist.AdmissionReport(
            outcome="unverified",
            input_role=_ASIM_INITIAL_SKIMS_ROLE,
            artifact_key=expectation.artifact_key,
            execution_path=str(execution_path),
            physical_target_path=str(execution_path.resolve(strict=False)),
            expected_source="prior_run",
            expected_run_id=expectation.expected_run_id,
            observed_artifact_id=observed_identity,
            observations=(ambiguous_code,),
        )

    expected_identity, error_observations = _resolve_artifact_expected_identity(
        matches[0],
        expected_bytes_path=expectation.expected_bytes_path,
        candidate_path=execution_path,
    )
    if expected_identity is None:
        return consist.AdmissionReport(
            outcome="unverified",
            input_role=_ASIM_INITIAL_SKIMS_ROLE,
            artifact_key=expectation.artifact_key,
            execution_path=str(execution_path),
            physical_target_path=str(execution_path.resolve(strict=False)),
            expected_source="prior_run",
            expected_run_id=expectation.expected_run_id,
            observed_artifact_id=observed_identity,
            observations=error_observations,
        )

    matched = observed_identity == expected_identity
    return consist.AdmissionReport(
        outcome="verified" if matched else "mismatched",
        input_role=_ASIM_INITIAL_SKIMS_ROLE,
        artifact_key=expectation.artifact_key,
        execution_path=str(execution_path),
        physical_target_path=str(execution_path.resolve(strict=False)),
        expected_source="prior_run",
        expected_run_id=expectation.expected_run_id,
        expected_artifact_id=expected_identity,
        observed_artifact_id=observed_identity,
        observations=("matched",) if matched else ("mismatched",),
        reason=(
            None
            if matched
            else "Resolved file does not match the expected prior-run artifact."
        ),
    )


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
        if explicit_sidecar:
            report = report.model_copy(
                update={
                    "outcome": "unverified",
                    "observations": (*report.observations, "sidecar_unreadable"),
                    "reason": f"Configured provenance sidecar could not be read: {exc}",
                }
            )
        return report, None

    sidecar_identity = sidecar_payload.get("identity")
    if sidecar_identity != expectation.identity or (
        report.observed_artifact_id is not None
        and sidecar_identity != report.observed_artifact_id
    ):
        report = report.model_copy(
            update={
                "outcome": "mismatched",
                "observations": (*report.observations, "sidecar_identity_mismatched"),
                "reason": (
                    "Provenance sidecar identity does not match expected or observed skim digest."
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
        report = _check_prior_run_expectation(
            tracker=tracker,
            execution_path=execution_path,
            expectation=expectation,
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
