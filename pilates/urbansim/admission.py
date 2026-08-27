"""PILATES policy for admitting the bootstrap UrbanSim datastore."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import consist

from pilates.config import PilatesConfig
from pilates.config.models import DeclaredDigestExpectation
from pilates.urbansim.postprocessor import get_usim_datastore_fname


logger = logging.getLogger(__name__)
_URBANSIM_DATASTORE_BASE_ROLE = "usim_datastore_base_h5"


class UrbanSimInputAdmissionError(RuntimeError):
    """Raised when configured UrbanSim input-admission policy rejects a file."""


class AdmissionMetadataLogger(Protocol):
    """Minimal active-run metadata capability required for admission evidence."""

    def log_meta(self, **metadata: object) -> None: ...


@dataclass(frozen=True)
class _LegacyDeclaredDigestReport:
    """Compatibility report for Consist releases before declared-digest admission."""

    outcome: str
    input_role: str
    artifact_key: None
    execution_path: str
    physical_target_path: str
    expected_source: str
    expected_run_id: None
    expected_artifact_id: str
    observed_artifact_id: str | None
    observations: tuple[str, ...]

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "artifact_key": self.artifact_key,
                "execution_path": self.execution_path,
                "expected_artifact_id": self.expected_artifact_id,
                "expected_run_id": self.expected_run_id,
                "expected_source": self.expected_source,
                "input_role": self.input_role,
                "observed_artifact_id": self.observed_artifact_id,
                "observations": self.observations,
                "outcome": self.outcome,
                "physical_target_path": self.physical_target_path,
            },
            separators=(",", ":"),
            sort_keys=True,
        )


def _check_declared_digest(
    *, execution_path: Path, expectation: DeclaredDigestExpectation
) -> consist.AdmissionReport | _LegacyDeclaredDigestReport:
    """Use Consist's declared-digest API, with a narrow pre-release fallback."""
    try:
        checker = consist.check_admission_reference_expected_identity
    except AttributeError:
        return _check_declared_digest_legacy(
            execution_path=execution_path, expectation=expectation
        )

    reference = consist.AdmissionReference(
        artifact_key=None,
        input_role=_URBANSIM_DATASTORE_BASE_ROLE,
        execution_path=execution_path,
    )
    return checker(
        reference=reference,
        expectation=consist.DeclaredDigestExpectation(
            identity=consist.FileIdentity.parse(expectation.identity),
            source_label=expectation.source_label,
            source_uri=expectation.source_uri,
        ),
    )


def _check_declared_digest_legacy(
    *, execution_path: Path, expectation: DeclaredDigestExpectation
) -> _LegacyDeclaredDigestReport:
    """Preserve declared-digest semantics when the installed Consist is older."""
    observed_identity: str | None = None
    if not execution_path.is_file():
        outcome = "unreadable"
        observations = ("file_unreadable",)
    else:
        try:
            observed_identity = "sha256:file:" + hashlib.sha256(
                execution_path.read_bytes()
            ).hexdigest()
        except OSError:
            outcome = "unreadable"
            observations = ("file_unreadable",)
        else:
            outcome = (
                "verified"
                if observed_identity == expectation.identity
                else "mismatched"
            )
            observations = ("matched",) if outcome == "verified" else ("mismatched",)

    return _LegacyDeclaredDigestReport(
        outcome=outcome,
        input_role=_URBANSIM_DATASTORE_BASE_ROLE,
        artifact_key=None,
        execution_path=str(execution_path),
        physical_target_path=str(execution_path.resolve()),
        expected_source="declared_digest",
        expected_run_id=None,
        expected_artifact_id=str(expectation.identity),
        observed_artifact_id=observed_identity,
        observations=observations,
    )


def preflight_bootstrap_urbansim_datastore_admission(
    *,
    settings: PilatesConfig,
    metadata_logger: AdmissionMetadataLogger,
    workspace_path: Path,
    report_dir: Path,
    existing_admission_reports: Mapping[str, object] | None = None,
) -> consist.AdmissionReport | None:
    """Check the bootstrap-staged UrbanSim datastore when configured.

    ``existing_admission_reports`` is supplied by the active run owner because
    the metadata logger intentionally only writes metadata; it does not assume
    a tracker-specific read API.
    """
    urbansim_config = settings.urbansim
    if urbansim_config is None or urbansim_config.admission is None:
        return None
    policy = urbansim_config.admission.initial_datastore
    if policy is None:
        return None

    expectation = policy.expectation
    if not isinstance(expectation, DeclaredDigestExpectation):
        raise TypeError(
            "Bootstrap UrbanSim datastore admission requires a declared_digest expectation"
        )

    execution_path = (
        Path(workspace_path)
        / urbansim_config.local_mutable_data_folder
        / get_usim_datastore_fname(settings, io="input")
    )
    report = _check_declared_digest(
        execution_path=execution_path,
        expectation=expectation,
    )
    _persist_admission_report(
        report_dir=report_dir,
        metadata_logger=metadata_logger,
        report=report,
        existing_admission_reports=existing_admission_reports,
    )

    if report.outcome == "verified":
        return report
    if policy.mode == "strict":
        raise UrbanSimInputAdmissionError(
            "UrbanSim bootstrap datastore admission rejected before execution: "
            f"{report.outcome}"
        )
    logger.warning(
        "UrbanSim bootstrap datastore admission recorded %s and will continue "
        "because mode=warn",
        report.outcome,
    )
    return report


def _persist_admission_report(
    *,
    report_dir: Path,
    metadata_logger: AdmissionMetadataLogger,
    report: consist.AdmissionReport,
    existing_admission_reports: Mapping[str, object] | None,
) -> None:
    """Write deterministic evidence and merge it into active-run metadata."""
    canonical_report = report.canonical_json()
    payload = json.loads(canonical_report)
    report_path = Path(report_dir) / "admission" / "usim-datastore-base-h5.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonical_report + "\n", encoding="utf-8")
    admission_reports = dict(existing_admission_reports or {})
    admission_reports[_URBANSIM_DATASTORE_BASE_ROLE] = payload
    metadata_logger.log_meta(admission_reports=admission_reports)
