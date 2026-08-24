"""PILATES policy for admitting the bootstrap UrbanSim datastore."""

from __future__ import annotations

import json
import logging
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
    reference = consist.AdmissionReference(
        artifact_key=None,
        input_role=_URBANSIM_DATASTORE_BASE_ROLE,
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
