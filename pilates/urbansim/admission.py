"""PILATES policy for admitting the bootstrap UrbanSim datastore."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import consist

from pilates.config import PilatesConfig
from pilates.config.models import DeclaredDigestExpectation
from pilates.urbansim.postprocessor import get_usim_datastore_fname
from pilates.utils.admission_helpers import (
    AdmissionMetadataLogger,
    persist_admission_report,
)


logger = logging.getLogger(__name__)
_URBANSIM_DATASTORE_BASE_ROLE = "usim_datastore_base_h5"
_URBANSIM_DATASTORE_REPORT_FILENAME = "usim-datastore-base-h5.json"


class UrbanSimInputAdmissionError(RuntimeError):
    """Raised when configured UrbanSim input-admission policy rejects a file."""


def preflight_bootstrap_urbansim_datastore_admission(
    *,
    settings: PilatesConfig,
    metadata_logger: AdmissionMetadataLogger,
    workspace_path: Path,
    report_dir: Path,
    existing_admission_reports: Mapping[str, object] | None = None,
) -> consist.admission.AdmissionReport | None:
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
    report = consist.admission.check_admission_reference_expected_identity(
        reference=consist.admission.AdmissionReference(
            artifact_key=None,
            input_role=_URBANSIM_DATASTORE_BASE_ROLE,
            execution_path=execution_path,
        ),
        expectation=consist.admission.DeclaredDigestExpectation(
            identity=consist.admission.FileIdentity.parse(expectation.identity),
            source_label=expectation.source_label,
            source_uri=expectation.source_uri,
        ),
    )
    persist_admission_report(
        report=report,
        report_dir=report_dir,
        report_filename=_URBANSIM_DATASTORE_REPORT_FILENAME,
        role=_URBANSIM_DATASTORE_BASE_ROLE,
        metadata_logger=metadata_logger,
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
