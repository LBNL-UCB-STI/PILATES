"""PILATES policy for admitting the bootstrap UrbanSim datastore."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import consist

from pilates.config import PilatesConfig
from pilates.config.models import DeclaredDigestExpectation
from pilates.urbansim.postprocessor import get_usim_datastore_fname
from pilates.urbansim.preprocessor import (
    resolve_urbansim_initial_datastore_source_path,
)
from pilates.utils.admission_helpers import (
    AdmissionMetadataLogger,
    parse_expected_file_digest,
    persist_admission_report,
    resolve_local_source_uri,
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
    """Check the initial UrbanSim datastore when configured.

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

    workspace_candidate = (
        Path(workspace_path)
        / urbansim_config.local_mutable_data_folder
        / get_usim_datastore_fname(settings, io="input")
    )
    if workspace_candidate.is_file():
        execution_path = workspace_candidate
    else:
        source_uri_path = resolve_local_source_uri(expectation.source_uri)
        static_source = resolve_urbansim_initial_datastore_source_path(settings)
        if source_uri_path is not None and source_uri_path.is_file():
            execution_path = source_uri_path
        elif static_source is not None and static_source.is_file():
            execution_path = static_source
        elif source_uri_path is not None:
            execution_path = source_uri_path
        elif static_source is not None:
            execution_path = static_source
        else:
            execution_path = workspace_candidate
    report = consist.admission.check_admission_reference_expected_identity(
        reference=consist.admission.AdmissionReference(
            artifact_key=None,
            input_role=_URBANSIM_DATASTORE_BASE_ROLE,
            execution_path=execution_path,
        ),
        expectation=consist.admission.DeclaredDigestExpectation(
            identity=parse_expected_file_digest(expectation.identity),
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
