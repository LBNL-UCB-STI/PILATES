"""Unit tests for ActivitySim bootstrap skim admission and cross-context provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import consist
import h5py
import numpy as np
import pytest

from pilates.activitysim.admission import (
    ActivitySimInputAdmissionError,
    has_initial_skims_admission_policy,
    preflight_activitysim_initial_skims_admission,
)
from pilates.beam.skim_provenance import write_skim_provenance_sidecar
from pilates.config.models import (
    ActivitySimAdmissionConfig,
    AdmissionPolicyConfig,
    DeclaredDigestExpectation,
    FullSkimsCreatorConfig,
    PriorRunExpectation,
    admission_policy_fingerprint,
)
from pilates.utils.consist_config import build_activitysim_identity_config


def _write_omx_file(
    path: Path,
    *,
    with_provenance_attrs: bool = True,
    matrix_value: float = 1.0,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        if with_provenance_attrs:
            h5.attrs["OMX_VERSION"] = np.bytes_("0.2")
            h5.attrs["SHAPE"] = np.array([2, 2], dtype=np.int32)
            h5.attrs["data_model_version"] = "beam_omx_v1"
            h5.attrs["skim_matrix_version"] = "asim_omx_v1"
            h5.attrs["beam_run_id"] = "abc1234"
            h5.attrs["created_at"] = "2026-10-14T12:00:00Z"
        data_grp = h5.create_group("data")
        data_grp.create_dataset(
            "SOV_TIME__AM", data=np.full((2, 2), matrix_value, dtype=np.float64)
        )
        lookup_grp = h5.create_group("lookup")
        lookup_grp.create_dataset(
            "zone_number", data=np.array([101, 102], dtype=np.int32)
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:file:{digest}"


def _make_settings(
    *,
    initial_skims_policy: AdmissionPolicyConfig | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="test_region"),
        shared=SimpleNamespace(skims={"fname": "skims.omx"}),
        beam=SimpleNamespace(
            local_mutable_data_folder="beam/mutable",
            full_skim=FullSkimsCreatorConfig(skims_kind="activitySimOmx"),
        ),
        activitysim=SimpleNamespace(
            household_sample_size=0,
            chunk_size=1000,
            num_processes=4,
            file_format="parquet",
            warm_start_activities=False,
            replan_iters=0,
            replan_hh_samp_size=0,
            replan_after="non_mandatory_tour_scheduling",
            random_seed=42,
            database=SimpleNamespace(enabled=False, use_processed_data=True, year=None),
            local_mutable_data_folder="activitysim/data",
            admission=(
                ActivitySimAdmissionConfig(initial_skims=initial_skims_policy)
                if initial_skims_policy is not None
                else None
            ),
        ),
    )


def test_unconfigured_initial_skims_admission_is_noop(tmp_path: Path) -> None:
    """Scenario 1: Unconfigured initial_skims policy returns None without side effects."""
    settings = _make_settings(initial_skims_policy=None)
    assert has_initial_skims_admission_policy(settings) is False

    logger = MagicMock()
    report = preflight_activitysim_initial_skims_admission(
        settings=settings,
        metadata_logger=logger,
        workspace_path=tmp_path / "workspace",
        report_dir=tmp_path / "archive",
    )
    assert report is None
    assert not (tmp_path / "archive" / "admission" / "asim-omx-skims.json").exists()
    logger.log_meta.assert_not_called()


def test_declared_digest_admission_with_sidecar_and_legacy_omx(tmp_path: Path) -> None:
    """Scenario 2: DeclaredDigestExpectation verifies raw OMX bytes, sidecar, and legacy OMX."""
    workspace_path = tmp_path / "workspace"
    report_dir = tmp_path / "archive"
    skims_path = workspace_path / "activitysim" / "data" / "skims.omx"

    # Legacy OMX without custom root attributes
    identity = _write_omx_file(skims_path, with_provenance_attrs=False)
    sidecar_path, _ = write_skim_provenance_sidecar(
        skims_path,
        producer_step="beam_full_skim",
        region="test_region",
        year=2020,
        iteration=0,
    )

    policy = AdmissionPolicyConfig(
        mode="strict",
        expectation=DeclaredDigestExpectation(
            identity=identity,
            source_label="standalone full_skim baseline",
            provenance_sidecar_path=str(sidecar_path),
        ),
    )
    settings = _make_settings(initial_skims_policy=policy)
    logger = MagicMock()

    report = preflight_activitysim_initial_skims_admission(
        settings=settings,
        metadata_logger=logger,
        workspace_path=workspace_path,
        report_dir=report_dir,
    )
    assert report is not None
    assert report.outcome == "verified"

    persisted_path = report_dir / "admission" / "asim-omx-skims.json"
    assert persisted_path.is_file()
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert persisted["outcome"] == "verified"
    assert persisted["skim_inspection"]["skim_format"] == "omx"
    assert persisted["skim_inspection"]["has_embedded_provenance"] is False
    assert persisted["sidecar_provenance"]["identity"] == identity


def test_declared_digest_sidecar_mismatch_rejects_in_strict_mode(
    tmp_path: Path,
) -> None:
    """Scenario 2b: DeclaredDigestExpectation rejects when sidecar identity disagrees."""
    workspace_path = tmp_path / "workspace"
    report_dir = tmp_path / "archive"
    skims_path = workspace_path / "activitysim" / "data" / "skims.omx"
    identity = _write_omx_file(skims_path, with_provenance_attrs=True)

    bad_sidecar = tmp_path / "bad_skims.omx.provenance.json"
    bad_sidecar.write_text(
        json.dumps(
            {
                "schema_version": "pilates.skim_provenance.v1",
                "identity": "sha256:file:" + ("0" * 64),
            }
        ),
        encoding="utf-8",
    )

    policy = AdmissionPolicyConfig(
        mode="strict",
        expectation=DeclaredDigestExpectation(
            identity=identity,
            provenance_sidecar_path=str(bad_sidecar),
        ),
    )
    settings = _make_settings(initial_skims_policy=policy)

    with pytest.raises(
        ActivitySimInputAdmissionError, match="sidecar_identity_mismatched"
    ):
        preflight_activitysim_initial_skims_admission(
            settings=settings,
            metadata_logger=MagicMock(),
            workspace_path=workspace_path,
            report_dir=report_dir,
        )


def test_prior_run_same_db_output_direction_admission(tmp_path: Path) -> None:
    """Scenario 3: PriorRunExpectation in the same provenance.duckdb with direction='output'."""
    db_path = tmp_path / "shared_provenance.duckdb"
    run_dir = tmp_path / "tracker_runs"
    tracker = consist.Tracker(run_dir=run_dir, db_path=db_path)

    producer_skim = tmp_path / "producer" / "skims.omx"
    _write_omx_file(producer_skim, with_provenance_attrs=True, matrix_value=2.5)

    with tracker.start_run(
        "beam_full_skim_producer_run",
        model="beam_full_skim",
    ):
        tracker.log_artifact(
            producer_skim,
            key="final_skims_omx",
            direction="output",
        )

    workspace_path = tmp_path / "consumer_workspace"
    staged_skim = workspace_path / "activitysim" / "data" / "skims.omx"
    staged_skim.parent.mkdir(parents=True, exist_ok=True)
    staged_skim.write_bytes(producer_skim.read_bytes())

    policy = AdmissionPolicyConfig(
        mode="strict",
        expectation=PriorRunExpectation(
            expected_run_id="beam_full_skim_producer_run",
            artifact_key="final_skims_omx",
            direction="output",
        ),
    )
    settings = _make_settings(initial_skims_policy=policy)
    report_dir = tmp_path / "consumer_archive"
    logger = MagicMock()

    report = preflight_activitysim_initial_skims_admission(
        settings=settings,
        metadata_logger=logger,
        tracker=tracker,
        workspace_path=workspace_path,
        report_dir=report_dir,
    )
    assert report is not None
    assert report.outcome == "verified"
    assert report.expected_run_id == "beam_full_skim_producer_run"
    assert report.artifact_key == "final_skims_omx"

    persisted = json.loads(
        (report_dir / "admission" / "asim-omx-skims.json").read_text(encoding="utf-8")
    )
    assert persisted["skim_inspection"]["has_embedded_provenance"] is True
    assert persisted["skim_inspection"]["beam_run_id"] == "abc1234"


def test_prior_run_cross_db_admission(tmp_path: Path) -> None:
    """Scenario 4: PriorRunExpectation across different provenance.duckdb files via source_tracker_db."""
    producer_db_path = tmp_path / "producer_db" / "provenance.duckdb"
    producer_db_path.parent.mkdir(parents=True, exist_ok=True)
    producer_tracker = consist.Tracker(
        run_dir=tmp_path / "producer_runs",
        db_path=producer_db_path,
    )

    producer_skim = tmp_path / "producer_out" / "skims.omx"
    _write_omx_file(producer_skim, with_provenance_attrs=True, matrix_value=3.5)

    with producer_tracker.start_run(
        "external_beam_full_skim_run",
        model="beam_full_skim",
    ):
        producer_tracker.log_artifact(
            producer_skim,
            key="beam_full_skims",
            direction="output",
        )

    consumer_db_path = tmp_path / "consumer_db" / "provenance.duckdb"
    consumer_db_path.parent.mkdir(parents=True, exist_ok=True)
    consumer_tracker = consist.Tracker(
        run_dir=tmp_path / "consumer_runs",
        db_path=consumer_db_path,
    )

    workspace_path = tmp_path / "consumer_workspace"
    staged_skim = workspace_path / "activitysim" / "data" / "skims.omx"
    staged_skim.parent.mkdir(parents=True, exist_ok=True)
    staged_skim.write_bytes(producer_skim.read_bytes())

    policy = AdmissionPolicyConfig(
        mode="strict",
        expectation=PriorRunExpectation(
            expected_run_id="external_beam_full_skim_run",
            artifact_key="beam_full_skims",
            direction="output",
            source_tracker_db=str(producer_db_path),
        ),
    )
    settings = _make_settings(initial_skims_policy=policy)
    report_dir = tmp_path / "consumer_archive"

    report = preflight_activitysim_initial_skims_admission(
        settings=settings,
        metadata_logger=MagicMock(),
        tracker=consumer_tracker,
        workspace_path=workspace_path,
        report_dir=report_dir,
    )
    assert report is not None
    assert report.outcome == "verified"

    # Mutate staged bytes and verify strict rejection
    _write_omx_file(staged_skim, with_provenance_attrs=True, matrix_value=99.0)
    with pytest.raises(ActivitySimInputAdmissionError, match="mismatched"):
        preflight_activitysim_initial_skims_admission(
            settings=settings,
            metadata_logger=MagicMock(),
            tracker=consumer_tracker,
            workspace_path=workspace_path,
            report_dir=report_dir,
        )


def test_activitysim_admission_fingerprint_excludes_host_paths() -> None:
    """admission_policy_fingerprint includes direction when non-default and omits host paths."""
    policy_a = AdmissionPolicyConfig(
        mode="strict",
        expectation=PriorRunExpectation(
            expected_run_id="skim-run-1",
            artifact_key="final_skims_omx",
            direction="output",
            source_tracker_db="/cluster/a/provenance.duckdb",
        ),
    )
    policy_b = AdmissionPolicyConfig(
        mode="strict",
        expectation=PriorRunExpectation(
            expected_run_id="skim-run-1",
            artifact_key="final_skims_omx",
            direction="output",
            source_tracker_db="/cluster/b/provenance.duckdb",
        ),
    )
    assert admission_policy_fingerprint(policy_a) == {
        "mode": "strict",
        "kind": "prior_run",
        "expected_run_id": "skim-run-1",
        "artifact_key": "final_skims_omx",
        "direction": "output",
    }
    assert admission_policy_fingerprint(policy_a) == admission_policy_fingerprint(
        policy_b
    )

    settings = _make_settings(initial_skims_policy=policy_a)
    id_cfg = build_activitysim_identity_config(settings)
    assert id_cfg["admission"]["initial_skims"] == admission_policy_fingerprint(
        policy_a
    )


def test_activitysim_skim_admission_fast_hashing_producer_with_persisted_identity(
    tmp_path: Path,
) -> None:
    """Scenario 5: Fast-hashing producer persists admission_file_identity via _attach_skim_provenance_to_output_artifacts for cross-tracker verification."""
    from pilates.beam.skim_provenance import write_skim_provenance_sidecar
    from pilates.workflows.steps.beam import (
        _attach_skim_provenance_to_output_artifacts,
    )

    producer_db_path = tmp_path / "producer_db" / "provenance.duckdb"
    producer_db_path.parent.mkdir(parents=True, exist_ok=True)
    producer_tracker = consist.Tracker(
        run_dir=tmp_path / "producer_runs",
        db_path=producer_db_path,
        hashing_strategy="fast",
    )

    producer_skim = tmp_path / "producer_out" / "skims.omx"
    _write_omx_file(producer_skim, with_provenance_attrs=True, matrix_value=5.25)

    with producer_tracker.start_run(
        "fast_beam_full_skim_run",
        model="beam_full_skim",
    ):
        art = producer_tracker.log_artifact(
            producer_skim,
            key="beam_full_skims",
            direction="output",
        )
        write_skim_provenance_sidecar(
            producer_skim,
            producer_step="beam_full_skim",
            run_id="fast_beam_full_skim_run",
            region="sfbay",
            iteration=0,
            tracker_db_path=producer_db_path,
        )
        _attach_skim_provenance_to_output_artifacts(
            {"beam_full_skims": art},
            workspace=MagicMock(full_path=tmp_path),
            skim_keys=("beam_full_skims",),
        )

    consumer_db_path = tmp_path / "consumer_db" / "provenance.duckdb"
    consumer_db_path.parent.mkdir(parents=True, exist_ok=True)
    consumer_tracker = consist.Tracker(
        run_dir=tmp_path / "consumer_runs",
        db_path=consumer_db_path,
    )

    workspace_path = tmp_path / "consumer_workspace"
    staged_skim = workspace_path / "activitysim" / "data" / "skims.omx"
    staged_skim.parent.mkdir(parents=True, exist_ok=True)
    staged_skim.write_bytes(producer_skim.read_bytes())

    # Test with expected_direction alias and default direction="any"
    policy_alias = AdmissionPolicyConfig.model_validate(
        {
            "mode": "strict",
            "expectation": {
                "kind": "prior_run",
                "expected_run_id": "fast_beam_full_skim_run",
                "artifact_key": "beam_full_skims",
                "expected_direction": "output",
                "source_tracker_db": str(producer_db_path),
            },
        }
    )
    assert isinstance(policy_alias.expectation, PriorRunExpectation)
    assert policy_alias.expectation.direction == "output"
    assert policy_alias.expectation.expected_direction == "output"

    policy_default = AdmissionPolicyConfig.model_validate(
        {
            "mode": "strict",
            "expectation": {
                "kind": "prior_run",
                "expected_run_id": "fast_beam_full_skim_run",
                "artifact_key": "beam_full_skims",
                "source_tracker_db": str(producer_db_path),
            },
        }
    )
    assert isinstance(policy_default.expectation, PriorRunExpectation)
    assert policy_default.expectation.direction == "any"
    assert admission_policy_fingerprint(policy_default) == {
        "mode": "strict",
        "kind": "prior_run",
        "expected_run_id": "fast_beam_full_skim_run",
        "artifact_key": "beam_full_skims",
    }

    settings = _make_settings(initial_skims_policy=policy_alias)
    report = preflight_activitysim_initial_skims_admission(
        settings=settings,
        metadata_logger=MagicMock(),
        tracker=consumer_tracker,
        workspace_path=workspace_path,
        report_dir=tmp_path / "consumer_archive",
    )
    assert report is not None
    assert report.outcome == "verified"
