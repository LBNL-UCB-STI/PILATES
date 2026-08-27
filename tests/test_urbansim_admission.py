from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.config.models import PilatesConfig
from pilates.utils import consist_runtime as cr


_DECLARED_IDENTITY = "sha256:file:" + "a" * 64
_ROLE = "usim_datastore_base_h5"


class _MetadataLogger:
    def __init__(self) -> None:
        self.meta: dict[str, object] = {}

    def log_meta(self, **metadata: object) -> None:
        self.meta.update(metadata)


def _settings(
    mode: str | None = "strict", identity: str = _DECLARED_IDENTITY
) -> PilatesConfig:
    admission = (
        None
        if mode is None
        else {
            "initial_datastore": {
                "mode": mode,
                "expectation": {
                    "kind": "declared_digest",
                    "identity": identity,
                    "source_uri": "s3://catalog/baselines/input.h5",
                    "source_label": "catalog baseline",
                },
            }
        }
    )
    return PilatesConfig(
        run={
            "region": "test",
            "scenario": "test",
            "start_year": 2020,
            "end_year": 2021,
            "output_directory": "/tmp/output",
            "output_run_name": "test-run",
            "models": {
                "land_use": "urbansim",
                "travel": None,
                "activity_demand": None,
                "vehicle_ownership": None,
            },
        },
        shared={
            "geography": {"FIPS": {"county": ["06001"]}, "local_crs": "EPSG:32048"},
            "skims": {"fname": "skims.h5"},
            "database": {"enabled": True, "type": "duckdb", "path": "/tmp/test.duckdb"},
        },
        infrastructure={
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
        urbansim={
            "local_data_input_folder": "usim_input",
            "local_mutable_data_folder": "usim_mutable",
            "client_base_folder": "/app",
            "client_data_folder": "/tmp",
            "input_file_template": "input_{region_id}.h5",
            "input_file_template_year": "input_{region_id}_{year}.h5",
            "output_file_template": "output_{year}.h5",
            "command_template": "echo",
            "region_mappings": {"region_to_region_id": {"test": "123"}},
            "admission": admission,
        },
    )


def _staged_datastore(workspace_path: Path) -> Path:
    staged = workspace_path / "usim_mutable" / "input_123.h5"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"observed")
    return staged


def _identity(contents: bytes) -> str:
    return "sha256:file:" + hashlib.sha256(contents).hexdigest()


@pytest.mark.parametrize(
    ("hashing_strategy", "expected_trusted_identity"),
    [("full", True), ("fast", False)],
)
def test_admitted_datastore_logging_persists_trusted_identity_only_with_full_hashing(
    tmp_path: Path,
    hashing_strategy: str,
    expected_trusted_identity: bool,
) -> None:
    """Admission evidence becomes artifact identity only after normal logging."""
    import consist
    from pilates.urbansim import admission

    h5py = pytest.importorskip("h5py")
    workspace_path = tmp_path / "workspace"
    staged = workspace_path / "usim_mutable" / "input_123.h5"
    staged.parent.mkdir(parents=True)
    with h5py.File(staged, "w") as handle:
        handle.create_dataset("households", data=[1, 2, 3])
    expected_identity = _identity(staged.read_bytes())

    tracker_root = tmp_path / "tracker"
    db_path = tmp_path / "provenance.duckdb"
    tracker = cr.create_tracker(
        settings=SimpleNamespace(
            run=SimpleNamespace(consist_hashing_strategy=hashing_strategy)
        ),
        run_dir=tracker_root,
        db_path=str(db_path),
        mounts={"workspace": str(tmp_path)},
    )
    assert tracker is not None

    run_id = f"admitted-{hashing_strategy}"
    with cr.use_tracker(tracker):
        with tracker.start_run(run_id, model="test") as active_run:
            report = admission.preflight_bootstrap_urbansim_datastore_admission(
                settings=_settings(identity=expected_identity),
                metadata_logger=active_run,
                workspace_path=workspace_path,
                report_dir=tracker_root,
            )
            assert report is not None and report.outcome == "verified"
            assert _ROLE not in tracker.get_run_inputs(run_id)
            cr.log_input(staged, key=_ROLE)

    reloaded = consist.Tracker(
        run_dir=tracker_root,
        db_path=str(db_path),
        mounts={"workspace": str(tmp_path)},
        hashing_strategy=hashing_strategy,
    )
    persisted = reloaded.get_run_inputs(run_id)[_ROLE]

    if expected_trusted_identity:
        assert str(consist.ArtifactIdentity.from_artifact(persisted)) == expected_identity
    else:
        with pytest.raises(ValueError, match="trusted immutable identity"):
            consist.ArtifactIdentity.from_artifact(persisted)


def test_verified_declared_digest_writes_sidecar_and_merges_report_metadata(
    tmp_path: Path,
) -> None:
    from pilates.urbansim import admission

    workspace_path = tmp_path / "workspace"
    staged = _staged_datastore(workspace_path)
    logger = _MetadataLogger()

    report = admission.preflight_bootstrap_urbansim_datastore_admission(
        settings=_settings(identity=_identity(b"observed")),
        metadata_logger=logger,
        workspace_path=workspace_path,
        report_dir=tmp_path / "run-a",
        existing_admission_reports={"another_input": {"outcome": "verified"}},
    )

    assert report.outcome == "verified"
    assert report.artifact_key is None
    assert report.input_role == _ROLE
    assert report.execution_path == str(staged)
    assert report.expected_source == "declared_digest"
    report_path = tmp_path / "run-a" / "admission" / "usim-datastore-base-h5.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["outcome"] == "verified"
    assert logger.meta["admission_reports"] == {
        "another_input": {"outcome": "verified"},
        _ROLE: report_payload,
    }


def test_declared_digest_compatibility_report_preserves_external_identity_semantics(
    tmp_path: Path,
) -> None:
    from pilates.urbansim import admission

    workspace_path = tmp_path / "workspace"
    staged = _staged_datastore(workspace_path)
    report = admission.preflight_bootstrap_urbansim_datastore_admission(
        settings=_settings(identity=_identity(b"observed")),
        metadata_logger=_MetadataLogger(),
        workspace_path=workspace_path,
        report_dir=tmp_path / "run-a",
    )

    assert report is not None
    assert report.outcome == "verified"
    assert report.artifact_key is None
    assert report.expected_source == "declared_digest"
    assert report.expected_run_id is None
    assert report.observed_artifact_id == _identity(staged.read_bytes())


def test_strict_mismatch_writes_evidence_before_rejecting(
    tmp_path: Path,
) -> None:
    from pilates.urbansim import admission

    workspace_path = tmp_path / "workspace"
    _staged_datastore(workspace_path)
    logger = _MetadataLogger()
    with pytest.raises(admission.UrbanSimInputAdmissionError, match="mismatched"):
        admission.preflight_bootstrap_urbansim_datastore_admission(
            settings=_settings("strict"),
            metadata_logger=logger,
            workspace_path=workspace_path,
            report_dir=tmp_path / "run-a",
        )

    report_payload = json.loads(
        (tmp_path / "run-a" / "admission" / "usim-datastore-base-h5.json").read_text(
            encoding="utf-8"
        )
    )
    assert report_payload["outcome"] == "mismatched"
    assert logger.meta["admission_reports"] == {_ROLE: report_payload}


def test_warn_mismatch_records_evidence_and_continues(
    tmp_path: Path,
) -> None:
    from pilates.urbansim import admission

    workspace_path = tmp_path / "workspace"
    _staged_datastore(workspace_path)
    report = admission.preflight_bootstrap_urbansim_datastore_admission(
        settings=_settings("warn"),
        metadata_logger=_MetadataLogger(),
        workspace_path=workspace_path,
        report_dir=tmp_path / "run-a",
    )

    assert report.outcome == "mismatched"


def test_absent_policy_does_not_call_consist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pilates.urbansim import admission

    monkeypatch.setattr(
        admission,
        "_check_declared_digest",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not check")),
    )

    report = admission.preflight_bootstrap_urbansim_datastore_admission(
        settings=_settings(None),
        metadata_logger=_MetadataLogger(),
        workspace_path=tmp_path / "workspace",
        report_dir=tmp_path / "run-a",
    )

    assert report is None


def test_second_report_preserves_preexisting_admission_role(
    tmp_path: Path,
) -> None:
    from pilates.urbansim import admission

    workspace_path = tmp_path / "workspace"
    _staged_datastore(workspace_path)
    logger = _MetadataLogger()
    admission.preflight_bootstrap_urbansim_datastore_admission(
        settings=_settings(identity=_identity(b"observed")),
        metadata_logger=logger,
        workspace_path=workspace_path,
        report_dir=tmp_path / "run-a",
        existing_admission_reports={"beam_linkstats_warmstart": {"outcome": "verified"}},
    )

    report_payload = logger.meta["admission_reports"][_ROLE]
    assert report_payload["outcome"] == "verified"
    assert logger.meta["admission_reports"] == {
        "beam_linkstats_warmstart": {"outcome": "verified"},
        _ROLE: report_payload,
    }
