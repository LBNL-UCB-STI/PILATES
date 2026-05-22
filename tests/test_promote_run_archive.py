from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from pilates.config import PilatesConfig
from pilates.runtime.promote_run_archive import (
    _register_verified_run_output_recovery_copies,
    promote_run_to_recovery_roots,
)


def _minimal_config(
    tmp_path: Path, *, recovery_roots: list[str] | None = None
) -> PilatesConfig:
    return PilatesConfig(
        **{
            "run": {
                "region": "test",
                "scenario": "baseline",
                "start_year": 2020,
                "end_year": 2021,
                "use_stubs": False,
                "land_use_freq": 1,
                "travel_model_freq": 1,
                "vehicle_ownership_freq": 1,
                "supply_demand_iters": 1,
                "output_directory": str(tmp_path / "scratch"),
                "output_run_name": "demo-run",
                "local_workspace_root": str(tmp_path / "local"),
                "recovery_archive_roots": recovery_roots or [],
                "enable_archive_copy": True,
                "models": {
                    "land_use": None,
                    "travel": None,
                    "activity_demand": None,
                    "vehicle_ownership": None,
                },
            },
            "shared": {
                "geography": {
                    "FIPS": {"county": ["06001"]},
                    "local_crs": "EPSG:32048",
                },
                "skims": {
                    "zone_type": "taz",
                    "fname": "skims.h5",
                    "geoms_fname": "geoms.geojson",
                    "geoms_index_col": "TAZ",
                },
                "database": {
                    "enabled": True,
                    "type": "duckdb",
                    "path": str(tmp_path / "shared.duckdb"),
                },
            },
            "infrastructure": {
                "container_manager": "docker",
                "singularity_images": {},
                "docker_images": {},
                "docker_config": {"stdout": False, "pull_latest": False},
            },
        }
    )


def _make_tracker(consist, run_dir: Path):
    db_path = run_dir / ".consist" / "provenance.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return consist.Tracker(
        run_dir=run_dir,
        db_path=str(db_path),
        mounts={
            "workspace": str(run_dir),
            "inputs": str(tmp_path := run_dir.parent.parent),
            "scratch": str(run_dir.parent),
        },
        allow_external_paths=True,
    )


def _seed_run_archive(consist, run_dir: Path):
    tracker = _make_tracker(consist, run_dir)
    _seed_output_run(
        tracker,
        run_dir,
        run_name="promote_run",
        model="demo",
        key="demo_output",
        relative_path="outputs/result.txt",
        content="result-data",
    )

    run = tracker.find_runs(model="demo", limit=10)[0]
    return tracker, str(run.id), run_dir / "outputs" / "result.txt"


def _seed_output_run(
    tracker,
    run_dir: Path,
    *,
    run_name: str,
    model: str,
    key: str,
    relative_path: str,
    content: str,
) -> tuple[str, Path]:
    output_path = run_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    (run_dir / "run_state.yaml").write_text("stage: done\n", encoding="utf-8")
    manifest_dir = run_dir / ".workflow" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest_2020_0.yaml").write_text("done: true\n", encoding="utf-8")

    with tracker.start_run(run_name, model=model):
        tracker.log_artifact(str(output_path), key=key, direction="output")

    run = tracker.find_runs(model=model, limit=10)[0]
    return str(run.id), output_path


def test_register_verified_recovery_copies_uses_consist_registration_api(tmp_path):
    source_run_dir = tmp_path / "source-run"
    recovery_run_dir = tmp_path / "recovery-run"
    output = source_run_dir / "outputs" / "result.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("result-data", encoding="utf-8")
    expected_hash = hashlib.sha256(b"result-data").hexdigest()

    calls = []

    class FakeFS:
        @staticmethod
        def get_remappable_relative_path(container_uri):
            assert container_uri == "workspace://outputs/result.txt"
            return "outputs/result.txt"

    class FakeTracker:
        fs = FakeFS()

        def get_run_outputs(self, run_id):
            assert run_id == "run-1"
            return {
                "demo_output": SimpleNamespace(
                    container_uri="workspace://outputs/result.txt"
                )
            }

        def register_run_output_recovery_copies(self, run_id, recovery_root, **kwargs):
            calls.append((run_id, recovery_root, kwargs))
            return SimpleNamespace(
                registered={"demo_output": object()},
                blocked={},
                summary="registered=1",
            )

        def set_artifact_recovery_roots(self, *_args, **_kwargs):
            raise AssertionError("legacy recovery-root setter should not be used")

    registered = _register_verified_run_output_recovery_copies(
        FakeTracker(),
        source_run_dir=source_run_dir,
        recovery_run_dir=recovery_run_dir,
        run_ids=["run-1"],
    )

    assert registered == 1
    assert calls == [
        (
            "run-1",
            str(recovery_run_dir),
            {
                "append": True,
                "content_hashes": {"demo_output": expected_hash},
                "verify": True,
            },
        )
    ]


def test_promote_run_to_recovery_root_copies_run_dir_and_updates_recovery_roots(
    tmp_path,
):
    consist = pytest.importorskip("consist")
    recovery_root = tmp_path / "nfs"
    settings = _minimal_config(tmp_path, recovery_roots=[str(recovery_root)])
    run_dir = Path(settings.run.output_directory) / settings.run.output_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker, run_id, output_path = _seed_run_archive(consist, run_dir)
    promoted = recovery_root / run_dir.name

    try:
        result = promote_run_to_recovery_roots(
            settings,
            archive_run_dir=str(run_dir),
            tracker=tracker,
        )

        assert result.success is True
        assert promoted.exists()
        assert (promoted / "run_state.yaml").exists()
        assert (promoted / ".workflow" / "manifests" / "manifest_2020_0.yaml").exists()
        assert (promoted / ".consist" / "provenance.duckdb").exists()

        source_outputs = tracker.get_run_outputs(run_id)
        assert source_outputs["demo_output"].recovery_roots == [str(promoted.resolve())]

        promoted_tracker = _make_tracker(consist, promoted)
        try:
            promoted_run = promoted_tracker.find_runs(model="demo", limit=10)[0]
            promoted_outputs = promoted_tracker.get_run_outputs(str(promoted_run.id))
            assert promoted_outputs["demo_output"].recovery_roots == [
                str(promoted.resolve())
            ]
        finally:
            promoted_tracker.db.engine.dispose()

        output_path.unlink()
        hydrated = tracker.hydrate_run_outputs(
            run_id,
            target_root=str(tmp_path / "rehydrated"),
            keys=["demo_output"],
        )
        assert hydrated["demo_output"].status == "materialized_from_filesystem"
        assert hydrated["demo_output"].path is not None
        assert hydrated["demo_output"].path.read_text(encoding="utf-8") == "result-data"

        marker = json.loads(
            (run_dir / ".consist" / "recovery_promotion.json").read_text(
                encoding="utf-8"
            )
        )
        assert marker["source_run_dir"] == str(run_dir)
        assert marker["roots"][0]["status"] == "promoted"
        assert (promoted / ".consist" / "recovery_promotion.json").exists()
    finally:
        tracker.db.engine.dispose()


def test_promote_run_to_recovery_root_scopes_seeded_db_merge(tmp_path):
    consist = pytest.importorskip("consist")
    recovery_root = tmp_path / "nfs"
    settings = _minimal_config(tmp_path, recovery_roots=[str(recovery_root)])
    main_run_dir = tmp_path / "main-catalog"
    main_tracker = _make_tracker(consist, main_run_dir)
    main_db_path = main_run_dir / ".consist" / "provenance.duckdb"

    try:
        old_run_id, _old_path = _seed_output_run(
            main_tracker,
            main_run_dir,
            run_name="historical_root",
            model="history",
            key="old_output",
            relative_path="outputs/old-result.txt",
            content="old-result-data",
        )
    finally:
        main_tracker.db.engine.dispose()

    run_dir = Path(settings.run.output_directory) / settings.run.output_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    local_db_path = run_dir / ".consist" / "provenance.duckdb"
    local_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(main_db_path, local_db_path)

    tracker = _make_tracker(consist, run_dir)
    new_run_id, _new_path = _seed_output_run(
        tracker,
        run_dir,
        run_name="new_root",
        model="demo",
        key="new_output",
        relative_path="outputs/new-result.txt",
        content="new-result-data",
    )
    promoted = recovery_root / run_dir.name

    try:
        result = promote_run_to_recovery_roots(
            settings,
            archive_run_dir=str(run_dir),
            tracker=tracker,
            merge_main_db=str(main_db_path),
        )

        assert result.success is True
        assert result.root_run_id == new_run_id
        assert result.scoped_run_ids == [new_run_id]
        assert result.merge_result is not None
        if result.merge_result.get("skipped"):
            assert result.merge_result["reason"] in {
                "consist_db_merge_failed",
                "consist_db_merge_failed_after_retry",
            }
            assert "DETACH" in result.merge_result["error"]
        else:
            assert result.merge_result["merge_result"]["runs_merged"] == [
                new_run_id
            ]
            assert result.merge_result["merge_result"]["runs_skipped"] == []

        old_outputs = tracker.get_run_outputs(old_run_id)
        new_outputs = tracker.get_run_outputs(new_run_id)
        assert str(promoted.resolve()) not in (
            old_outputs["old_output"].recovery_roots or []
        )
        assert new_outputs["new_output"].recovery_roots == [str(promoted.resolve())]

        if not result.merge_result.get("skipped"):
            merged_tracker = _make_tracker(consist, main_run_dir)
            try:
                merged_run_ids = {
                    str(run.id) for run in merged_tracker.find_runs(limit=100)
                }
                assert old_run_id in merged_run_ids
                assert new_run_id in merged_run_ids
            finally:
                merged_tracker.db.engine.dispose()
    finally:
        tracker.db.engine.dispose()


def test_promote_run_to_recovery_roots_is_idempotent(tmp_path):
    consist = pytest.importorskip("consist")
    recovery_root = tmp_path / "nfs"
    settings = _minimal_config(tmp_path, recovery_roots=[str(recovery_root)])
    run_dir = Path(settings.run.output_directory) / settings.run.output_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker, run_id, _output_path = _seed_run_archive(consist, run_dir)
    try:
        first = promote_run_to_recovery_roots(
            settings,
            archive_run_dir=str(run_dir),
            tracker=tracker,
        )
        second = promote_run_to_recovery_roots(
            settings,
            archive_run_dir=str(run_dir),
            tracker=tracker,
        )

        assert first.success is True
        assert second.success is True
        outputs = tracker.get_run_outputs(run_id)
        assert outputs["demo_output"].recovery_roots == [
            str((recovery_root / run_dir.name).resolve())
        ]
    finally:
        tracker.db.engine.dispose()


def test_promote_run_to_recovery_roots_handles_partial_failure(tmp_path):
    consist = pytest.importorskip("consist")
    good_root = tmp_path / "nfs-good"
    bad_root = tmp_path / "not-a-dir"
    bad_root.write_text("occupied", encoding="utf-8")
    settings = _minimal_config(
        tmp_path,
        recovery_roots=[str(good_root), str(bad_root)],
    )
    run_dir = Path(settings.run.output_directory) / settings.run.output_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker, run_id, _output_path = _seed_run_archive(consist, run_dir)
    try:
        result = promote_run_to_recovery_roots(
            settings,
            archive_run_dir=str(run_dir),
            tracker=tracker,
        )

        assert result.success is False
        assert len(result.succeeded_roots) == 1
        assert len(result.failed_roots) == 1
        assert (good_root / run_dir.name).exists()

        outputs = tracker.get_run_outputs(run_id)
        assert outputs["demo_output"].recovery_roots == [
            str((good_root / run_dir.name).resolve())
        ]
    finally:
        tracker.db.engine.dispose()
