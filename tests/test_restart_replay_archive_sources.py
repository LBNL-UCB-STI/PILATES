from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.config import PilatesConfig
from pilates.runtime.promote_run_archive import _archive_db_path
from pilates.utils import coupler_helpers
from pilates.utils.consist_db_snapshot import (
    restore_local_consist_db_from_snapshot,
    snapshot_meta_filename,
)
from pilates.workflows.artifact_keys import ZARR_SKIMS
from pilates.workflows.outputs_base import serialize_step_outputs
from pilates.workflows.stages.supply_demand_resume import (
    _restore_activity_demand_outputs_for_resume,
)
from pilates.workflows.steps import StepOutputsHolder
from tests.workflow_contract_harness import CouplerStub


def _write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _minimal_config(tmp_path: Path) -> PilatesConfig:
    return PilatesConfig(
        **{
            "run": {
                "region": "test",
                "scenario": "baseline",
                "start_year": 2017,
                "end_year": 2018,
                "use_stubs": False,
                "land_use_freq": 1,
                "travel_model_freq": 1,
                "vehicle_ownership_freq": 1,
                "supply_demand_iters": 1,
                "output_directory": str(tmp_path / "scratch"),
                "output_run_name": "demo-run",
                "local_workspace_root": str(tmp_path / "local-root"),
                "consist_db_filename": "provenance.duckdb",
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


class ScratchArchiveArtifact:
    def __init__(
        self,
        *,
        key: str,
        local_path: Path,
        local_run_dir: Path,
        archive_run_dir: Path,
    ) -> None:
        self.id = f"artifact-{key}"
        self.key = key
        self._path = local_path
        self.local_run_dir = local_run_dir
        self.archive_run_dir = archive_run_dir
        rel_path = local_path.relative_to(local_run_dir)
        self.container_uri = f"workspace://{rel_path}"
        self.archive_path = archive_run_dir / rel_path

    @property
    def path(self) -> Path:
        return self._path


class ScratchArchiveTracker:
    def __init__(self, *, run_id: str, outputs: dict[str, ScratchArchiveArtifact]):
        self.run_id = run_id
        self.outputs = outputs
        self.materialized_keys: list[str] = []

    def get_run_outputs(self, run_id: str):
        assert run_id == self.run_id
        return self.outputs

    def materialize_artifact(
        self,
        artifact: ScratchArchiveArtifact,
        *,
        target_root: Path | str,
        source_root: Path | str | None,
        preserve_existing: bool,
        on_missing: str,
        validate_content_hash: str,
    ):
        assert Path(source_root) == artifact.archive_run_dir
        assert preserve_existing is True
        assert on_missing == "warn"
        assert validate_content_hash == "if-present"
        destination = Path(target_root) / artifact._path.relative_to(
            artifact.local_run_dir
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(artifact.archive_path.read_text(encoding="utf-8"))
        self.materialized_keys.append(artifact.key)
        return SimpleNamespace(path=destination, resolvable=True)


def test_restart_restores_activitysim_handoff_from_scratch_archive_before_nfs_promotion(
    monkeypatch,
    tmp_path,
) -> None:
    local_run_dir = tmp_path / "local" / "demo-run"
    archive_run_dir = tmp_path / "scratch-archive" / "demo-run"
    local_run_dir.mkdir(parents=True)
    _write(local_run_dir / "settings.yaml", "run: {}\n")
    _write(local_run_dir / "run_state.yaml", "current_sub_stage: traffic_assignment\n")

    _write(archive_run_dir / ".consist" / "provenance.duckdb", "scratch-db")
    _write(
        archive_run_dir / "run_state.yaml", "current_sub_stage: traffic_assignment\n"
    )

    local_output_dir = local_run_dir / "activitysim" / "output"
    local_iter_dir = local_output_dir / "year-2017-iteration-0"
    local_input_dir = local_output_dir / "inputs-year-2017-iteration-0"
    output_paths = {
        "beam_plans_asim_out": local_iter_dir / "beam_plans.parquet",
        "households_asim_out": local_iter_dir / "households.parquet",
        "persons_asim_out": local_iter_dir / "persons.parquet",
        "asim_input_skims_zarr_archived": local_input_dir / "skims.zarr",
    }
    local_usim = local_run_dir / "urbansim" / "data" / "model_data_2018.h5"
    for key, local_path in output_paths.items():
        archive_path = archive_run_dir / local_path.relative_to(local_run_dir)
        _write(archive_path, f"{key}\n")
    _write(archive_run_dir / local_usim.relative_to(local_run_dir), "usim\n")

    manifest_path = local_run_dir / ".workflow" / "year_2017_iteration_0.yaml"
    archive_manifest = archive_run_dir / manifest_path.relative_to(local_run_dir)
    archive_manifest.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest.write_text(
        yaml.safe_dump(
            {
                "activitysim_run": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "run_id": "activitysim-run-2017-0",
                },
                "activitysim_postprocess": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "outputs": serialize_step_outputs(
                        ActivitySimPostprocessOutputs(
                            usim_datastore_h5=local_usim,
                            asim_output_dir=local_output_dir,
                            processed_outputs=output_paths,
                        )
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    assert not manifest_path.exists()
    assert not any(path.exists() for path in output_paths.values())
    assert not local_usim.exists()
    assert not (tmp_path / "nfs").exists()

    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_run_dir))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_run_dir))

    coupler = CouplerStub()
    remembered_run_ids = []
    scenario = SimpleNamespace(
        remember_restored_run_id=lambda **kwargs: remembered_run_ids.append(kwargs)
    )
    tracker_outputs = {
        "beam_plans_asim_out": ScratchArchiveArtifact(
            key="beam_plans_asim_out",
            local_path=output_paths["beam_plans_asim_out"],
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        ),
        "households_asim_out": ScratchArchiveArtifact(
            key="households_asim_out",
            local_path=output_paths["households_asim_out"],
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        ),
        "persons_asim_out": ScratchArchiveArtifact(
            key="persons_asim_out",
            local_path=output_paths["persons_asim_out"],
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        ),
        ZARR_SKIMS: ScratchArchiveArtifact(
            key=ZARR_SKIMS,
            local_path=output_paths["asim_input_skims_zarr_archived"],
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        ),
    }
    tracker = ScratchArchiveTracker(
        run_id="activitysim-run-2017-0",
        outputs=tracker_outputs,
    )
    monkeypatch.setattr(coupler_helpers.cr, "current_tracker", lambda: tracker)
    restored = _restore_activity_demand_outputs_for_resume(
        scenario=scenario,
        coupler=coupler,
        workspace=SimpleNamespace(full_path=str(local_run_dir)),
        outputs_holder=StepOutputsHolder(),
        state=SimpleNamespace(
            current_year=2017,
            forecast_year=2018,
            current_inner_iter=0,
            file_loc=str(archive_run_dir / "run_state.yaml"),
        ),
        settings=SimpleNamespace(land_use_enabled=False),
        tracker=tracker,
        manifest_path=manifest_path,
    )

    assert restored is not None
    assert not manifest_path.exists()
    assert archive_manifest.exists()
    assert restored["beam_plans_asim_out"] == str(output_paths["beam_plans_asim_out"])
    assert restored["households_asim_out"] == str(output_paths["households_asim_out"])
    assert restored["persons_asim_out"] == str(output_paths["persons_asim_out"])
    assert restored[ZARR_SKIMS] == str(output_paths["asim_input_skims_zarr_archived"])
    for local_path in output_paths.values():
        assert local_path.exists()
    assert not local_usim.exists()
    assert sorted(tracker.materialized_keys) == sorted(tracker_outputs)
    assert remembered_run_ids == [
        {
            "model_name": "activitysim_run",
            "year": 2017,
            "iteration": 0,
            "run_id": "activitysim-run-2017-0",
        }
    ]
    assert coupler.get("beam_plans_asim_out") is tracker_outputs["beam_plans_asim_out"]
    assert coupler.get("households_asim_out") is tracker_outputs["households_asim_out"]
    assert coupler.get("persons_asim_out") is tracker_outputs["persons_asim_out"]
    assert coupler.get(ZARR_SKIMS) is tracker_outputs[ZARR_SKIMS]


def test_snapshot_latest_db_is_accepted_without_direct_archive_db(tmp_path) -> None:
    settings = _minimal_config(tmp_path)
    archive_run_dir = tmp_path / "scratch-archive" / settings.run.output_run_name
    latest_dir = archive_run_dir / ".consist" / "snapshots" / "latest"
    latest_db = _write(latest_dir / "provenance.duckdb", "snapshot-db")
    _write(
        latest_dir / snapshot_meta_filename("provenance.duckdb"),
        '{"snapshot_ts_utc": "2026-01-01T00:00:00Z"}\n',
    )
    local_db = tmp_path / "restart-local" / ".consist" / "provenance.duckdb"

    assert not (archive_run_dir / ".consist" / "provenance.duckdb").exists()

    assert _archive_db_path(settings, archive_run_dir=archive_run_dir) == latest_db
    assert restore_local_consist_db_from_snapshot(
        settings=settings,
        local_db_path=str(local_db),
        archive_run_dir=str(archive_run_dir),
    )
    assert local_db.read_text(encoding="utf-8") == "snapshot-db"
    assert (local_db.parent / snapshot_meta_filename("provenance.duckdb")).read_text(
        encoding="utf-8"
    ) == (latest_dir / snapshot_meta_filename("provenance.duckdb")).read_text(
        encoding="utf-8"
    )
