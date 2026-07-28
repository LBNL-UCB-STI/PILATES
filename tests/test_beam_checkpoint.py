from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.workflows.beam_checkpoint import (
    HistoricalOutputRequest,
    PinnedClosureMember,
    assert_committed_beam_run,
    hydrate_pinned_closure,
    mark_beam_postprocess_in_progress,
    publish_beam_run_checkpoint,
    read_beam_run_checkpoint,
    snapshot_and_publish_beam_run_checkpoint,
    validate_pinned_closure_snapshot,
    verify_archive_visible_pinned_closure_bytes,
    verify_archive_visible_recovery_bytes,
)


class _Hydration(dict):
    def __init__(self, source_run_id: str, items):
        super().__init__(items)
        self.source_run_id = source_run_id


def test_historical_output_request_is_owned_by_beam_checkpoint():
    assert HistoricalOutputRequest.__module__ == "pilates.workflows.beam_checkpoint"


def test_beam_checkpoint_is_atomic_and_becomes_nonrestartable(tmp_path):
    checkpoint = publish_beam_run_checkpoint(
        archive_run_dir=tmp_path,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        snapshot_ref=".consist/restart/checkpoints/pinned/tracker.duckdb",
        skim_variant="full",
        output_requests=(
            HistoricalOutputRequest(
                "linkstats", tmp_path / "beam" / "0.linkstats", True
            ),
        ),
    )

    assert checkpoint.producer_run_id == "beam-run-1"
    assert read_beam_run_checkpoint(tmp_path) == checkpoint

    mark_beam_postprocess_in_progress(tmp_path, checkpoint)
    assert read_beam_run_checkpoint(tmp_path) is None


def test_beam_checkpoint_round_trips_a_pinned_multi_producer_closure(tmp_path):
    closure = (
        PinnedClosureMember(
            member_id="beam-events",
            role="beam_events",
            producer_run_id="beam-run-1",
            output_key="events_parquet_2021_0",
            artifact_identity="beam-events-hash",
            artifact_kind="file",
            driver="parquet",
            destination=tmp_path / "beam" / "0.events.parquet",
            required=True,
        ),
        PinnedClosureMember(
            member_id="asim-zarr",
            role="zarr_skims",
            producer_run_id="asim-run-1",
            output_key="zarr_skims",
            artifact_identity="zarr-tree-hash",
            artifact_kind="directory",
            driver="zarr",
            destination=tmp_path / "asim" / "skims.zarr",
            required=True,
        ),
    )

    checkpoint = publish_beam_run_checkpoint(
        archive_run_dir=tmp_path,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        snapshot_ref=".consist/restart/checkpoints/pinned/tracker.duckdb",
        skim_variant="full",
        output_requests=(),
        closure_members=closure,
    )

    assert checkpoint.closure_members == closure
    assert read_beam_run_checkpoint(tmp_path) == checkpoint


def _closure_artifact(
    key: str,
    identity: str,
    *,
    driver: str,
    artifact_kind: str,
):
    return SimpleNamespace(
        key=key,
        hash=identity,
        driver=driver,
        meta={"directory_artifact": artifact_kind == "directory"},
    )


def test_pinned_closure_hydrates_multi_producer_files_and_manifest_zarr(tmp_path):
    events_destination = tmp_path / "workspace" / "beam" / "0.events.parquet"
    zarr_destination = tmp_path / "workspace" / "activitysim" / "cache" / "skims.zarr"
    members = (
        PinnedClosureMember(
            member_id="beam-events",
            role="events_parquet_2021_0",
            producer_run_id="beam-run-1",
            output_key="events_parquet_2021_0",
            artifact_identity="events-hash",
            artifact_kind="file",
            driver="parquet",
            destination=events_destination,
            required=True,
        ),
        PinnedClosureMember(
            member_id="asim-zarr",
            role="zarr_skims",
            producer_run_id="asim-run-1",
            output_key="zarr_skims",
            artifact_identity="zarr-hash",
            artifact_kind="directory",
            driver="zarr",
            destination=zarr_destination,
            required=True,
        ),
    )
    artifacts = {
        "beam-run-1": {
            "events_parquet_2021_0": _closure_artifact(
                "events_parquet_2021_0",
                "events-hash",
                driver="parquet",
                artifact_kind="file",
            )
        },
        "asim-run-1": {
            "zarr_skims": _closure_artifact(
                "zarr_skims",
                "zarr-hash",
                driver="zarr",
                artifact_kind="directory",
            )
        },
    }

    class _Tracker:
        def __init__(self):
            self.hydrate_calls = []

        def get_run(self, run_id):
            return SimpleNamespace(id=run_id, status="completed")

        def get_run_outputs(self, run_id):
            return artifacts[run_id]

        def hydrate_run_outputs_to_destinations(self, run_id, **kwargs):
            self.hydrate_calls.append((run_id, kwargs))
            hydrated = {}
            for key, destination in kwargs["destinations_by_key"].items():
                artifact = artifacts[run_id][key]
                if artifact.meta["directory_artifact"]:
                    destination.mkdir(parents=True)
                    (destination / ".zgroup").write_text("{}\n", encoding="utf-8")
                    status = "materialized_directory_from_filesystem"
                    artifact_kind = "directory"
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text("data", encoding="utf-8")
                    status = "materialized_from_filesystem"
                    artifact_kind = "file"
                hydrated[key] = SimpleNamespace(
                    path=destination,
                    status=status,
                    artifact_kind=artifact_kind,
                    resolvable=True,
                    artifact=artifact,
                )
            return _Hydration(run_id, hydrated)

    tracker = _Tracker()
    restored = hydrate_pinned_closure(
        tracker=tracker,
        source_root=tmp_path / "archive",
        members=members,
    )

    assert set(restored) == {"beam-events", "asim-zarr"}
    assert [run_id for run_id, _kwargs in tracker.hydrate_calls] == [
        "beam-run-1",
        "asim-run-1",
    ]
    assert all(
        kwargs["preserve_existing"] is False
        and kwargs["on_missing"] == "warn"
        and kwargs["db_fallback"] == "never"
        for _run_id, kwargs in tracker.hydrate_calls
    )
    assert events_destination.is_file()
    assert zarr_destination.is_dir()


@pytest.mark.parametrize(
    "members",
    [
        (
            PinnedClosureMember(
                "duplicate",
                "events",
                "beam-run",
                "events",
                "h1",
                "file",
                "parquet",
                Path("/tmp/a"),
                True,
            ),
            PinnedClosureMember(
                "duplicate",
                "skims",
                "beam-run",
                "skims",
                "h2",
                "file",
                "omx",
                Path("/tmp/b"),
                True,
            ),
        ),
        (
            PinnedClosureMember(
                "root",
                "zarr_skims",
                "asim-run",
                "zarr_skims",
                "h1",
                "directory",
                "zarr",
                Path("/tmp/skims.zarr"),
                True,
            ),
            PinnedClosureMember(
                "nested",
                "events",
                "beam-run",
                "events",
                "h2",
                "file",
                "parquet",
                Path("/tmp/skims.zarr/events"),
                True,
            ),
        ),
        (
            PinnedClosureMember(
                "events",
                "events",
                "beam-run",
                "events",
                "h1",
                "file",
                "parquet",
                Path("/tmp/collision"),
                True,
            ),
            PinnedClosureMember(
                "skims",
                "skims",
                "beam-run",
                "skims",
                "h2",
                "file",
                "omx",
                Path("/tmp/collision"),
                True,
            ),
        ),
    ],
)
def test_pinned_closure_rejects_collisions_before_hydration(members):
    tracker = SimpleNamespace(
        get_run=lambda _run_id: pytest.fail("must preflight before tracker access"),
        get_run_outputs=lambda _run_id: pytest.fail(
            "must preflight before tracker access"
        ),
        hydrate_run_outputs_to_destinations=lambda *_args, **_kwargs: pytest.fail(
            "must not hydrate"
        ),
    )

    with pytest.raises(RuntimeError, match="closure"):
        hydrate_pinned_closure(
            tracker=tracker,
            source_root=Path("/tmp/archive"),
            members=members,
        )

    with pytest.raises(RuntimeError, match="closure"):
        verify_archive_visible_pinned_closure_bytes(
            tracker=tracker,
            archive_run_dir=Path("/tmp/archive"),
            members=members,
        )


def test_pinned_closure_rejects_legacy_zarr_before_hydration(tmp_path):
    member = PinnedClosureMember(
        member_id="asim-zarr",
        role="zarr_skims",
        producer_run_id="asim-run",
        output_key="zarr_skims",
        artifact_identity="zarr-hash",
        artifact_kind="directory",
        driver="zarr",
        destination=tmp_path / "skims.zarr",
        required=True,
    )
    artifact = SimpleNamespace(
        key="zarr_skims",
        hash="zarr-hash",
        driver="zarr",
        meta={},
    )
    tracker = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(id=run_id, status="completed"),
        get_run_outputs=lambda _run_id: {"zarr_skims": artifact},
        hydrate_run_outputs_to_destinations=lambda *_args, **_kwargs: pytest.fail(
            "legacy Zarr must fail before hydration"
        ),
    )

    with pytest.raises(RuntimeError, match="artifact kind"):
        hydrate_pinned_closure(
            tracker=tracker,
            source_root=tmp_path / "archive",
            members=(member,),
        )


@pytest.mark.parametrize(
    ("actual_identity", "actual_driver", "match"),
    [
        ("different-hash", "parquet", "identity mismatch"),
        ("events-hash", "csv", "driver mismatch"),
    ],
)
def test_pinned_closure_rejects_artifact_identity_or_driver_mismatch(
    tmp_path, actual_identity, actual_driver, match
):
    member = PinnedClosureMember(
        "events",
        "events",
        "beam-run",
        "events",
        "events-hash",
        "file",
        "parquet",
        tmp_path / "events.parquet",
        True,
    )
    artifact = _closure_artifact(
        "events",
        actual_identity,
        driver=actual_driver,
        artifact_kind="file",
    )
    tracker = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(id=run_id, status="completed"),
        get_run_outputs=lambda _run_id: {"events": artifact},
        hydrate_run_outputs_to_destinations=lambda *_args, **_kwargs: pytest.fail(
            "descriptor mismatch must fail before hydration"
        ),
    )

    with pytest.raises(RuntimeError, match=match):
        hydrate_pinned_closure(
            tracker=tracker,
            source_root=tmp_path / "archive",
            members=(member,),
        )


def test_pinned_closure_rejects_zarr_descriptor_that_claims_file(tmp_path):
    member = PinnedClosureMember(
        "zarr",
        "zarr_skims",
        "asim-run",
        "zarr_skims",
        "zarr-hash",
        "file",
        "zarr",
        tmp_path / "skims.zarr",
        True,
    )
    artifact = _closure_artifact(
        "zarr_skims",
        "zarr-hash",
        driver="zarr",
        artifact_kind="file",
    )
    tracker = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(id=run_id, status="completed"),
        get_run_outputs=lambda _run_id: {"zarr_skims": artifact},
        hydrate_run_outputs_to_destinations=lambda *_args, **_kwargs: pytest.fail(
            "legacy Zarr must fail before hydration"
        ),
    )

    with pytest.raises(RuntimeError, match="manifest-backed Zarr"):
        hydrate_pinned_closure(
            tracker=tracker,
            source_root=tmp_path / "archive",
            members=(member,),
        )


def test_pinned_closure_cleans_first_group_when_later_group_fails(tmp_path):
    first_destination = tmp_path / "events.parquet"
    second_destination = tmp_path / "skims.zarr"
    members = (
        PinnedClosureMember(
            "events",
            "events",
            "beam-run",
            "events",
            "events-hash",
            "file",
            "parquet",
            first_destination,
            True,
        ),
        PinnedClosureMember(
            "zarr",
            "zarr_skims",
            "asim-run",
            "zarr_skims",
            "zarr-hash",
            "directory",
            "zarr",
            second_destination,
            True,
        ),
    )
    artifacts = {
        "beam-run": {
            "events": _closure_artifact(
                "events", "events-hash", driver="parquet", artifact_kind="file"
            )
        },
        "asim-run": {
            "zarr_skims": _closure_artifact(
                "zarr_skims", "zarr-hash", driver="zarr", artifact_kind="directory"
            )
        },
    }

    class _Tracker:
        def get_run(self, run_id):
            return SimpleNamespace(id=run_id, status="completed")

        def get_run_outputs(self, run_id):
            return artifacts[run_id]

        def hydrate_run_outputs_to_destinations(self, run_id, **kwargs):
            destination = next(iter(kwargs["destinations_by_key"].values()))
            artifact = next(iter(artifacts[run_id].values()))
            if run_id == "beam-run":
                destination.write_text("events", encoding="utf-8")
                return _Hydration(
                    run_id,
                    {
                        "events": SimpleNamespace(
                            path=destination,
                            status="materialized_from_filesystem",
                            artifact_kind="file",
                            resolvable=True,
                            artifact=artifact,
                        )
                    },
                )
            return _Hydration(run_id, {})

    with pytest.raises(RuntimeError, match="zarr_skims"):
        hydrate_pinned_closure(
            tracker=_Tracker(),
            source_root=tmp_path / "archive",
            members=members,
        )

    assert not first_destination.exists()
    assert not second_destination.exists()


def test_committed_beam_run_requires_completed_direct_run_and_selected_links(tmp_path):
    run = SimpleNamespace(status="completed", year=2019, iteration=0)
    tracker = SimpleNamespace(
        get_run=lambda run_id: run if run_id == "beam-run-1" else None,
        get_run_outputs=lambda _run_id: {"linkstats": object()},
    )
    checkpoint = publish_beam_run_checkpoint(
        archive_run_dir=tmp_path,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        snapshot_ref=".consist/restart/checkpoints/pinned/tracker.duckdb",
        skim_variant="full",
        output_requests=(
            HistoricalOutputRequest(
                "linkstats", tmp_path / "beam" / "0.linkstats", True
            ),
        ),
    )

    assert (
        assert_committed_beam_run(
            tracker=tracker,
            checkpoint=checkpoint,
            output_requests=(
                HistoricalOutputRequest(
                    "linkstats", tmp_path / "beam" / "0.linkstats", True
                ),
            ),
        )
        is run
    )


def test_snapshot_publication_validates_the_pinned_snapshot_not_a_matching_query(
    tmp_path,
):
    run = SimpleNamespace(status="completed", year=2019, iteration=0)
    calls = []

    class _LiveTracker:
        def snapshot_db(self, destination, *, checkpoint):
            calls.append((Path(destination), checkpoint))
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            Path(destination).write_text("snapshot", encoding="utf-8")

    snapshot_tracker = SimpleNamespace(
        get_run=lambda run_id: run if run_id == "beam-run-1" else None,
        get_run_outputs=lambda _run_id: {"linkstats": object()},
    )
    checkpoint = snapshot_and_publish_beam_run_checkpoint(
        tracker=_LiveTracker(),
        open_snapshot=lambda path: snapshot_tracker,
        archive_run_dir=tmp_path,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        skim_variant="full",
        output_requests=(
            HistoricalOutputRequest(
                "linkstats", tmp_path / "beam" / "0.linkstats", True
            ),
        ),
    )

    assert calls and calls[0][1] is True
    assert (tmp_path / checkpoint.snapshot_ref).is_file()


def test_failed_snapshot_does_not_publish_a_beam_checkpoint(tmp_path):
    class _Tracker:
        def snapshot_db(self, *_args, **_kwargs):
            raise RuntimeError("snapshot failure")

    with pytest.raises(RuntimeError, match="snapshot failure"):
        snapshot_and_publish_beam_run_checkpoint(
            tracker=_Tracker(),
            open_snapshot=lambda _path: pytest.fail("must not open snapshot"),
            archive_run_dir=tmp_path,
            producer_run_id="beam-run-1",
            scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
            skim_variant="full",
            output_requests=(
                HistoricalOutputRequest(
                    "linkstats", tmp_path / "beam" / "0.linkstats", True
                ),
            ),
        )

    assert read_beam_run_checkpoint(tmp_path) is None


def test_checkpoint_archive_verification_accepts_immutable_zarr_directory(tmp_path):
    key = "raw_od_skims_zarr_2018_0"

    class _Tracker:
        def hydrate_run_outputs_to_destinations(self, run_id, **kwargs):
            destination = kwargs["destinations_by_key"][key]
            destination.mkdir(parents=True)
            (destination / ".zgroup").write_text("{}\n", encoding="utf-8")
            return _Hydration(
                run_id,
                {
                    key: SimpleNamespace(
                        path=destination,
                        status="materialized_directory_from_filesystem",
                        artifact_kind="directory",
                        resolvable=True,
                        artifact=SimpleNamespace(driver="zarr"),
                    )
                },
            )

    verify_archive_visible_recovery_bytes(
        tracker=_Tracker(),
        archive_run_dir=tmp_path,
        producer_run_id="beam-run-1",
        output_requests=(HistoricalOutputRequest(key, Path("/unused"), True),),
    )


def test_checkpoint_archive_verification_rejects_generic_directory(tmp_path):
    key = "unexpected_directory"

    class _Tracker:
        def hydrate_run_outputs_to_destinations(self, run_id, **kwargs):
            destination = kwargs["destinations_by_key"][key]
            destination.mkdir(parents=True)
            return _Hydration(
                run_id,
                {
                    key: SimpleNamespace(
                        path=destination,
                        status="materialized_directory_from_filesystem",
                        artifact_kind="directory",
                        resolvable=True,
                        artifact=SimpleNamespace(driver="parquet"),
                    )
                },
            )

    with pytest.raises(RuntimeError, match=key):
        verify_archive_visible_recovery_bytes(
            tracker=_Tracker(),
            archive_run_dir=tmp_path,
            producer_run_id="beam-run-1",
            output_requests=(HistoricalOutputRequest(key, Path("/unused"), True),),
        )


def test_real_consist_snapshot_reopens_pinned_completed_beam_run(tmp_path):
    import consist

    archive_run_dir = tmp_path / "archive"
    db_path = archive_run_dir / ".consist" / "provenance.duckdb"
    output_path = archive_run_dir / "beam" / "0.linkstats.csv.gz"
    db_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"linkstats")
    tracker = consist.Tracker(run_dir=archive_run_dir, db_path=db_path)
    with tracker.start_run("beam-run-1", "beam_run", year=2019, iteration=0):
        tracker.log_output(output_path, key="linkstats")

    def _open_snapshot(snapshot_path: Path):
        return consist.Tracker(
            run_dir=archive_run_dir,
            db_path=snapshot_path,
            allow_external_paths=True,
            access_mode="read_only",
        )

    checkpoint = snapshot_and_publish_beam_run_checkpoint(
        tracker=tracker,
        open_snapshot=_open_snapshot,
        archive_run_dir=archive_run_dir,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        skim_variant="disabled",
        output_requests=(
            HistoricalOutputRequest(
                "linkstats", tmp_path / "workspace" / "0.linkstats.csv.gz", True
            ),
        ),
    )

    snapshot = _open_snapshot(archive_run_dir / checkpoint.snapshot_ref)
    assert snapshot.get_run("beam-run-1").status == "completed"
    assert set(snapshot.get_run_outputs("beam-run-1")) == {"linkstats"}


def test_pinned_closure_snapshot_rejects_prior_scope_beam_output(tmp_path):
    artifact = SimpleNamespace(hash="events-hash", driver="parquet", meta={})
    tracker = SimpleNamespace(
        get_run=lambda _run_id: SimpleNamespace(
            status="completed", year=2020, iteration=0
        ),
        get_run_outputs=lambda _run_id: {"events_parquet_2021_0": artifact},
    )
    member = PinnedClosureMember(
        member_id="beam-events",
        role="events_parquet_2021_0",
        producer_run_id="beam-run-previous-year",
        output_key="events_parquet_2021_0",
        artifact_identity="events-hash",
        artifact_kind="file",
        driver="parquet",
        destination=tmp_path / "events.parquet",
        required=True,
    )

    with pytest.raises(RuntimeError, match="scope does not match"):
        validate_pinned_closure_snapshot(
            tracker=tracker,
            members=(member,),
            scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        )


def test_real_consist_snapshot_hydrates_multi_producer_successor_closure(tmp_path):
    import consist

    archive_run_dir = tmp_path / "archive"
    db_path = archive_run_dir / ".consist" / "provenance.duckdb"
    events_source = archive_run_dir / "beam" / "source.events.parquet"
    zarr_source = archive_run_dir / "activitysim" / "source.skims.zarr"
    db_path.parent.mkdir(parents=True)
    events_source.parent.mkdir(parents=True)
    events_source.write_bytes(b"events")
    zarr_source.mkdir(parents=True)
    (zarr_source / ".zgroup").write_text("{}\n", encoding="utf-8")

    tracker = consist.Tracker(run_dir=archive_run_dir, db_path=db_path)
    with tracker.start_run("beam-run-1", "beam_run", year=2021, iteration=0):
        tracker.log_output(events_source, key="events_parquet_2021_0")
    with tracker.start_run(
        "activitysim-run-1", "activitysim_run", year=2020, iteration=0
    ):
        tracker.log_output(
            zarr_source,
            key="zarr_skims",
            artifact_kind="directory",
            driver="zarr",
        )

    beam_artifact = tracker.get_run_outputs("beam-run-1")["events_parquet_2021_0"]
    zarr_artifact = tracker.get_run_outputs("activitysim-run-1")["zarr_skims"]
    closure = (
        PinnedClosureMember(
            "beam-events",
            "events_parquet_2021_0",
            "beam-run-1",
            "events_parquet_2021_0",
            str(beam_artifact.hash),
            "file",
            beam_artifact.driver,
            tmp_path / "workspace" / "beam" / "0.events.parquet",
            True,
        ),
        PinnedClosureMember(
            "activitysim-zarr",
            "zarr_skims",
            "activitysim-run-1",
            "zarr_skims",
            str(zarr_artifact.hash),
            "directory",
            zarr_artifact.driver,
            tmp_path / "workspace" / "activitysim" / "cache" / "skims.zarr",
            True,
        ),
    )

    def _open_snapshot(snapshot_path: Path):
        return consist.Tracker(
            run_dir=archive_run_dir,
            db_path=snapshot_path,
            allow_external_paths=True,
            access_mode="read_only",
        )

    checkpoint = snapshot_and_publish_beam_run_checkpoint(
        tracker=tracker,
        open_snapshot=_open_snapshot,
        archive_run_dir=archive_run_dir,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        skim_variant="disabled",
        output_requests=(),
        closure_members=closure,
    )
    restored = hydrate_pinned_closure(
        tracker=_open_snapshot(archive_run_dir / checkpoint.snapshot_ref),
        source_root=archive_run_dir,
        members=checkpoint.closure_members,
    )

    assert restored["beam-events"].path == closure[0].destination
    assert restored["activitysim-zarr"].path == closure[1].destination
    assert closure[0].destination.read_bytes() == b"events"
    assert (closure[1].destination / ".zgroup").is_file()
