import os
from pathlib import Path
from types import SimpleNamespace

from consist import Artifact, MaterializedArtifact, Tracker

from pilates.utils import coupler_helpers
from pilates.workflows.coupler_namespace import set_coupler_from_artifact


def test_artifact_to_path_ignores_uri(tmp_path) -> None:
    workspace = SimpleNamespace(full_path=str(tmp_path))
    uri = "s3://bucket/data/file"
    assert coupler_helpers.artifact_to_path(uri, workspace) == uri


def test_artifact_to_path_joins_relative(tmp_path) -> None:
    workspace = SimpleNamespace(full_path=str(tmp_path))
    rel_path = os.path.join("data", "file.txt")
    expected = os.path.join(str(tmp_path), rel_path)
    assert coupler_helpers.artifact_to_path(rel_path, workspace) == expected


class CouplerWithSetFromArtifact:
    def __init__(self) -> None:
        self.calls = []

    def set_from_artifact(self, key, value) -> None:
        self.calls.append(("set_from_artifact", key, value))

    def set(self, key, value) -> None:
        self.calls.append(("set", key, value))


class CouplerWithSetOnly:
    def __init__(self) -> None:
        self.calls = []

    def set(self, key, value) -> None:
        self.calls.append(("set", key, value))


class CouplerWithNamespaceView(CouplerWithSetFromArtifact):
    class _View:
        def __init__(self, outer, namespace) -> None:
            self._outer = outer
            self._namespace = namespace

        def set(self, key, value) -> None:
            self._outer.calls.append(("view.set", f"{self._namespace}/{key}", value))

    def view(self, namespace):
        return self._View(self, namespace)


class HistoricalArtifact:
    def __init__(self, *, uri: str, path: Path) -> None:
        self.id = "artifact-1"
        self.key = "households_asim_out"
        self.uri = uri
        self.driver = "parquet"
        self.member = None
        self.meta = {}
        self.digest = None
        self.abs_path = None
        self.run_id = None
        self._path = path

    @property
    def path(self) -> Path:
        return self._path


def test_set_coupler_from_artifact_prefers_set_from_artifact() -> None:
    coupler = CouplerWithSetFromArtifact()
    set_coupler_from_artifact(
        coupler, "usim_datastore_h5", "artifact", fallback="fallback"
    )
    assert coupler.calls == [("set_from_artifact", "usim_datastore_h5", "artifact")]


def test_set_coupler_from_artifact_falls_back_to_set() -> None:
    coupler = CouplerWithSetOnly()
    set_coupler_from_artifact(
        coupler, "usim_datastore_h5", None, fallback="/tmp/path.h5"
    )
    assert coupler.calls == [("set", "usim_datastore_h5", "/tmp/path.h5")]


def test_set_coupler_from_artifact_canonicalizes_namespaced_key() -> None:
    coupler = CouplerWithSetOnly()
    set_coupler_from_artifact(
        coupler,
        "activitysim/households_asim_in",
        None,
        fallback="/tmp/households.csv",
    )
    assert coupler.calls == [("set", "households_asim_in", "/tmp/households.csv")]


def test_set_coupler_from_artifact_publishes_namespaced_and_legacy_keys() -> None:
    coupler = CouplerWithNamespaceView()
    set_coupler_from_artifact(
        coupler,
        "linkstats_warmstart",
        None,
        fallback="/tmp/linkstats.csv.gz",
    )
    assert (
        "view.set",
        "beam/linkstats_warmstart",
        "/tmp/linkstats.csv.gz",
    ) in coupler.calls
    assert (
        "set_from_artifact",
        "linkstats_warmstart",
        "/tmp/linkstats.csv.gz",
    ) in coupler.calls


def test_set_coupler_from_artifact_skips_namespaced_alias_for_artifact_values() -> None:
    coupler = CouplerWithNamespaceView()
    artifact = SimpleNamespace(id="artifact-1")

    set_coupler_from_artifact(
        coupler,
        "usim_datastore_h5",
        artifact,
        fallback="/tmp/path.h5",
    )

    assert coupler.calls == [("set_from_artifact", "usim_datastore_h5", artifact)]


def test_artifact_to_existing_path_materializes_missing_artifact_with_tracker(
    monkeypatch,
    tmp_path,
) -> None:
    current_local = tmp_path / "current-run"
    rel_path = "activitysim/output/year-2018-iteration-0/households.parquet"
    expected_local = current_local / rel_path
    artifact = HistoricalArtifact(
        uri=f"workspace://{rel_path}",
        path=tmp_path / "historical-run" / rel_path,
    )
    materialized_calls = []

    class _MaterializingTracker(Tracker):
        def materialize_artifact(
            self,
            requested_artifact,
            *,
            target_root,
            source_root=None,
            preserve_existing=True,
            on_missing="warn",
            validate_content_hash="if-present",
            **_kwargs,
        ):
            materialized_calls.append(
                {
                    "artifact": requested_artifact,
                    "target_root": Path(target_root),
                    "source_root": source_root,
                    "preserve_existing": preserve_existing,
                    "on_missing": on_missing,
                    "validate_content_hash": validate_content_hash,
                }
            )
            expected_local.parent.mkdir(parents=True)
            expected_local.write_text("households", encoding="utf-8")
            return MaterializedArtifact(
                artifact=Artifact(
                    key=artifact.key,
                    uri=artifact.uri,
                    driver=artifact.driver,
                ),
                key=artifact.key,
                path=expected_local,
                source_path=artifact.path,
                target_path=expected_local,
                status="materialized_from_historical_root",
                resolvable=True,
            )

    tracker = _MaterializingTracker(run_dir=tmp_path / "tracker")
    monkeypatch.setattr(
        coupler_helpers.cr,
        "current_tracker",
        lambda: tracker,
    )

    resolved = coupler_helpers.artifact_to_existing_path(
        artifact,
        workspace=SimpleNamespace(full_path=current_local),
    )

    assert resolved == str(expected_local)
    assert expected_local.read_text(encoding="utf-8") == "households"
    assert materialized_calls == [
        {
            "artifact": artifact,
            "target_root": current_local,
            "source_root": None,
            "preserve_existing": True,
            "on_missing": "warn",
            "validate_content_hash": "if-present",
        }
    ]


def test_artifact_to_existing_path_uses_artifact_tracker_weakref_when_current_tracker_inactive(
    tmp_path: Path, monkeypatch
) -> None:
    current_local = tmp_path / "run-current"
    current_local.mkdir()
    expected_local = current_local / "beam" / "output" / "0.events.parquet"
    missing_run = tmp_path / "missing-run"
    artifact = Artifact(
        key="events_parquet_2015_0",
        uri="workspace://beam/output/0.events.parquet",
        driver="parquet",
        run_id="beam_run_1",
        meta={},
    )

    class _MaterializingTracker(Tracker):
        def materialize_artifact(
            self,
            requested_artifact: object,
            *,
            target_root: str | Path,
            source_root: str | Path | None = None,
            preserve_existing: bool = True,
            on_missing: str = "warn",
            allow_external_paths: bool | None = None,
            validate_content_hash: str = "if-present",
            **_kwargs: object,
        ) -> MaterializedArtifact:
            del (
                requested_artifact,
                target_root,
                source_root,
                preserve_existing,
                on_missing,
                allow_external_paths,
                validate_content_hash,
            )
            expected_local.parent.mkdir(parents=True, exist_ok=True)
            expected_local.write_text("events", encoding="utf-8")
            return MaterializedArtifact(
                artifact=artifact.with_abs_path(expected_local),
                key=artifact.key,
                path=expected_local,
                source_path=missing_run / "beam" / "output" / "0.events.parquet",
                target_path=expected_local,
                status="materialized_from_recovery_root",
                resolvable=True,
            )

    tracker_stub = _MaterializingTracker(
        run_dir=missing_run,
        mounts={"workspace": str(missing_run)},
    )
    artifact = artifact.with_resolver(tracker_stub)

    monkeypatch.setattr(coupler_helpers.cr, "current_tracker", lambda: None)

    resolved = coupler_helpers.artifact_to_existing_path(
        artifact,
        workspace=SimpleNamespace(full_path=current_local),
    )

    assert resolved == str(expected_local)
    assert expected_local.read_text(encoding="utf-8") == "events"
