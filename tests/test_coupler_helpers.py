import os
from pathlib import Path
from types import SimpleNamespace

from pilates.utils import coupler_helpers


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
    def __init__(self, *, container_uri: str, path: Path) -> None:
        self.id = "artifact-1"
        self.key = "households_asim_out"
        self.container_uri = container_uri
        self._path = path

    @property
    def path(self) -> Path:
        return self._path


def test_set_coupler_from_artifact_prefers_set_from_artifact() -> None:
    coupler = CouplerWithSetFromArtifact()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "usim_datastore_h5", "artifact", fallback="fallback"
    )
    assert coupler.calls == [("set_from_artifact", "usim_datastore_h5", "artifact")]


def test_set_coupler_from_artifact_falls_back_to_set() -> None:
    coupler = CouplerWithSetOnly()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "usim_datastore_h5", None, fallback="/tmp/path.h5"
    )
    assert coupler.calls == [("set", "usim_datastore_h5", "/tmp/path.h5")]


def test_set_coupler_from_artifact_resolves_alias_key() -> None:
    coupler = CouplerWithSetOnly()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "asim_households_in", None, fallback="/tmp/households.csv"
    )
    assert coupler.calls == [("set", "households_asim_in", "/tmp/households.csv")]


def test_set_coupler_from_artifact_publishes_namespaced_and_legacy_keys() -> None:
    coupler = CouplerWithNamespaceView()
    coupler_helpers.set_coupler_from_artifact(
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

    coupler_helpers.set_coupler_from_artifact(
        coupler,
        "usim_h5_updated",
        artifact,
        fallback="/tmp/path.h5",
    )

    assert coupler.calls == [("set_from_artifact", "usim_datastore_h5", artifact)]


def test_artifact_to_existing_path_materializes_missing_artifact_with_tracker(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("PILATES_LOCAL_RUN_DIR", raising=False)
    monkeypatch.delenv("PILATES_ARCHIVE_RUN_DIR", raising=False)
    current_local = tmp_path / "current-run"
    rel_path = "activitysim/output/year-2018-iteration-0/households.parquet"
    expected_local = current_local / rel_path
    artifact = HistoricalArtifact(
        container_uri=f"workspace://{rel_path}",
        path=tmp_path / "historical-run" / rel_path,
    )
    materialized_calls = []

    def _materialize_artifact(
        requested_artifact,
        *,
        target_root,
        source_root,
        preserve_existing,
        on_missing,
        validate_content_hash,
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
        return SimpleNamespace(path=expected_local, resolvable=True)

    monkeypatch.setattr(
        coupler_helpers.cr,
        "current_tracker",
        lambda: SimpleNamespace(materialize_artifact=_materialize_artifact),
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
