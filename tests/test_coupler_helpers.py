import consist
import pytest
from types import SimpleNamespace

from pilates.utils import coupler_helpers


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
    assert ("view.set", "beam/linkstats_warmstart", "/tmp/linkstats.csv.gz") in coupler.calls
    assert ("set_from_artifact", "linkstats_warmstart", "/tmp/linkstats.csv.gz") in coupler.calls


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


def test_set_coupler_from_artifact_falls_back_from_noop_artifact() -> None:
    coupler = CouplerWithNamespaceView()
    noop = consist.NoopArtifact(
        key="usim_datastore_h5",
        path="/tmp/noop-path.h5",
        container_uri="workspace://noop-path.h5",
    )

    coupler_helpers.set_coupler_from_artifact(
        coupler,
        "usim_datastore_h5",
        noop,
        fallback="/tmp/fallback-path.h5",
    )

    assert ("view.set", "urbansim/usim_datastore_h5", "/tmp/fallback-path.h5") in coupler.calls
    assert ("set_from_artifact", "usim_datastore_h5", "/tmp/fallback-path.h5") in coupler.calls
    assert all(call[-1] is not noop for call in coupler.calls)


def test_log_and_set_output_raises_without_active_run(monkeypatch) -> None:
    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: None)
    with pytest.raises(RuntimeError):
        coupler_helpers.log_and_set_output(
            key="usim_datastore_h5",
            path="/tmp/path.h5",
            description="test",
            coupler=CouplerWithSetOnly(),
        )


def test_log_and_set_output_publishes_logged_artifact_with_active_run(monkeypatch) -> None:
    coupler = CouplerWithSetFromArtifact()
    archived = []
    artifact = object()

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: artifact,
    )
    monkeypatch.setattr(
        coupler_helpers,
        "_enqueue_archive_copy",
        lambda key, path: archived.append((key, path)),
    )

    coupler_helpers.log_and_set_output(
        key="linkstats_warmstart",
        path="/tmp/linkstats.parquet",
        description="test",
        coupler=coupler,
    )

    assert archived == [("linkstats_warmstart", "/tmp/linkstats.parquet")]
    assert ("set_from_artifact", "linkstats_warmstart", artifact) in coupler.calls


def test_log_and_set_output_reuses_matching_current_run_artifact(monkeypatch) -> None:
    coupler = CouplerWithSetFromArtifact()
    artifact = SimpleNamespace(key="usim_datastore_h5", container_uri="/tmp/path.h5")
    tracker = SimpleNamespace(current_consist=SimpleNamespace(outputs=[artifact]))

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())
    monkeypatch.setattr(coupler_helpers.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: pytest.fail("should not log duplicate output"),
    )

    coupler_helpers.log_and_set_output(
        key="usim_datastore_h5",
        path="/tmp/path.h5",
        description="test",
        coupler=coupler,
    )

    assert coupler.calls == [("set_from_artifact", "usim_datastore_h5", artifact)]


def test_log_and_set_output_publishes_h5_container_artifact_not_tuple(
    monkeypatch,
) -> None:
    coupler = CouplerWithSetFromArtifact()
    container_artifact = object()
    table_artifact = object()

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: (container_artifact, [table_artifact]),
    )

    coupler_helpers.log_and_set_output(
        key="usim_h5_updated",
        path="/tmp/path.h5",
        description="test",
        coupler=coupler,
    )

    assert coupler.calls == [
        ("set_from_artifact", "usim_datastore_h5", container_artifact)
    ]


def test_log_and_set_input_raises_without_active_run(monkeypatch) -> None:
    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: None)
    with pytest.raises(RuntimeError):
        coupler_helpers.log_and_set_input(
            key="usim_datastore_h5",
            path="/tmp/path.h5",
            description="test",
            coupler=CouplerWithSetOnly(),
        )


def test_log_and_set_input_publishes_logged_artifact_with_active_run(monkeypatch) -> None:
    coupler = CouplerWithSetFromArtifact()
    artifact = object()

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: artifact,
    )

    coupler_helpers.log_and_set_input(
        key="usim_datastore_h5",
        path="/tmp/path.h5",
        description="test",
        coupler=coupler,
    )

    assert coupler.calls == [("set_from_artifact", "usim_datastore_h5", artifact)]


def test_log_output_only_logs_and_enqueues_archive_with_active_run(monkeypatch) -> None:
    logged = {"called": False}
    archived = []

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: logged.__setitem__("called", True),
    )
    monkeypatch.setattr(
        coupler_helpers,
        "_enqueue_archive_copy",
        lambda key, path: archived.append((key, path)),
    )

    coupler_helpers.log_output_only(
        key="usim_datastore_h5",
        path="/tmp/path.h5",
        description="test",
    )

    assert logged["called"] is True
    assert archived == [("usim_datastore_h5", "/tmp/path.h5")]


def test_log_output_only_logs_and_enqueues_archive_without_active_run(monkeypatch) -> None:
    logged = {"called": False}
    archived = []

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: None)
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: logged.__setitem__("called", True),
    )
    monkeypatch.setattr(
        coupler_helpers,
        "_enqueue_archive_copy",
        lambda key, path: archived.append((key, path)),
    )

    coupler_helpers.log_output_only(
        key="usim_datastore_h5",
        path="/tmp/path.h5",
        description="test",
    )

    assert logged["called"] is True
    assert archived == [("usim_datastore_h5", "/tmp/path.h5")]


def test_log_input_only_logs_without_active_run_and_does_not_enqueue_archive(monkeypatch) -> None:
    logged = {"called": False}
    archived = []

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: None)
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: logged.__setitem__("called", True),
    )
    monkeypatch.setattr(
        coupler_helpers,
        "_enqueue_archive_copy",
        lambda key, path: archived.append((key, path)),
    )

    coupler_helpers.log_input_only(
        key="usim_datastore_h5",
        path="/tmp/path.h5",
        description="test",
    )

    assert logged["called"] is True
    assert archived == []


def test_log_input_only_logs_with_active_run_and_does_not_enqueue_archive(monkeypatch) -> None:
    logged = {"called": False}
    archived = []

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: logged.__setitem__("called", True),
    )
    monkeypatch.setattr(
        coupler_helpers,
        "_enqueue_archive_copy",
        lambda key, path: archived.append((key, path)),
    )

    coupler_helpers.log_input_only(
        key="usim_datastore_h5",
        path="/tmp/path.h5",
        description="test",
    )

    assert logged["called"] is True
    assert archived == []
