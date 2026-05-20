import pytest
from types import SimpleNamespace

from pilates.runtime import consist_audit
from pilates.utils import coupler_helpers


class NoopArtifact:
    def __init__(self) -> None:
        self.id = "noop"


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


def test_log_and_set_output_publishes_plain_path_without_active_run(
    monkeypatch,
) -> None:
    coupler = CouplerWithSetOnly()
    logged_meta = {}
    archived = []

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: None)
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **kwargs: logged_meta.update(kwargs) or NoopArtifact(),
    )
    monkeypatch.setattr(
        coupler_helpers,
        "_enqueue_archive_copy",
        lambda key, path: archived.append((key, path)),
    )

    coupler_helpers.log_and_set_output(
        key="usim_datastore_h5",
        path="/tmp/path.h5",
        description="test",
        coupler=coupler,
    )

    assert logged_meta["meta"]["enabled"] is False
    assert archived == [("usim_datastore_h5", "/tmp/path.h5")]
    assert coupler.calls == [("set", "usim_datastore_h5", "/tmp/path.h5")]


def test_log_and_set_output_publishes_logged_artifact_with_active_run(
    monkeypatch,
) -> None:
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


def test_artifact_to_existing_path_materializes_historical_workspace_archive(
    monkeypatch,
    tmp_path,
) -> None:
    current_local = tmp_path / "local" / "current-run"
    current_archive = tmp_path / "archive" / "current-run"
    cached_archive = tmp_path / "archive" / "cached-run"
    rel_path = "activitysim/output/year-2018-iteration-0/households.parquet"
    source = cached_archive / rel_path
    source.parent.mkdir(parents=True)
    source.write_text("households", encoding="utf-8")

    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(current_local))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(current_archive))
    monkeypatch.setattr(
        coupler_helpers.cr,
        "current_tracker",
        lambda: SimpleNamespace(
            get_run=lambda run_id: SimpleNamespace(
                id=run_id,
                parent_run_id="cached-run",
                meta={
                    "_physical_run_dir": str(tmp_path / "local" / "cached-run"),
                },
            )
        ),
    )

    artifact = SimpleNamespace(
        key="households_asim_out",
        container_uri=f"workspace://{rel_path}",
        run_id="cached-step-run",
        meta={},
    )

    resolved = coupler_helpers.artifact_to_existing_path(
        artifact,
        workspace=SimpleNamespace(full_path=current_local),
        materialize_from_archive=True,
    )

    expected_local = current_local / rel_path
    assert resolved == str(expected_local)
    assert expected_local.read_text(encoding="utf-8") == "households"


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


def test_log_and_set_input_publishes_plain_path_without_active_run(monkeypatch) -> None:
    coupler = CouplerWithSetOnly()
    logged_meta = {}

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: None)
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **kwargs: logged_meta.update(kwargs) or NoopArtifact(),
    )

    coupler_helpers.log_and_set_input(
        key="usim_datastore_h5",
        path="/tmp/path.h5",
        description="test",
        coupler=coupler,
    )

    assert logged_meta["meta"]["enabled"] is False
    assert coupler.calls == [("set", "usim_datastore_h5", "/tmp/path.h5")]


def test_log_and_set_input_publishes_logged_artifact_with_active_run(
    monkeypatch,
) -> None:
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


def test_log_output_only_handles_artifact_facet_event_type(monkeypatch) -> None:
    """BEAM split-event facets use ``event_type``; lifecycle audit records do too."""
    logged = {"called": False}
    archived = []

    monkeypatch.delenv(coupler_helpers._ARCHIVE_LOCAL_ENV, raising=False)
    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: logged.__setitem__("called", True) or NoopArtifact(),
    )
    monkeypatch.setattr(
        coupler_helpers,
        "_enqueue_archive_copy",
        lambda key, path: archived.append((key, path)),
    )

    facet_fields = coupler_helpers._artifact_lifecycle_fields_from_meta(
        {
            "facet": {
                "artifact_family": "beam_event_parquet",
                "event_type": "actstart",
            }
        }
    )

    assert facet_fields["artifact_event_type"] == "actstart"
    assert "event_type" not in facet_fields

    coupler_helpers.log_output_only(
        key="events_parquet_2019_0_type_actstart",
        path="/tmp/1.events.actstart.parquet",
        description="test",
        facet={
            "artifact_family": "beam_event_parquet",
            "event_type": "actstart",
        },
    )

    assert logged["called"] is True
    assert archived == [
        ("events_parquet_2019_0_type_actstart", "/tmp/1.events.actstart.parquet")
    ]


def test_artifact_lifecycle_fields_sanitize_reserved_facet_fields() -> None:
    facet_fields = coupler_helpers._artifact_lifecycle_fields_from_meta(
        {
            "facet": {
                "artifact_family": "beam_input_archived",
                "key": "facet-key",
                "path": "facet-path",
                "src": "facet-src",
                "dest": "facet-dest",
                "year": 2035,
                "iteration": 2,
                "direction": "facet-direction",
                "description": "facet-description",
                "artifact_id": "facet-artifact-id",
                "producing_run_id": "facet-run-id",
                "recorded_at": "facet-recorded-at",
            }
        }
    )

    assert facet_fields["artifact_family"] == "beam_input_archived"
    assert facet_fields["artifact_key"] == "facet-key"
    assert facet_fields["artifact_path"] == "facet-path"
    assert facet_fields["artifact_src"] == "facet-src"
    assert facet_fields["artifact_dest"] == "facet-dest"
    assert facet_fields["artifact_year"] == 2035
    assert facet_fields["artifact_iteration"] == 2
    assert facet_fields["artifact_direction"] == "facet-direction"
    assert facet_fields["artifact_description"] == "facet-description"
    assert facet_fields["artifact_artifact_id"] == "facet-artifact-id"
    assert facet_fields["artifact_producing_run_id"] == "facet-run-id"
    assert facet_fields["artifact_recorded_at"] == "facet-recorded-at"
    for reserved_field in (
        "key",
        "path",
        "src",
        "dest",
        "year",
        "iteration",
        "direction",
        "description",
        "artifact_id",
        "producing_run_id",
        "recorded_at",
    ):
        assert reserved_field not in facet_fields
    assert facet_fields["sanitized_lifecycle_fields"] == {
        "artifact_id": "artifact_artifact_id",
        "dest": "artifact_dest",
        "description": "artifact_description",
        "direction": "artifact_direction",
        "iteration": "artifact_iteration",
        "key": "artifact_key",
        "path": "artifact_path",
        "producing_run_id": "artifact_producing_run_id",
        "recorded_at": "artifact_recorded_at",
        "src": "artifact_src",
        "year": "artifact_year",
    }


def test_log_output_only_sanitizes_facet_lifecycle_collisions(monkeypatch) -> None:
    emitted = []

    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        coupler_helpers,
        "_log_with_optional_h5_container",
        lambda **_kwargs: SimpleNamespace(id="logged-artifact-id"),
    )
    monkeypatch.setattr(coupler_helpers, "_enqueue_archive_copy", lambda *_args: None)
    monkeypatch.setattr(
        coupler_helpers,
        "_emit_artifact_lifecycle_event",
        lambda event_type, **fields: emitted.append((event_type, fields)),
    )

    coupler_helpers.log_output_only(
        key="beam_input_2035_2_archived",
        path="/tmp/beam-inputs",
        description="logged description",
        facet={
            "artifact_family": "beam_input_archived",
            "event_type": "facet-event",
            "key": "facet-key",
            "path": "facet-path",
            "direction": "facet-direction",
            "description": "facet-description",
            "artifact_id": "facet-artifact-id",
            "producing_run_id": "facet-run-id",
            "recorded_at": "facet-recorded-at",
        },
    )

    assert len(emitted) == 1
    event_type, fields = emitted[0]
    assert event_type == "artifact_logged"
    assert fields["key"] == "beam_input_2035_2_archived"
    assert fields["path"] == "/tmp/beam-inputs"
    assert fields["direction"] == "output"
    assert fields["description"] == "logged description"
    assert fields["artifact_id"] == "logged-artifact-id"
    assert fields["artifact_event_type"] == "facet-event"
    assert fields["artifact_key"] == "facet-key"
    assert fields["artifact_path"] == "facet-path"
    assert fields["artifact_direction"] == "facet-direction"
    assert fields["artifact_description"] == "facet-description"
    assert fields["artifact_artifact_id"] == "facet-artifact-id"
    assert fields["artifact_producing_run_id"] == "facet-run-id"
    assert fields["artifact_recorded_at"] == "facet-recorded-at"
    assert fields["sanitized_lifecycle_fields"] == {
        "artifact_id": "artifact_artifact_id",
        "description": "artifact_description",
        "direction": "artifact_direction",
        "event_type": "artifact_event_type",
        "key": "artifact_key",
        "path": "artifact_path",
        "producing_run_id": "artifact_producing_run_id",
        "recorded_at": "artifact_recorded_at",
    }


def test_log_output_only_logs_and_enqueues_archive_without_active_run(
    monkeypatch,
) -> None:
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


def test_log_input_only_logs_without_active_run_and_does_not_enqueue_archive(
    monkeypatch,
) -> None:
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


def test_log_input_only_logs_with_active_run_and_does_not_enqueue_archive(
    monkeypatch,
) -> None:
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


def test_emit_artifact_lifecycle_event_requires_existing_summary(
    monkeypatch, tmp_path
) -> None:
    emitted = []
    local_root = tmp_path / "local" / "run"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setattr(
        consist_audit,
        "emit_artifact_lifecycle_audit_event",
        lambda **fields: emitted.append(fields),
    )

    coupler_helpers._emit_artifact_lifecycle_event(
        "archive_copy_checkpoint",
        require_existing_summary=True,
        key="beam_input_plans_archived",
        path=str(local_root / "beam" / "output" / "plans.csv.gz"),
    )

    assert emitted == []

    summary_path = (
        local_root / ".workflow" / "diagnostics" / "artifact_lifecycle_audit_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")

    coupler_helpers._emit_artifact_lifecycle_event(
        "archive_copy_checkpoint",
        require_existing_summary=True,
        key="beam_input_plans_archived",
        path=str(local_root / "beam" / "output" / "plans.csv.gz"),
    )

    assert emitted == [
        {
            "event_type": "archive_copy_checkpoint",
            "key": "beam_input_plans_archived",
            "path": str(local_root / "beam" / "output" / "plans.csv.gz"),
        }
    ]
