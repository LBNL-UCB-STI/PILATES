import types

import duckdb

from pilates.utils import consist_runtime as cr


class _FakeNoopTracker:
    pass


class _FakeTracker:
    pass


def test_create_tracker_returns_none_when_consist_falls_back_to_noop(monkeypatch):
    fake_consist = types.SimpleNamespace(
        Tracker=lambda **kwargs: _FakeTracker(),
        create_tracker=lambda *, enabled, tracker_factory: _FakeNoopTracker(),
    )
    monkeypatch.setattr(cr, "consist", fake_consist)

    try:
        tracker = cr.create_tracker(run_dir="/tmp/consist-runs")
        assert tracker is None
    finally:
        cr.set_enabled(None)


def test_create_tracker_returns_tracker_when_consist_succeeds(monkeypatch):
    fake_consist = types.SimpleNamespace(
        Tracker=lambda **kwargs: _FakeTracker(),
        create_tracker=lambda *, enabled, tracker_factory: tracker_factory(),
    )
    monkeypatch.setattr(cr, "consist", fake_consist)

    try:
        tracker = cr.create_tracker(run_dir="/tmp/consist-runs")
        assert isinstance(tracker, _FakeTracker)
    finally:
        cr.set_enabled(None)


def test_create_tracker_retries_after_schema_compatibility_repair(monkeypatch):
    calls = {"count": 0}

    def _create_tracker(*, enabled, tracker_factory):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeNoopTracker()
        return _FakeTracker()

    fake_consist = types.SimpleNamespace(
        Tracker=lambda **kwargs: _FakeTracker(),
        create_tracker=_create_tracker,
    )
    monkeypatch.setattr(cr, "consist", fake_consist)
    monkeypatch.setattr(
        cr,
        "_repair_tracker_db_schema_compatibility",
        lambda tracker_kwargs: True,
    )

    try:
        tracker = cr.create_tracker(run_dir="/tmp/consist-runs", db_path="/tmp/x.duckdb")
        assert isinstance(tracker, _FakeTracker)
        assert calls["count"] == 2
    finally:
        cr.set_enabled(None)


def test_repair_tracker_db_schema_adds_missing_container_uri(tmp_path):
    db_path = tmp_path / "compat.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE artifact (id VARCHAR, uri VARCHAR)")
        conn.execute("INSERT INTO artifact (id, uri) VALUES ('a1', 'workspace://x')")
    finally:
        conn.close()

    migrated = cr._repair_tracker_db_schema_compatibility({"db_path": str(db_path)})
    assert migrated is True

    conn = duckdb.connect(str(db_path))
    try:
        cols = conn.execute("PRAGMA table_info('artifact')").fetchall()
        names = {row[1] for row in cols}
        assert "container_uri" in names
        rows = conn.execute(
            "SELECT uri, container_uri FROM artifact WHERE id = 'a1'"
        ).fetchall()
        assert rows == [("workspace://x", "workspace://x")]
    finally:
        conn.close()
