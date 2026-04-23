import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

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


def test_create_tracker_uses_configured_hashing_strategy(monkeypatch):
    captured: dict = {}

    def _tracker_factory(**kwargs):
        captured.update(kwargs)
        return _FakeTracker()

    fake_consist = types.SimpleNamespace(
        Tracker=_tracker_factory,
        create_tracker=lambda *, enabled, tracker_factory: tracker_factory(),
    )
    monkeypatch.setattr(cr, "consist", fake_consist)

    settings = SimpleNamespace(run=SimpleNamespace(consist_hashing_strategy="full"))
    try:
        tracker = cr.create_tracker(settings=settings, run_dir="/tmp/consist-runs")
        assert isinstance(tracker, _FakeTracker)
        assert captured.get("hashing_strategy") == "full"
    finally:
        cr.set_enabled(None)


def test_h5_fast_hash_override_disabled_when_tracker_hashing_is_full(tmp_path, monkeypatch):
    file_path = tmp_path / "sample.h5"
    file_path.write_text("test", encoding="utf-8")
    monkeypatch.setattr(cr, "_tracker_hashing_strategy", "full")

    assert cr._maybe_fast_hash_h5(str(file_path), {}) == {}


def test_scenario_disabled_delegates_to_consist_noop_context(monkeypatch):
    calls = []
    sentinel = object()

    @contextmanager
    def _scenario(name, tracker=None, *, enabled=True, **kwargs):
        calls.append((name, tracker, enabled, kwargs))
        yield sentinel

    monkeypatch.setattr(cr, "consist", types.SimpleNamespace(scenario=_scenario))

    with cr.scenario("noop-step", enabled=False, phase="test") as scenario:
        assert scenario is sentinel

    assert calls == [("noop-step", None, False, {"phase": "test"})]


def test_log_input_raises_when_enabled_outside_active_run(monkeypatch):
    calls = []

    def _log_input(path, key=None, *, enabled=True, **meta):
        calls.append((enabled, path, key, dict(meta)))
        if enabled:
            raise RuntimeError("no active run")
        return types.SimpleNamespace(path=path, key=key, meta=dict(meta))

    monkeypatch.setattr(cr, "consist", types.SimpleNamespace(log_input=_log_input))

    with pytest.raises(RuntimeError, match="no active run"):
        cr.log_input("/tmp/in.txt", key="input_key", enabled=True)

    assert len(calls) == 1
    enabled, path, key, meta = calls[0]
    assert enabled is True
    assert key == "input_key"
    assert meta == {}
    assert str(path).endswith("/tmp/in.txt")


def test_current_run_id_returns_none_when_runtime_disabled(monkeypatch):
    fake_consist = types.SimpleNamespace(current_run=lambda: SimpleNamespace(id="run-123"))
    monkeypatch.setattr(cr, "consist", fake_consist)
    cr.set_enabled(False)

    try:
        assert cr.current_run_id() is None
    finally:
        cr.set_enabled(None)


def test_current_run_id_returns_none_when_consist_has_no_active_run(monkeypatch):
    fake_consist = types.SimpleNamespace(current_run=lambda: None)
    monkeypatch.setattr(cr, "consist", fake_consist)

    try:
        assert cr.current_run_id() is None
    finally:
        cr.set_enabled(None)


def test_current_run_id_returns_stringified_run_id(monkeypatch):
    fake_consist = types.SimpleNamespace(current_run=lambda: SimpleNamespace(id=12345))
    monkeypatch.setattr(cr, "consist", fake_consist)

    try:
        assert cr.current_run_id() == "12345"
    finally:
        cr.set_enabled(None)
