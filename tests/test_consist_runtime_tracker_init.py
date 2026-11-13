import types
from types import SimpleNamespace

import pytest
import consist

from pilates.utils import consist_runtime as cr


class _FakeTracker:
    pass


def test_create_tracker_returns_tracker_when_consist_succeeds(monkeypatch):
    fake_consist = types.SimpleNamespace(
        Tracker=lambda **kwargs: _FakeTracker(),
    )
    monkeypatch.setattr(cr, "consist", fake_consist)

    tracker = cr.create_tracker(run_dir="/tmp/consist-runs")
    assert isinstance(tracker, _FakeTracker)


def test_create_tracker_uses_configured_hashing_strategy(monkeypatch):
    captured: dict = {}

    def _tracker_factory(**kwargs):
        captured.update(kwargs)
        return _FakeTracker()

    fake_consist = types.SimpleNamespace(
        Tracker=_tracker_factory,
    )
    monkeypatch.setattr(cr, "consist", fake_consist)

    settings = SimpleNamespace(run=SimpleNamespace(consist_hashing_strategy="full"))
    tracker = cr.create_tracker(settings=settings, run_dir="/tmp/consist-runs")
    assert isinstance(tracker, _FakeTracker)
    assert captured.get("hashing_strategy") == "full"


def test_ephemeral_tracker():
    tracker = consist.Tracker(db_path=None, run_dir=None)
    assert isinstance(tracker, consist.Tracker)
    assert tracker.is_ephemeral is True


def test_h5_fast_hash_override_disabled_when_tracker_hashing_is_full(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "sample.h5"
    file_path.write_text("test", encoding="utf-8")
    monkeypatch.setattr(cr, "_tracker_hashing_strategy", "full")

    assert cr._maybe_fast_hash_h5(str(file_path), {}) == {}


def test_log_input_raises_when_outside_active_run(monkeypatch):
    calls = []

    def _log_input(path, key=None, **meta):
        calls.append((path, key, dict(meta)))
        raise RuntimeError("no active run")

    monkeypatch.setattr(cr, "consist", types.SimpleNamespace(log_input=_log_input))

    with pytest.raises(RuntimeError, match="no active run"):
        cr.log_input("/tmp/in.txt", key="input_key")

    assert len(calls) == 1
    path, key, meta = calls[0]
    assert key == "input_key"
    assert meta == {}
    assert str(path).endswith("/tmp/in.txt")


def test_current_run_id_returns_none_when_consist_has_no_active_run(monkeypatch):
    fake_consist = types.SimpleNamespace(current_run=lambda: None)
    monkeypatch.setattr(cr, "consist", fake_consist)

    assert cr.current_run_id() is None


def test_current_run_id_returns_stringified_run_id(monkeypatch):
    fake_consist = types.SimpleNamespace(current_run=lambda: SimpleNamespace(id=12345))
    monkeypatch.setattr(cr, "consist", fake_consist)

    assert cr.current_run_id() == "12345"
