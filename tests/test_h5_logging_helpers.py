import types

from pilates.utils import consist_runtime as cr
from pilates.utils import coupler_helpers as ch


def _install_consist_stub(monkeypatch, calls):
    def _log_input(path, key=None, enabled=None, **meta):
        calls.append(("input", key, meta))
        return {"path": path, "key": key}

    def _log_output(path, key=None, enabled=None, **meta):
        calls.append(("output", key, meta))
        return {"path": path, "key": key}

    def _log_h5_container(path, key=None, direction="input", **meta):
        calls.append(("h5_container", direction, key, meta))
        return {"path": path, "key": key}

    stub = types.SimpleNamespace(
        log_input=_log_input,
        log_output=_log_output,
    )
    monkeypatch.setattr(cr, "consist", stub)
    monkeypatch.setattr(cr, "log_h5_container", _log_h5_container)


def test_log_output_only_uses_h5_container_when_flagged(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    ch.log_output_only(
        key="usim_datastore_h5",
        path="/tmp/data.h5",
        description="test",
        h5_container=True,
    )

    assert calls
    assert calls[0][0] == "h5_container"
    assert calls[0][1] == "output"


def test_log_input_only_falls_back_without_h5_flag(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    ch.log_input_only(
        key="usim_datastore_h5",
        path="/tmp/data.h5",
        description="test",
    )

    assert calls
    assert calls[0][0] == "input"
