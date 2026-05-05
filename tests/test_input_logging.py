from typing import Dict, Optional

from pilates.utils import input_logging


def test_log_inputs_no_active_run_skips(monkeypatch) -> None:
    inputs = {"a": "/tmp/a", "b": "/tmp/b"}
    descriptions: Dict[str, Optional[str]] = {"a": "desc", "b": "desc"}

    monkeypatch.setattr(input_logging.cr, "current_run_id", lambda: None)

    def _fail_log_input(*args, **kwargs):
        raise AssertionError("log_input should not be called without active run")

    monkeypatch.setattr(input_logging.cr, "log_input", _fail_log_input)

    input_logging.log_inputs(inputs, descriptions)


def test_log_inputs_active_run_logs(monkeypatch) -> None:
    inputs = {"a": "/tmp/a", "b": "/tmp/b"}
    descriptions: Dict[str, Optional[str]] = {"a": "desc", "b": None}
    calls = []

    monkeypatch.setattr(input_logging.cr, "current_run_id", lambda: "run-1")

    def _log_input(path, key=None, description=None):
        calls.append((path, key, description))
        return None

    monkeypatch.setattr(input_logging.cr, "log_input", _log_input)

    input_logging.log_inputs(inputs, descriptions)

    assert calls == [("/tmp/a", "a", "desc")]
