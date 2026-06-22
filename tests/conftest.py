from __future__ import annotations

from pathlib import Path

import pytest

from pilates.utils import consist_runtime as cr


_DISABLE_CONSIST_LOGGING_BASENAMES = {
    "test_activitysim_compile_run_handshake.py",
    "test_archive_copy_workflow.py",
    "test_manifest_cache_parity.py",
    "test_workflow_contract_validation.py",
}


@pytest.fixture(autouse=True)
def _disable_consist_logging_for_isolated_step_tests(request):
    path = getattr(request.node, "path", None)
    basename = getattr(path, "name", None)
    should_disable = basename in _DISABLE_CONSIST_LOGGING_BASENAMES
    if should_disable:
        cr.set_enabled(False)
    try:
        yield
    finally:
        if should_disable:
            cr.set_enabled(None)


@pytest.fixture(autouse=True)
def _run_tests_from_repo_root(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])


@pytest.fixture
def golden_stub_env(tmp_path, monkeypatch):
    """Expose the golden stub environment without globally loading its module."""
    from tests.test_golden_stub_workflow import golden_stub_env as _golden_stub_env

    yield from _golden_stub_env.__wrapped__(tmp_path, monkeypatch)
