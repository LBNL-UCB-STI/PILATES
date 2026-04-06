from __future__ import annotations

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
