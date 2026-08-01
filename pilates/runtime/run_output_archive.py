"""Archive completed native Consist child runs into the configured recovery root."""

from __future__ import annotations

import os
from pathlib import Path

from consist import RunResult
from consist.core.tracker import Tracker

from pilates.runtime.archive_paths import archive_roots

_ARCHIVE_ENABLE_ENV = "PILATES_ENABLE_ARCHIVE_COPY"


def _archive_enabled() -> bool:
    return os.environ.get(_ARCHIVE_ENABLE_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configured_child_recovery_root(run_id: str) -> str | None:
    """Return the enabled archive location isolated to one completed child run."""

    roots = archive_roots()
    if roots is None:
        return None
    local_root, archive_root = roots
    if os.path.normcase(os.path.normpath(local_root)) == os.path.normcase(
        os.path.normpath(archive_root)
    ):
        return None
    return str(Path(archive_root) / "consist-recovery" / run_id)


def archive_completed_run(*, tracker: Tracker | None, result: RunResult) -> RunResult:
    """Archive enabled child-run outputs and return their refreshed artifacts."""

    if not _archive_enabled():
        return result
    archive_root = _configured_child_recovery_root(str(result.run.id))
    if archive_root is None:
        return result
    if tracker is None:
        raise RuntimeError("Cannot archive completed run without a Consist tracker.")
    archived = tracker.archive_run_outputs(
        str(result.run.id), archive_root, mode="copy"
    )
    return RunResult(
        run=result.run,
        outputs=dict(archived.outputs),
        cache_hit=result.cache_hit,
    )
