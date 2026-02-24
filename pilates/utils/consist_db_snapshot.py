"""
Helpers for run-local Consist DB restore and checkpoint snapshots.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


def is_local_consist_db_enabled(settings: Any) -> bool:
    """Return ``True`` when Consist DB should be maintained on node-local storage."""
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "consist_db_local_run", True))


def resolve_consist_db_paths(
    *,
    settings: Any,
    local_run_dir: str,
    archive_run_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve local/archive Consist DB paths for the current run.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        ``(local_db_path, archive_db_path)`` when database tracking is enabled, else
        ``(None, None)``.
    """
    db_cfg = getattr(getattr(settings, "shared", None), "database", None)
    if not db_cfg or not getattr(db_cfg, "enabled", False):
        return None, None

    configured_path = getattr(db_cfg, "path", None)
    if not is_local_consist_db_enabled(settings):
        if configured_path:
            resolved = os.path.realpath(os.path.expandvars(str(configured_path)))
            return resolved, resolved
        return None, None

    filename = getattr(getattr(settings, "run", None), "consist_db_filename", None)
    if not filename:
        if configured_path:
            filename = os.path.basename(str(configured_path))
        else:
            filename = "provenance.duckdb"
    filename = os.path.basename(str(filename)) or "provenance.duckdb"
    local_db_path = os.path.join(local_run_dir, ".consist", filename)
    archive_db_path = os.path.join(archive_run_dir, ".consist", filename)
    return local_db_path, archive_db_path


def mirror_consist_db_to_archive(
    local_db_path: Optional[str],
    archive_db_path: Optional[str],
) -> None:
    """
    Best-effort mirror of DB (and WAL sidecar when present) from local to archive path.

    This is used as a fallback when snapshot publication fails during shutdown.
    """
    if not local_db_path or not archive_db_path:
        return

    local_real = os.path.realpath(local_db_path)
    archive_real = os.path.realpath(archive_db_path)
    if local_real == archive_real:
        return

    if not os.path.exists(local_real):
        logger.warning(
            "Skipping Consist DB mirror; source DB does not exist: %s",
            local_real,
        )
        return

    try:
        os.makedirs(os.path.dirname(archive_real), exist_ok=True)
        shutil.copy2(local_real, archive_real)
        logger.info("Mirrored Consist DB: %s -> %s", local_real, archive_real)
    except OSError as exc:
        logger.warning(
            "Failed to mirror Consist DB from %s to %s: %s",
            local_real,
            archive_real,
            exc,
        )
        return

    local_wal = f"{local_real}.wal"
    if os.path.exists(local_wal):
        archive_wal = f"{archive_real}.wal"
        try:
            shutil.copy2(local_wal, archive_wal)
            logger.info("Mirrored Consist DB WAL: %s -> %s", local_wal, archive_wal)
        except OSError as exc:
            logger.warning(
                "Failed to mirror Consist DB WAL from %s to %s: %s",
                local_wal,
                archive_wal,
                exc,
            )


def _is_db_snapshot_enabled(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "consist_db_snapshot_enabled", True))


def _db_snapshot_interval_seconds(settings: Any) -> int:
    run_cfg = getattr(settings, "run", None)
    value = getattr(run_cfg, "consist_db_snapshot_interval_seconds", 600)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 600


def _db_snapshot_on_outer_iteration(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "consist_db_snapshot_on_outer_iteration", True))


def _db_snapshot_keep_last(settings: Any) -> int:
    run_cfg = getattr(settings, "run", None)
    value = getattr(run_cfg, "consist_db_snapshot_keep_last", 3)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 3


def _db_restore_on_start_enabled(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "consist_db_restore_on_start", True))


def _db_restore_on_start_strict(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "consist_db_restore_strict", False))


def snapshot_meta_filename(db_filename: str) -> str:
    """Return paired metadata sidecar name for a snapshot DB basename."""
    return f"{Path(db_filename).stem}.snapshot_meta.json"


def _snapshot_root_dir(archive_run_dir: str) -> Path:
    return Path(archive_run_dir) / ".consist" / "snapshots"


def snapshot_latest_dir(archive_run_dir: str) -> Path:
    """Return ``.../.consist/snapshots/latest`` for a run archive directory."""
    return _snapshot_root_dir(archive_run_dir) / "latest"


def snapshot_history_dir(archive_run_dir: str) -> Path:
    """Return ``.../.consist/snapshots/history`` for a run archive directory."""
    return _snapshot_root_dir(archive_run_dir) / "history"


def restore_local_consist_db_from_snapshot(
    *,
    settings: Any,
    local_db_path: Optional[str],
    archive_run_dir: str,
) -> bool:
    """
    Restore a local run DB from the latest archived snapshot when local DB is missing.
    """
    if (
        not local_db_path
        or not _db_restore_on_start_enabled(settings)
        or not is_local_consist_db_enabled(settings)
    ):
        return False

    local_db = Path(local_db_path)
    if local_db.exists():
        return False

    latest_dir = snapshot_latest_dir(archive_run_dir)
    source_db = latest_dir / local_db.name
    if not source_db.exists():
        return False

    meta_filename = snapshot_meta_filename(local_db.name)
    source_wal = latest_dir / f"{local_db.name}.wal"
    source_meta = latest_dir / meta_filename
    local_meta = local_db.parent / meta_filename
    local_wal = local_db.parent / f"{local_db.name}.wal"
    local_db.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(source_db, local_db)
        if source_wal.exists():
            shutil.copy2(source_wal, local_wal)
        if source_meta.exists():
            shutil.copy2(source_meta, local_meta)

        snapshot_ts = None
        if source_meta.exists():
            try:
                metadata = json.loads(source_meta.read_text(encoding="utf-8"))
                snapshot_ts = metadata.get("snapshot_ts_utc")
            except Exception:
                snapshot_ts = None
        logger.info(
            "Restored local Consist DB from snapshot %s -> %s (snapshot_ts_utc=%s)",
            source_db,
            local_db,
            snapshot_ts or "unknown",
        )
        return True
    except Exception as exc:
        msg = (
            "Failed restoring local Consist DB from archived snapshot "
            f"{source_db} -> {local_db}: {exc}"
        )
        if _db_restore_on_start_strict(settings):
            raise RuntimeError(msg) from exc
        logger.warning(msg)
        return False


def _sanitize_snapshot_reason(reason: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(reason or "snapshot"))
    normalized = normalized.strip("._-")
    return normalized[:80] or "snapshot"


def _snapshot_tracker_db(
    *,
    tracker: Any,
    local_db_path: Optional[str],
    dest_db_path: Path,
    checkpoint: bool,
    metadata: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    Snapshot via ``tracker.snapshot_db`` when available, else file-copy fallback.
    """
    snapshot_fn = getattr(tracker, "snapshot_db", None)
    if callable(snapshot_fn):
        try:
            snapshot_fn(str(dest_db_path), checkpoint=checkpoint)
        except TypeError:
            snapshot_fn(str(dest_db_path))
        return dest_db_path.exists()

    logger.warning(
        "Tracker.snapshot_db unavailable; falling back to file copy snapshot for %s",
        dest_db_path,
    )
    return _fallback_copy_db_snapshot(
        local_db_path=local_db_path,
        dest_db_path=dest_db_path,
        checkpoint=checkpoint,
        metadata=metadata,
    )


def _fallback_copy_db_snapshot(
    *,
    local_db_path: Optional[str],
    dest_db_path: Path,
    checkpoint: bool,
    metadata: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    Fallback snapshot path for older Consist versions without ``tracker.snapshot_db``.
    """
    if not local_db_path:
        return False
    source_db = Path(local_db_path)
    if not source_db.exists():
        logger.warning("Cannot fallback-snapshot missing local DB: %s", source_db)
        return False

    dest_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db, dest_db_path)

    source_wal = source_db.with_suffix(f"{source_db.suffix}.wal")
    dest_wal = dest_db_path.with_suffix(f"{dest_db_path.suffix}.wal")
    if not checkpoint and source_wal.exists():
        shutil.copy2(source_wal, dest_wal)

    meta_payload: Dict[str, Any] = dict(metadata or {})
    meta_payload.setdefault("source_db_path", str(source_db))
    meta_payload.setdefault(
        "snapshot_ts_utc",
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    (dest_db_path.parent / snapshot_meta_filename(dest_db_path.name)).write_text(
        json.dumps(meta_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return True


def _publish_latest_snapshot(
    *,
    source_snapshot_dir: Path,
    latest_dir: Path,
    db_filename: str,
) -> None:
    latest_dir.mkdir(parents=True, exist_ok=True)

    db_name = db_filename
    wal_name = f"{db_filename}.wal"
    copy_candidates = [db_name, snapshot_meta_filename(db_name)]
    if (source_snapshot_dir / wal_name).exists():
        copy_candidates.append(wal_name)

    for name in copy_candidates:
        source = source_snapshot_dir / name
        if not source.exists():
            continue
        destination = latest_dir / name
        tmp = latest_dir / f".{name}.tmp-{os.getpid()}-{time.time_ns()}"
        shutil.copy2(source, tmp)
        os.replace(tmp, destination)

    stale_wal = latest_dir / wal_name
    if wal_name not in copy_candidates and stale_wal.exists():
        stale_wal.unlink(missing_ok=True)


class ConsistDbSnapshotManager:
    """
    Coordinate startup-safe snapshot operations at explicit orchestration safe points.

    The manager does not run background snapshot threads; callers trigger snapshots
    from known workflow boundaries (interval checks, outer iteration boundaries,
    and finalize).
    """

    def __init__(
        self,
        *,
        settings: Any,
        tracker: Any,
        local_db_path: Optional[str],
        archive_run_dir: str,
    ) -> None:
        self.settings = settings
        self.tracker = tracker
        self.local_db_path = local_db_path
        self.snapshot_enabled = (
            bool(local_db_path)
            and bool(tracker)
            and _is_db_snapshot_enabled(settings)
            and is_local_consist_db_enabled(settings)
        )
        self.interval_seconds = _db_snapshot_interval_seconds(settings)
        self.snapshot_on_outer_iteration = _db_snapshot_on_outer_iteration(settings)
        self.keep_last = _db_snapshot_keep_last(settings)
        self._last_snapshot_monotonic: Optional[float] = None

        self.latest_dir = snapshot_latest_dir(archive_run_dir)
        self.history_dir = snapshot_history_dir(archive_run_dir)
        if self.snapshot_enabled:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            self.latest_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_filename(self) -> str:
        """Snapshot DB basename derived from the active local DB path."""
        if self.local_db_path:
            return Path(self.local_db_path).name
        return "provenance.duckdb"

    def maybe_snapshot_interval(self, *, reason: str) -> bool:
        """Snapshot when interval conditions are satisfied for the given safe point."""
        if not self.snapshot_enabled:
            return False
        if self.interval_seconds == 0:
            return self.snapshot(reason=reason, checkpoint=True)
        now = time.monotonic()
        if self._last_snapshot_monotonic is None:
            return self.snapshot(reason=reason, checkpoint=True)
        if now - self._last_snapshot_monotonic < float(self.interval_seconds):
            return False
        return self.snapshot(reason=reason, checkpoint=True)

    def on_outer_iteration_boundary(self, *, year: int, iteration: int) -> bool:
        """Handle outer-iteration boundary trigger using configured snapshot policy."""
        reason = f"outer_iteration_y{year}_i{iteration}"
        if self.snapshot_on_outer_iteration:
            return self.snapshot(reason=reason, checkpoint=True)
        return self.maybe_snapshot_interval(reason=reason)

    def final_snapshot(self) -> bool:
        """Attempt a final forced checkpoint snapshot during shutdown."""
        return self.snapshot(reason="finalize", checkpoint=True, force=True)

    def snapshot(self, *, reason: str, checkpoint: bool, force: bool = False) -> bool:
        """Create one snapshot in history and atomically publish it to latest."""
        if not self.snapshot_enabled:
            return False
        if self.tracker is None:
            return False

        snapshot_label = _sanitize_snapshot_reason(reason)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        history_entry = self.history_dir / f"{timestamp}_{snapshot_label}"
        dest_db_path = history_entry / self.db_filename
        metadata = {
            "reason": reason,
            "snapshot_trigger": "force" if force else "safe_point",
        }

        try:
            history_entry.mkdir(parents=True, exist_ok=False)
            ok = _snapshot_tracker_db(
                tracker=self.tracker,
                local_db_path=self.local_db_path,
                dest_db_path=dest_db_path,
                checkpoint=checkpoint,
                metadata=metadata,
            )
            if not ok:
                return False
            _publish_latest_snapshot(
                source_snapshot_dir=history_entry,
                latest_dir=self.latest_dir,
                db_filename=self.db_filename,
            )
            self._apply_retention()
            self._last_snapshot_monotonic = time.monotonic()
            logger.info(
                "Consist DB snapshot created (reason=%s, latest=%s)",
                reason,
                self.latest_dir / self.db_filename,
            )
            return True
        except Exception as exc:
            logger.warning("Consist DB snapshot failed (reason=%s): %s", reason, exc)
            return False

    def _apply_retention(self) -> None:
        """Keep only the most recent ``keep_last`` history entries."""
        if self.keep_last < 1:
            return
        entries = sorted(
            (path for path in self.history_dir.iterdir() if path.is_dir()),
            reverse=True,
        )
        for stale in entries[self.keep_last :]:
            shutil.rmtree(stale, ignore_errors=True)
