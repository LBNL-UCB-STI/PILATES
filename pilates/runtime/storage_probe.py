from __future__ import annotations

import logging
import os
import shutil
import socket
from typing import Dict


logger = logging.getLogger(__name__)
STORAGE_PROBE_ENV = "PILATES_LOG_STORAGE_PROBE"


def storage_probe_enabled() -> bool:
    value = os.environ.get(STORAGE_PROBE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_mount_table() -> Dict[str, str]:
    mounts: Dict[str, str] = {}
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    mountpoint = parts[1]
                    fstype = parts[2]
                    mounts[mountpoint] = fstype
    except OSError:
        return {}
    return mounts


def _mount_for_path(path: str, mounts: Dict[str, str]) -> str:
    path = os.path.realpath(path)
    best_match = ""
    for mountpoint in mounts:
        if path == mountpoint or path.startswith(mountpoint.rstrip("/") + "/"):
            if len(mountpoint) > len(best_match):
                best_match = mountpoint
    return best_match


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}EiB"


def log_local_storage_info() -> None:
    mounts = _read_mount_table()
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_NODELIST")
    logger.info(
        "Storage probe: host=%s job_id=%s nodelist=%s",
        hostname,
        job_id or "n/a",
        node_list or "n/a",
    )

    candidates = []
    for var in ("SLURM_TMPDIR", "TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(var)
        if value:
            candidates.append(value)
    candidates += [
        "/tmp",
        "/var/tmp",
        "/dev/shm",
        "/scratch",
        "/local",
        "/local_scratch",
        "/lscratch",
        "/mnt",
    ]

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        mountpoint = _mount_for_path(path, mounts)
        fstype = mounts.get(mountpoint, "unknown")
        logger.info(
            "Storage candidate: path=%s mount=%s fstype=%s free=%s total=%s",
            os.path.realpath(path),
            mountpoint or "unknown",
            fstype,
            _format_bytes(usage.free),
            _format_bytes(usage.total),
        )


def log_local_storage_info_if_enabled() -> None:
    if storage_probe_enabled():
        log_local_storage_info()
