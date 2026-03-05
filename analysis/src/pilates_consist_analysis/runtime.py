from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

DEFAULT_DB_CANDIDATES = (
    ".consist/snapshots/latest/provenance.duckdb",
    ".consist/provenance.duckdb",
    ".consist/snapshots/latest/consist.duckdb",
    ".consist/consist.duckdb",
)


def resolve_archive_run_dir(path: str | Path) -> Path:
    run_dir = Path(path).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Archive run directory does not exist: {run_dir}")
    return run_dir


def resolve_db_path(
    archive_run_dir: str | Path,
    *,
    db_path: Optional[str | Path] = None,
) -> Path:
    run_dir = resolve_archive_run_dir(archive_run_dir)
    if db_path is not None:
        candidate = Path(db_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Consist DB does not exist: {candidate}")
        return candidate

    for relative_path in DEFAULT_DB_CANDIDATES:
        candidate = run_dir / relative_path
        if candidate.exists():
            return candidate.resolve()

    choices = "\n  - ".join(DEFAULT_DB_CANDIDATES)
    raise FileNotFoundError(
        "Could not resolve Consist DB path under archive run directory. "
        f"Checked:\n  - {choices}"
    )


def build_mounts(
    *,
    archive_run_dir: str | Path,
    project_root: str | Path,
    output_root: Optional[str | Path] = None,
    extra_mounts: Optional[Mapping[str, str | Path]] = None,
) -> Dict[str, str]:
    mounts: Dict[str, str] = {
        "inputs": str(Path(project_root).expanduser().resolve()),
        "workspace": str(resolve_archive_run_dir(archive_run_dir)),
    }
    if output_root:
        mounts["scratch"] = str(Path(output_root).expanduser().resolve())
    for key, value in (extra_mounts or {}).items():
        mounts[str(key)] = str(Path(value).expanduser().resolve())
    return mounts


def create_analysis_tracker(
    *,
    archive_run_dir: str | Path,
    project_root: str | Path,
    db_path: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    extra_mounts: Optional[Mapping[str, str | Path]] = None,
    access_mode: str = "analysis",
    hashing_strategy: str = "fast",
) -> Any:
    resolved_archive_dir = resolve_archive_run_dir(archive_run_dir)
    resolved_db_path = resolve_db_path(resolved_archive_dir, db_path=db_path)
    mounts = build_mounts(
        archive_run_dir=resolved_archive_dir,
        project_root=project_root,
        output_root=output_root,
        extra_mounts=extra_mounts,
    )
    tracker_kwargs: Dict[str, Any] = {
        "run_dir": str(resolved_archive_dir),
        "db_path": str(resolved_db_path),
        "mounts": mounts,
        "access_mode": access_mode,
        "hashing_strategy": hashing_strategy,
    }

    try:
        from pilates.utils import consist_runtime as cr
    except ImportError:
        import consist

        return consist.Tracker(**tracker_kwargs)

    tracker = cr.create_tracker(enabled=True, **tracker_kwargs)
    if tracker is None:
        raise RuntimeError("Consist tracker creation returned None.")
    return tracker


def _result_to_dict(result: Any) -> Dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    payload: Dict[str, Any] = {}
    for key in dir(result):
        if key.startswith("_"):
            continue
        value = getattr(result, key)
        if callable(value):
            continue
        payload[key] = value
    return payload


def _database_maintenance(tracker: Any, *, archive_run_dir: str | Path) -> Any:
    if getattr(tracker, "db", None) is None:
        raise RuntimeError("Tracker has no attached DB manager.")

    try:
        from consist.core.maintenance import DatabaseMaintenance
    except Exception as exc:
        raise RuntimeError(
            "Could not import consist.core.maintenance.DatabaseMaintenance."
        ) from exc

    return DatabaseMaintenance(
        db=tracker.db,
        run_dir=resolve_archive_run_dir(archive_run_dir),
    )


def inspect_db_maintenance(
    tracker: Any,
    *,
    archive_run_dir: str | Path,
) -> Dict[str, Any]:
    maintenance = _database_maintenance(tracker, archive_run_dir=archive_run_dir)
    return _result_to_dict(maintenance.inspect())


def doctor_db_maintenance(
    tracker: Any,
    *,
    archive_run_dir: str | Path,
) -> Dict[str, Any]:
    maintenance = _database_maintenance(tracker, archive_run_dir=archive_run_dir)
    return _result_to_dict(maintenance.doctor())


def get_db_health(
    tracker: Any,
    *,
    archive_run_dir: str | Path,
) -> Dict[str, Any]:
    inspect_report = inspect_db_maintenance(tracker, archive_run_dir=archive_run_dir)
    doctor_report = doctor_db_maintenance(tracker, archive_run_dir=archive_run_dir)

    doctor_issue_counts = {
        "doctor_zombie_run_count": len(doctor_report.get("zombie_run_ids", []) or []),
        "doctor_completed_without_end_time_count": len(
            doctor_report.get("completed_without_end_time", []) or []
        ),
        "doctor_dangling_parent_run_count": len(
            doctor_report.get("dangling_parent_run_ids", []) or []
        ),
        "doctor_missing_producing_run_artifact_count": len(
            doctor_report.get("artifacts_with_missing_producing_run", []) or []
        ),
        "doctor_global_table_schema_drift_count": len(
            doctor_report.get("global_table_schema_drift", {}) or {}
        ),
    }

    inspect_issue_counts = {
        "inspect_orphaned_artifact_count": int(
            inspect_report.get("orphaned_artifact_count", 0) or 0
        ),
        "inspect_zombie_run_count": len(inspect_report.get("zombie_run_ids", []) or []),
        "inspect_json_db_parity_mismatch": 0
        if bool(inspect_report.get("json_db_parity", True))
        else 1,
    }

    healthy = (
        all(count == 0 for count in doctor_issue_counts.values())
        and inspect_issue_counts["inspect_zombie_run_count"] == 0
    )

    return {
        "healthy": healthy,
        "inspect": inspect_report,
        "doctor": doctor_report,
        "doctor_issue_counts": doctor_issue_counts,
        "inspect_issue_counts": inspect_issue_counts,
    }


def get_db_health_issues(
    health_payload: Mapping[str, Any],
    *,
    strict: bool = False,
) -> list[str]:
    issues: list[str] = []
    doctor_counts = dict(health_payload.get("doctor_issue_counts", {}) or {})
    inspect_counts = dict(health_payload.get("inspect_issue_counts", {}) or {})

    for key, count in doctor_counts.items():
        if int(count or 0) > 0:
            issues.append(f"{key}={int(count)}")
    if int(inspect_counts.get("inspect_zombie_run_count", 0) or 0) > 0:
        issues.append(
            f"inspect_zombie_run_count={int(inspect_counts['inspect_zombie_run_count'])}"
        )

    if strict:
        orphan_count = int(inspect_counts.get("inspect_orphaned_artifact_count", 0) or 0)
        if orphan_count > 0:
            issues.append(f"inspect_orphaned_artifact_count={orphan_count}")
        parity_mismatch = int(
            inspect_counts.get("inspect_json_db_parity_mismatch", 0) or 0
        )
        if parity_mismatch > 0:
            issues.append("inspect_json_db_parity_mismatch=1")

    return issues


def db_health_to_frame(health_payload: Mapping[str, Any]) -> pd.DataFrame:
    inspect_report = dict(health_payload.get("inspect", {}) or {})
    doctor_counts = dict(health_payload.get("doctor_issue_counts", {}) or {})
    inspect_counts = dict(health_payload.get("inspect_issue_counts", {}) or {})

    row: Dict[str, Any] = {
        "healthy": bool(health_payload.get("healthy", False)),
        "total_runs": inspect_report.get("total_runs"),
        "total_artifacts": inspect_report.get("total_artifacts"),
        "db_file_size_mb": inspect_report.get("db_file_size_mb"),
        "json_snapshot_count": inspect_report.get("json_snapshot_count"),
        "json_db_parity": inspect_report.get("json_db_parity"),
        "run_status_count": len(inspect_report.get("runs_by_status", {}) or {}),
        "global_table_count": len(inspect_report.get("global_table_sizes", {}) or {}),
    }
    row.update(inspect_counts)
    row.update(doctor_counts)
    return pd.DataFrame([row])


def assert_db_healthy(
    tracker: Any,
    *,
    archive_run_dir: str | Path,
    strict: bool = False,
) -> Dict[str, Any]:
    health = get_db_health(tracker, archive_run_dir=archive_run_dir)
    issues = get_db_health_issues(health, strict=strict)
    if issues:
        mode = "strict" if strict else "standard"
        raise RuntimeError(f"DB health check failed ({mode}): {', '.join(issues)}")
    return health
