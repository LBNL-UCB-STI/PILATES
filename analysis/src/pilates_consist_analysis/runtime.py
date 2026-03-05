from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

DEFAULT_DB_CANDIDATES = (
    ".consist/snapshots/latest/provenance.duckdb",
    ".consist/provenance.duckdb",
    ".consist/snapshots/latest/consist.duckdb",
    ".consist/consist.duckdb",
)

_RUN_FIELD_ALIASES = {
    "scenario_id": (
        "scenario_id",
        "scenario",
        "scenario.id",
        "facet.scenario_id",
    ),
    "year": ("year", "simulation_year", "facet.year"),
    "iteration": (
        "iteration",
        "outer_iteration",
        "simulation_iteration",
        "facet.iteration",
    ),
    "model": ("model_name", "model", "facet.model"),
}


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


def validate_run_tagging(tracker: Any) -> list[str]:
    """Return non-fatal warning messages about run tagging/linkage quality."""

    runs = _collect_runs_for_tag_validation(tracker)
    if not runs:
        return ["No runs available for run-tag validation."]

    total = len(runs)
    warnings_out: list[str] = []

    for field in ("scenario_id", "year", "iteration", "model"):
        missing = sum(1 for run in runs if _normalize_text(_run_field(run, field)) is None)
        if missing > 0:
            warnings_out.append(
                f"run_tagging.{field}.missing={missing}/{total}"
            )

    grouped: Dict[tuple[Optional[str], Optional[int], Optional[int]], list[Any]] = {}
    for run in runs:
        scenario_id = _normalize_text(_run_field(run, "scenario_id"))
        year = _as_int(_run_field(run, "year"))
        iteration = _as_int(_run_field(run, "iteration"))
        key = (scenario_id, year, iteration)
        grouped.setdefault(key, []).append(run)

    missing_beam_parent = 0
    mismatched_beam_parent = 0
    checked_beam_runs = 0

    for group_runs in grouped.values():
        asim_runs = [
            run
            for run in group_runs
            if "activitysim" in (_normalize_text(_run_field(run, "model")) or "").lower()
        ]
        beam_runs = [
            run
            for run in group_runs
            if "beam" in (_normalize_text(_run_field(run, "model")) or "").lower()
        ]
        if not asim_runs or not beam_runs:
            continue

        asim_run_id = _normalize_text(getattr(asim_runs[0], "id", None))
        if asim_run_id is None:
            continue

        for beam_run in beam_runs:
            checked_beam_runs += 1
            parent_run_id = _normalize_text(getattr(beam_run, "parent_run_id", None))
            if parent_run_id is None:
                missing_beam_parent += 1
            elif parent_run_id != asim_run_id:
                mismatched_beam_parent += 1

    if checked_beam_runs > 0 and missing_beam_parent > 0:
        warnings_out.append(
            "run_tagging.beam_parent_missing="
            f"{missing_beam_parent}/{checked_beam_runs} (expected ActivitySim parent_run_id hints)"
        )
    if checked_beam_runs > 0 and mismatched_beam_parent > 0:
        warnings_out.append(
            "run_tagging.beam_parent_mismatch="
            f"{mismatched_beam_parent}/{checked_beam_runs} (compared to ActivitySim sibling run id)"
        )

    return warnings_out


def _collect_runs_for_tag_validation(tracker: Any) -> list[Any]:
    collected: list[Any] = []
    if hasattr(tracker, "run_set"):
        try:
            collected.extend(list(tracker.run_set(label="tag-validation", limit=200000)))
        except Exception:
            pass

    if not collected and hasattr(tracker, "queries") and hasattr(tracker.queries, "find_runs"):
        try:
            collected.extend(list(tracker.queries.find_runs(limit=200000)))
        except Exception:
            pass

    deduped: Dict[str, Any] = {}
    for run in collected:
        run_id = _normalize_text(getattr(run, "id", None))
        if run_id is None:
            continue
        deduped[run_id] = run
    return list(deduped.values())


def _metadata_sources(run: Any) -> Sequence[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    for name in ("metadata", "meta"):
        value = getattr(run, name, None)
        if isinstance(value, Mapping):
            output.append(value)
    return output


def _lookup_mapping(mapping: Mapping[str, Any], key_path: str) -> Any:
    if key_path in mapping:
        return mapping.get(key_path)
    current: Any = mapping
    for part in key_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _run_field(run: Any, field: str) -> Any:
    aliases = _RUN_FIELD_ALIASES.get(field, (field,))
    for key in aliases:
        if "." not in key and hasattr(run, key):
            value = getattr(run, key)
            if _normalize_text(value) is not None:
                return value
    for source in _metadata_sources(run):
        for key in aliases:
            value = _lookup_mapping(source, key)
            if _normalize_text(value) is not None:
                return value
    return None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
