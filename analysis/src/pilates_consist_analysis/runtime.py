from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
