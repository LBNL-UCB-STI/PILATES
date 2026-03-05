from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable


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


def export_bundle(
    tracker: Any,
    *,
    archive_run_dir: str | Path,
    run_ids: Iterable[str] | str,
    out_path: str | Path,
    include_data: bool,
    include_snapshots: bool,
    include_children: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if getattr(tracker, "db", None) is None:
        raise RuntimeError("Tracker has no attached DB manager.")

    try:
        from consist.core.maintenance import DatabaseMaintenance
    except Exception as exc:
        raise RuntimeError(
            "Could not import consist.core.maintenance.DatabaseMaintenance."
        ) from exc

    maintenance = DatabaseMaintenance(
        db=tracker.db,
        run_dir=Path(archive_run_dir).expanduser().resolve(),
    )
    result = maintenance.export(
        run_ids=run_ids,
        out_path=Path(out_path).expanduser().resolve(),
        include_data=include_data,
        include_snapshots=include_snapshots,
        include_children=include_children,
        dry_run=dry_run,
    )
    return _result_to_dict(result)
