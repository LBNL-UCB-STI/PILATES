from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Optional, Sequence

from pilates.config import PilatesConfig, load_config
from pilates.runtime.failure_hints import log_consist_cli_instructions
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import RunLike
from pilates.utils.consist_db_snapshot import (
    resolve_consist_db_paths,
    snapshot_latest_dir,
)
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RootPromotionResult:
    recovery_root: str
    destination_run_dir: str
    status: str
    copy_performed: bool = False
    verified: bool = False
    artifact_metadata_updated: bool = False
    error: Optional[str] = None


@dataclass(slots=True)
class PromotionResult:
    source_run_dir: str
    dry_run: bool
    verify_only: bool
    merge_db_only: bool
    db_path: Optional[str]
    marker_path: Optional[str]
    root_run_id: Optional[str] = None
    scoped_run_ids: list[str] = field(default_factory=list)
    merge_result: Optional[dict[str, Any]] = None
    roots: list[RootPromotionResult] = field(default_factory=list)

    @property
    def succeeded_roots(self) -> list[RootPromotionResult]:
        return [root for root in self.roots if root.status == "promoted"]

    @property
    def failed_roots(self) -> list[RootPromotionResult]:
        return [root for root in self.roots if root.status == "failed"]

    @property
    def success(self) -> bool:
        if self.failed_roots:
            return False
        if self.merge_db_only:
            return bool(
                self.merge_result is not None
                and not self.merge_result.get("skipped", False)
            )
        return bool(self.roots)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _expand_roots(roots: Optional[Iterable[str | os.PathLike[str]]]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[str] = set()
    for root in roots or ():
        resolved = Path(os.path.expandvars(os.fspath(root))).expanduser().resolve()
        marker = str(resolved)
        if marker in seen:
            continue
        seen.add(marker)
        expanded.append(resolved)
    return expanded


def _source_archive_run_dir(
    settings: PilatesConfig,
    archive_run_dir: Optional[str | os.PathLike[str]] = None,
) -> Path:
    if archive_run_dir is not None:
        candidate = Path(os.path.expandvars(os.fspath(archive_run_dir))).expanduser()
        return candidate.resolve()

    run_cfg = settings.run
    archive_root = Path(run_cfg.output_directory).expanduser().resolve()
    direct = archive_root / run_cfg.output_run_name
    if direct.exists():
        return direct.resolve()

    pattern = f"pilates-run--{run_cfg.region}--{run_cfg.output_run_name}--*"
    matches = sorted(
        (path for path in archive_root.glob(pattern) if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if matches:
        if len(matches) > 1:
            logger.info(
                "Resolved latest matching archive run dir for output_run_name=%s: %s",
                run_cfg.output_run_name,
                matches[0],
            )
        return matches[0].resolve()

    raise FileNotFoundError(
        "Could not resolve archive run directory. Pass --run-dir explicitly or "
        f"ensure {direct} exists."
    )


def _recovery_roots(
    settings: PilatesConfig,
    *,
    roots: Optional[Iterable[str | os.PathLike[str]]] = None,
) -> list[Path]:
    configured = list(getattr(settings.run, "recovery_archive_roots", []) or [])
    if roots is not None:
        configured.extend(os.fspath(root) for root in roots)
    resolved = _expand_roots(configured)
    if not resolved:
        raise ValueError(
            "No recovery archive roots configured. Set run.recovery_archive_roots "
            "or pass --root."
        )
    return resolved


def _promotion_copy_ignore(
    dirpath: str, names: list[str], *, source_run_dir: Path
) -> set[str]:
    try:
        rel = Path(dirpath).relative_to(source_run_dir)
    except ValueError:
        return set()
    if rel == Path("activitysim") / "output":
        ignored: set[str] = set()
        if "final_pipeline" in names:
            ignored.add("final_pipeline")
        if "cache" in names:
            ignored.add("cache")
        return ignored
    if rel == Path("shared_cache"):
        return {"numba"} if "numba" in names else set()
    if rel == Path(".consist") / "snapshots":
        return {"history"} if "history" in names else set()
    return set()


def _copy_run_tree(source_run_dir: Path, destination_run_dir: Path) -> None:
    source_run_dir = source_run_dir.resolve()
    destination_run_dir.parent.mkdir(parents=True, exist_ok=True)
    if (source_run_dir / "activitysim" / "output" / "final_pipeline").exists():
        logger.info("Skipping ActivitySim final_pipeline tree during promotion copy")
    if (source_run_dir / "activitysim" / "output" / "cache").exists():
        logger.info("Skipping ActivitySim runtime cache tree during promotion copy")
    if (source_run_dir / "shared_cache" / "numba").exists():
        logger.info("Skipping shared numba cache during promotion copy")
    if (source_run_dir / ".consist" / "snapshots" / "history").exists():
        logger.info("Skipping Consist snapshot history tree during promotion copy")
    shutil.copytree(
        source_run_dir,
        destination_run_dir,
        dirs_exist_ok=True,
        ignore=lambda dirpath, names: _promotion_copy_ignore(
            dirpath, names, source_run_dir=source_run_dir
        ),
    )
    excluded_paths = [
        destination_run_dir / "activitysim" / "output" / "final_pipeline",
        destination_run_dir / "activitysim" / "output" / "cache",
        destination_run_dir / "shared_cache" / "numba",
        destination_run_dir / ".consist" / "snapshots" / "history",
    ]
    for excluded_path in excluded_paths:
        if excluded_path.exists():
            logger.info("Pruning excluded promotion tree: %s", excluded_path)
            shutil.rmtree(excluded_path, ignore_errors=True)


def _tree_inventory(root: Path) -> tuple[int, int, int]:
    root = root.resolve()
    file_count = 0
    dir_count = 0
    byte_count = 0
    for current_root, dirs, files in os.walk(root):
        try:
            rel = Path(current_root).relative_to(root)
        except ValueError:
            rel = None
        if rel == Path("activitysim") / "output" and "final_pipeline" in dirs:
            dirs.remove("final_pipeline")
        if rel == Path("activitysim") / "output" and "cache" in dirs:
            dirs.remove("cache")
        if rel == Path("shared_cache") and "numba" in dirs:
            dirs.remove("numba")
        if rel == Path(".consist") / "snapshots" and "history" in dirs:
            dirs.remove("history")
        dir_count += len(dirs)
        file_count += len(files)
        for name in files:
            try:
                byte_count += (Path(current_root) / name).stat().st_size
            except OSError:
                continue
    return file_count, dir_count, byte_count


def _verify_promoted_run_dir(
    *,
    source_run_dir: Path,
    destination_run_dir: Path,
    require_consist_state: bool,
) -> None:
    if not destination_run_dir.exists():
        raise FileNotFoundError(
            f"Promoted run directory does not exist: {destination_run_dir}"
        )

    source_consist_dir = source_run_dir / ".consist"
    if require_consist_state and source_consist_dir.exists():
        destination_consist_dir = destination_run_dir / ".consist"
        if not destination_consist_dir.exists():
            raise FileNotFoundError(
                f"Promoted run is missing .consist state: {destination_consist_dir}"
            )

    source_run_state = source_run_dir / "run_state.yaml"
    if source_run_state.exists():
        destination_run_state = destination_run_dir / "run_state.yaml"
        if not destination_run_state.exists():
            raise FileNotFoundError(
                f"Promoted run is missing run_state.yaml: {destination_run_state}"
            )

    source_workflow_dir = source_run_dir / ".workflow"
    if source_workflow_dir.exists():
        destination_workflow_dir = destination_run_dir / ".workflow"
        if not destination_workflow_dir.exists():
            raise FileNotFoundError(
                f"Promoted run is missing workflow metadata dir: {destination_workflow_dir}"
            )


def _archive_db_path(
    settings: PilatesConfig,
    *,
    archive_run_dir: Path,
) -> Optional[Path]:
    _local_db_path, archive_db_path = resolve_consist_db_paths(
        settings=settings,
        local_run_dir=str(archive_run_dir),
        archive_run_dir=str(archive_run_dir),
    )
    if not archive_db_path:
        return None
    resolved = Path(archive_db_path)
    if resolved.exists():
        return resolved
    latest_snapshot = snapshot_latest_dir(str(archive_run_dir)) / resolved.name
    if latest_snapshot.exists():
        return latest_snapshot
    return None


def _open_archive_tracker(
    settings: PilatesConfig,
    *,
    archive_run_dir: Path,
):
    db_path = _archive_db_path(settings, archive_run_dir=archive_run_dir)
    if db_path is None:
        return None
    tracker = cr.create_tracker(
        settings=settings,
        run_dir=str(archive_run_dir),
        db_path=str(db_path),
        allow_external_paths=True,
        mounts={
            "inputs": str(_project_root()),
            "workspace": str(archive_run_dir),
            "scratch": str(Path(settings.run.output_directory).expanduser().resolve()),
        },
        project_root=str(_project_root()),
    )
    return tracker


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _retry_merge_shard_path(source_run_dir: Path, root_run_id: str) -> Path:
    safe_root_run_id = root_run_id.replace(os.sep, "_")
    return (
        source_run_dir
        / ".consist"
        / "retry-shards"
        / f"promotion-shard-{safe_root_run_id}-{uuid.uuid4().hex}.duckdb"
    )


def _all_runs(tracker: Any) -> list[Any]:
    return list(tracker.find_runs(limit=100_000))


def _run_id_text(run: Any) -> str:
    return str(run.id).strip() if isinstance(run, RunLike) else ""


def _run_tags(run: Any) -> list[str]:
    tags = getattr(run, "tags", None) or []
    if isinstance(tags, str):
        return [tags]
    return [str(tag) for tag in tags]


def _run_sort_key(run: Any) -> tuple[str, str]:
    created_at = getattr(run, "created_at", None)
    return (
        str(created_at) if created_at is not None else "",
        _run_id_text(run),
    )


def _resolve_db_file_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.fspath(path))).expanduser().resolve()


def _validate_merge_conflict(conflict: str) -> str:
    normalized = str(conflict).strip().lower()
    if normalized not in {"error", "skip"}:
        raise ValueError("merge_conflict must be one of: error, skip")
    return normalized


def _existing_main_run_ids(main_db_path: str | os.PathLike[str]) -> set[str]:
    try:
        from consist.core.persistence import DatabaseManager
    except Exception as exc:  # pragma: no cover - depends on optional consist install
        raise RuntimeError(
            "Could not import Consist DatabaseManager for main DB inspection."
        ) from exc

    resolved_main_db_path = _resolve_db_file_path(main_db_path)
    if not resolved_main_db_path.exists():
        return set()

    db = DatabaseManager(str(resolved_main_db_path))
    try:
        with db.engine.begin() as conn:
            rows = conn.exec_driver_sql('SELECT id FROM "run"').fetchall()
        return {str(row[0]) for row in rows}
    finally:
        db.engine.dispose()


def _copy_db_file_with_wal(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_wal = Path(f"{source}.wal")
    if source_wal.exists():
        shutil.copy2(source_wal, Path(f"{destination}.wal"))


def _remove_db_file_with_wal(path: Path) -> None:
    path.unlink(missing_ok=True)
    wal_path = Path(f"{path}.wal")
    wal_path.unlink(missing_ok=True)


def _expand_run_subtree(root_run_id: str, runs: Sequence[Any]) -> list[str]:
    children_by_parent: dict[str, list[str]] = {}
    available_ids = {_run_id_text(run) for run in runs if _run_id_text(run)}
    if root_run_id not in available_ids:
        raise ValueError(f"root run ID not found in archive DB: {root_run_id}")
    for run in runs:
        run_id = _run_id_text(run)
        parent_id = getattr(run, "parent_run_id", None)
        if parent_id is None:
            continue
        children_by_parent.setdefault(str(parent_id), []).append(run_id)
    for children in children_by_parent.values():
        children.sort()

    scoped: list[str] = []
    seen: set[str] = set()
    queue: list[str] = [root_run_id]
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        scoped.append(current)
        queue.extend(
            child for child in children_by_parent.get(current, []) if child not in seen
        )
    return scoped


def _resolve_root_run_id(
    tracker: Any,
    *,
    root_run_id: Optional[str] = None,
    main_db_path: Optional[str | os.PathLike[str]] = None,
) -> tuple[str, list[str]]:
    runs = _all_runs(tracker)
    if root_run_id:
        resolved = str(root_run_id)
        return resolved, _expand_run_subtree(resolved, runs)

    inherited_run_ids = (
        _existing_main_run_ids(main_db_path) if main_db_path is not None else set()
    )
    root_candidates = [
        run
        for run in runs
        if getattr(run, "parent_run_id", None) is None
        and _run_id_text(run) not in inherited_run_ids
    ]
    pilates_candidates = [
        run for run in root_candidates if "pilates_simulation" in set(_run_tags(run))
    ]
    if pilates_candidates:
        root_candidates = pilates_candidates
    else:
        orchestrator_candidates = [
            run
            for run in root_candidates
            if str(getattr(run, "model_name", "")) == "pilates_orchestrator"
        ]
        if orchestrator_candidates:
            root_candidates = orchestrator_candidates

    root_candidates = sorted(root_candidates, key=_run_sort_key, reverse=True)
    if len(root_candidates) != 1:
        rendered = [
            {
                "id": _run_id_text(run),
                "model": str(getattr(run, "model_name", "")),
                "status": str(getattr(run, "status", "")),
                "tags": _run_tags(run),
                "created_at": str(getattr(run, "created_at", "")),
            }
            for run in root_candidates[:20]
        ]
        raise ValueError(
            "Could not resolve exactly one promotion root run. "
            "Pass --root-run-id explicitly. Candidates: "
            + json.dumps(rendered, sort_keys=True)
        )

    resolved = _run_id_text(root_candidates[0])
    return resolved, _expand_run_subtree(resolved, runs)


def _artifact_relative_path(tracker: Any, artifact: Any) -> Optional[Path]:
    container_uri = getattr(artifact, "container_uri", None)
    if not container_uri:
        return None
    fs = getattr(tracker, "fs", None)
    get_relative = getattr(fs, "get_remappable_relative_path", None)
    if not callable(get_relative):
        return None
    relative = get_relative(container_uri)
    if relative is None:
        return None
    return Path(relative)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_output_content_hashes(
    tracker: Any,
    *,
    source_run_dir: Path,
    run_id: str,
) -> dict[str, str]:
    content_hashes: dict[str, str] = {}
    outputs = tracker.get_run_outputs(str(run_id))
    for key, artifact in outputs.items():
        relative_path = _artifact_relative_path(tracker, artifact)
        if relative_path is None:
            continue
        source_path = source_run_dir / relative_path
        if not source_path.is_file():
            continue
        content_hashes[str(key)] = _sha256_file(source_path)
    return content_hashes


def _register_verified_run_output_recovery_copies(
    tracker: Any,
    *,
    source_run_dir: Path,
    recovery_run_dir: Path,
    run_ids: Optional[Iterable[str]] = None,
    verify: bool = True,
) -> int:
    register_copies = getattr(tracker, "register_run_output_recovery_copies", None)
    if not callable(register_copies):
        logger.warning(
            "Consist tracker does not expose register_run_output_recovery_copies; "
            "skipping recovery-copy metadata registration."
        )
        return 0

    selected_run_ids = (
        list(run_ids)
        if run_ids is not None
        else [_run_id_text(run) for run in _all_runs(tracker) if _run_id_text(run)]
    )
    registered = 0
    for run_id in selected_run_ids:
        content_hashes = (
            _run_output_content_hashes(
                tracker,
                source_run_dir=source_run_dir,
                run_id=str(run_id),
            )
            if verify
            else {}
        )
        result = register_copies(
            str(run_id),
            str(recovery_run_dir),
            verify=verify,
            append=True,
            content_hashes=content_hashes or None,
        )
        registered_for_run = getattr(result, "registered", {})
        registered += len(registered_for_run)
        blocked = getattr(result, "blocked", {})
        if blocked:
            logger.info(
                "Recovery-copy registration blocked %d output(s) for run %s: %s",
                len(blocked),
                run_id,
                getattr(result, "summary", blocked),
            )
    return registered


def _sync_consist_state(source_run_dir: Path, destination_run_dir: Path) -> None:
    source_consist_dir = source_run_dir / ".consist"
    if not source_consist_dir.exists():
        return
    destination_consist_dir = destination_run_dir / ".consist"
    destination_consist_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_consist_dir, destination_consist_dir, dirs_exist_ok=True)
    history_dir = destination_consist_dir / "snapshots" / "history"
    if history_dir.exists():
        logger.info(
            "Pruning excluded Consist snapshot history during sync: %s", history_dir
        )
        shutil.rmtree(history_dir, ignore_errors=True)


def _marker_path(source_run_dir: Path) -> Path:
    return source_run_dir / ".consist" / "recovery_promotion.json"


def _write_promotion_marker(
    *,
    source_run_dir: Path,
    result: PromotionResult,
) -> Path:
    marker_path = _marker_path(source_run_dir)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    marker_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return marker_path


def _sync_marker_to_destinations(
    *,
    source_run_dir: Path,
    destinations: Sequence[Path],
) -> None:
    marker_path = _marker_path(source_run_dir)
    if not marker_path.exists():
        return
    for destination_run_dir in destinations:
        target = destination_run_dir / ".consist" / "recovery_promotion.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(marker_path, target)


def _dispose_tracker(tracker: Any) -> None:
    db = getattr(tracker, "db", None)
    engine = getattr(db, "engine", None)
    if engine is not None:
        engine.dispose()


def _merge_scoped_run_db(
    *,
    source_tracker: Any,
    source_run_dir: Path,
    root_run_id: str,
    main_db_path: str | os.PathLike[str],
    conflict: str,
    include_data: bool,
    dry_run: bool,
    shard_path: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    conflict = _validate_merge_conflict(conflict)
    resolved_main_db_path = _resolve_db_file_path(main_db_path)
    main_db_preexisting = resolved_main_db_path.exists()

    try:
        from consist.core.maintenance import DatabaseMaintenance
        from consist.core.persistence import DatabaseManager
    except Exception as exc:  # pragma: no cover - depends on optional consist install
        raise RuntimeError("Could not import Consist DB maintenance helpers.") from exc

    source_db = getattr(source_tracker, "db", None)
    if source_db is None:
        raise RuntimeError("Source tracker has no database manager for DB merge.")

    source_maintenance = DatabaseMaintenance(db=source_db, run_dir=source_run_dir)
    shard_output = (
        _resolve_db_file_path(shard_path)
        if shard_path is not None
        else source_run_dir
        / ".consist"
        / f"promotion-shard-{root_run_id.replace(os.sep, '_')}.duckdb"
    )
    if not dry_run:
        shard_output.parent.mkdir(parents=True, exist_ok=True)
        export_result = source_maintenance.export(
            root_run_id,
            shard_output,
            include_data=include_data,
            include_snapshots=False,
            include_children=True,
            dry_run=False,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="pilates-promotion-") as tmpdir:
            preview_shard = Path(tmpdir) / "promotion-shard.duckdb"
            export_result = source_maintenance.export(
                root_run_id,
                preview_shard,
                include_data=include_data,
                include_snapshots=False,
                include_children=True,
                dry_run=False,
            )
            preview_target_db_path = (
                resolved_main_db_path
                if main_db_preexisting
                else Path(tmpdir) / resolved_main_db_path.name
            )
            target_db = DatabaseManager(str(preview_target_db_path))
            try:
                target_maintenance = DatabaseMaintenance(
                    db=target_db,
                    run_dir=preview_target_db_path.parent,
                )
                merge_result = target_maintenance.merge(
                    preview_shard,
                    conflict=conflict,
                    include_snapshots=False,
                    dry_run=True,
                )
            finally:
                target_db.engine.dispose()
                payload = {
                    "dry_run": True,
                    "main_db_path": str(resolved_main_db_path),
                    "main_db_seeded": not main_db_preexisting,
                    "root_run_id": root_run_id,
                    "shard_path": None,
                    "temporary_shard_path": str(preview_shard),
                    "temporary_shard_removed": True,
                    "export_result": asdict(export_result),
                    "merge_result": asdict(merge_result),
                }
            return _json_safe(payload)

    if main_db_preexisting:
        target_db = DatabaseManager(str(resolved_main_db_path))
        try:
            target_maintenance = DatabaseMaintenance(
                db=target_db,
                run_dir=resolved_main_db_path.parent,
            )
            merge_result = target_maintenance.merge(
                shard_output,
                conflict=conflict,
                include_snapshots=False,
                dry_run=False,
            )
        finally:
            target_db.engine.dispose()
        _remove_db_file_with_wal(shard_output)
        payload = {
            "dry_run": False,
            "main_db_path": str(resolved_main_db_path),
            "root_run_id": root_run_id,
            "shard_path": str(shard_output),
            "export_result": asdict(export_result),
            "merge_result": asdict(merge_result),
        }
        return _json_safe(payload)

    with tempfile.TemporaryDirectory(prefix="pilates-promotion-main-db-") as tmpdir:
        temp_main_db_path = Path(tmpdir) / resolved_main_db_path.name
        target_db = DatabaseManager(str(temp_main_db_path))
        try:
            target_maintenance = DatabaseMaintenance(
                db=target_db,
                run_dir=temp_main_db_path.parent,
            )
            merge_result = target_maintenance.merge(
                shard_output,
                conflict=conflict,
                include_snapshots=False,
                dry_run=False,
            )
        finally:
            target_db.engine.dispose()

        _copy_db_file_with_wal(temp_main_db_path, resolved_main_db_path)
        _remove_db_file_with_wal(shard_output)
        payload = {
            "dry_run": False,
            "main_db_path": str(resolved_main_db_path),
            "main_db_seeded": True,
            "root_run_id": root_run_id,
            "shard_path": str(shard_output),
            "export_result": asdict(export_result),
            "merge_result": asdict(merge_result),
        }
        return _json_safe(payload)


def _merge_scoped_run_db_with_retry(
    *,
    settings: PilatesConfig,
    source_tracker: Any,
    source_run_dir: Path,
    root_run_id: str,
    main_db_path: str | os.PathLike[str],
    conflict: str,
    include_data: bool,
    dry_run: bool,
    shard_path: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    try:
        return _merge_scoped_run_db(
            source_tracker=source_tracker,
            source_run_dir=source_run_dir,
            root_run_id=root_run_id,
            main_db_path=main_db_path,
            conflict=conflict,
            include_data=include_data,
            dry_run=dry_run,
            shard_path=shard_path,
        )
    except OperationalError as exc:
        logger.warning(
            "Consist DB merge failed once; retrying with a fresh archive tracker "
            "after a short pause: %s",
            exc,
            exc_info=True,
        )
        time.sleep(0.25)
        retry_tracker = _open_archive_tracker(settings, archive_run_dir=source_run_dir)
        if retry_tracker is None:
            raise
        try:
            retry_shard_path = _retry_merge_shard_path(
                source_run_dir,
                root_run_id,
            )
            if shard_path is not None:
                original_shard_path = _resolve_db_file_path(shard_path)
                if original_shard_path.exists():
                    try:
                        original_shard_path.unlink()
                    except OSError:
                        logger.debug(
                            "Failed to remove partial Consist shard before retry",
                            exc_info=True,
                        )
            return _merge_scoped_run_db(
                source_tracker=retry_tracker,
                source_run_dir=source_run_dir,
                root_run_id=root_run_id,
                main_db_path=main_db_path,
                conflict=conflict,
                include_data=include_data,
                dry_run=dry_run,
                shard_path=retry_shard_path,
            )
        finally:
            retry_db = getattr(retry_tracker, "db", None)
            retry_engine = getattr(retry_db, "engine", None)
            if retry_engine is not None:
                try:
                    retry_engine.dispose()
                except Exception:
                    logger.debug(
                        "Failed to dispose retry tracker engine", exc_info=True
                    )


def promote_run_to_recovery_roots(
    settings: PilatesConfig,
    archive_run_dir: Optional[str | os.PathLike[str]] = None,
    tracker: Any = None,
    roots: Optional[Iterable[str | os.PathLike[str]]] = None,
    verify: bool = True,
    root_run_id: Optional[str] = None,
    merge_main_db: Optional[str | os.PathLike[str]] = None,
    merge_conflict: str = "error",
    merge_include_data: bool = True,
    merge_dry_run: bool = False,
    merge_shard_path: Optional[str | os.PathLike[str]] = None,
    merge_only: bool = False,
    merge_db_only: bool = False,
    recovery_copy_verify: bool = True,
    *,
    dry_run: bool = False,
    verify_only: bool = False,
) -> PromotionResult:
    overall_started = time.perf_counter()
    source_run_dir = _source_archive_run_dir(settings, archive_run_dir)
    db_path = _archive_db_path(settings, archive_run_dir=source_run_dir)
    require_consist_state = db_path is not None
    if merge_main_db is not None:
        merge_conflict = _validate_merge_conflict(merge_conflict)
    if merge_only and merge_main_db is None:
        raise ValueError("--merge-only requires --merge-main-db")
    if merge_db_only and merge_main_db is None:
        raise ValueError("--merge-db-only requires --merge-main-db")
    if merge_db_only and merge_only:
        raise ValueError("--merge-db-only cannot be combined with --merge-only")

    recovery_roots = [] if merge_db_only else _recovery_roots(settings, roots=roots)

    logger.info(
        "Starting archive promotion: source=%s recovery_roots=%d db_path=%s "
        "dry_run=%s verify_only=%s merge_main_db=%s merge_only=%s "
        "merge_db_only=%s "
        "recovery_copy_verify=%s",
        source_run_dir,
        0 if merge_db_only else len(recovery_roots),
        db_path if db_path is not None else "none",
        dry_run,
        verify_only,
        merge_main_db if merge_main_db is not None else "none",
        merge_only,
        merge_db_only,
        recovery_copy_verify,
    )

    result = PromotionResult(
        source_run_dir=str(source_run_dir),
        dry_run=dry_run,
        verify_only=verify_only,
        merge_db_only=merge_db_only,
        db_path=str(db_path) if db_path is not None else None,
        marker_path=None,
    )

    opened_tracker = None
    working_tracker = tracker or cr.current_tracker()
    if (
        working_tracker is not None
        and db_path is not None
        and not dry_run
        and not verify_only
    ):
        tracker_db_path = getattr(working_tracker, "db_path", None)
        if tracker_db_path:
            try:
                resolved_tracker_db_path = (
                    Path(os.path.expandvars(os.fspath(tracker_db_path)))
                    .expanduser()
                    .resolve()
                )
            except OSError:
                resolved_tracker_db_path = None
            if (
                resolved_tracker_db_path is not None
                and resolved_tracker_db_path != db_path
            ):
                refreshed_tracker = _open_archive_tracker(
                    settings,
                    archive_run_dir=source_run_dir,
                )
                if refreshed_tracker is not None:
                    logger.info(
                        "Refreshing archive tracker from resolved DB path %s "
                        "instead of stale tracker path %s",
                        db_path,
                        tracker_db_path,
                    )
                    working_tracker = refreshed_tracker
                    opened_tracker = refreshed_tracker
    needs_tracker = (
        not dry_run
        and not verify_only
        and db_path is not None
        and (
            merge_main_db is not None or root_run_id is not None or db_path is not None
        )
    )
    if working_tracker is None and needs_tracker:
        opened_tracker = _open_archive_tracker(settings, archive_run_dir=source_run_dir)
        working_tracker = opened_tracker
    if (
        merge_main_db is not None
        and not dry_run
        and not verify_only
        and working_tracker is None
    ):
        raise RuntimeError(
            "Cannot merge into a main DB because no run-local Consist tracker "
            "could be opened for the archive run."
        )

    successful_destinations: list[Path] = []
    try:
        scoped_run_ids: list[str] = []
        resolved_root_run_id: Optional[str] = None
        if working_tracker is not None and not dry_run and not verify_only:
            resolved_root_run_id, scoped_run_ids = _resolve_root_run_id(
                working_tracker,
                root_run_id=root_run_id,
                main_db_path=merge_main_db,
            )
            result.root_run_id = resolved_root_run_id
            result.scoped_run_ids = scoped_run_ids
            logger.info(
                "Resolved promotion root run id %s with %d scoped run(s)",
                resolved_root_run_id,
                len(scoped_run_ids),
            )

        for recovery_root in recovery_roots:
            destination_run_dir = recovery_root / source_run_dir.name
            entry = RootPromotionResult(
                recovery_root=str(recovery_root),
                destination_run_dir=str(destination_run_dir),
                status="pending",
            )
            result.roots.append(entry)

            if destination_run_dir.resolve() == source_run_dir.resolve():
                entry.status = "skipped_same_as_source"
                continue

            try:
                root_started = time.perf_counter()
                source_file_count, source_dir_count, source_byte_count = (
                    _tree_inventory(source_run_dir)
                )
                logger.info(
                    "Promoting archive to recovery root %s -> %s "
                    "(files=%d dirs=%d size=%.2f GiB)",
                    recovery_root,
                    destination_run_dir,
                    source_file_count,
                    source_dir_count,
                    source_byte_count / (1024**3),
                )
                if dry_run:
                    entry.status = "dry_run"
                    continue

                if merge_only:
                    if not destination_run_dir.exists():
                        raise FileNotFoundError(
                            "Cannot merge-only promote because the promoted archive "
                            f"copy does not exist: {destination_run_dir}"
                        )
                    logger.info(
                        "Reusing existing promoted archive copy at %s",
                        destination_run_dir,
                    )
                elif not verify_only:
                    logger.info(
                        "Copying archive tree to %s with shutil.copytree(...)",
                        destination_run_dir,
                    )
                    _copy_run_tree(source_run_dir, destination_run_dir)
                    entry.copy_performed = True
                    logger.info(
                        "Finished copying archive tree to %s in %.2fs",
                        destination_run_dir,
                        time.perf_counter() - root_started,
                    )

                if verify:
                    logger.info("Verifying promoted archive at %s", destination_run_dir)
                    _verify_promoted_run_dir(
                        source_run_dir=source_run_dir,
                        destination_run_dir=destination_run_dir,
                        require_consist_state=require_consist_state,
                    )
                    entry.verified = True

                if not verify_only and working_tracker is not None:
                    logger.info(
                        "Registering recovery copies for %s (%sbyte verification)",
                        destination_run_dir,
                        "with " if recovery_copy_verify else "without ",
                    )
                    metadata_updates = _register_verified_run_output_recovery_copies(
                        working_tracker,
                        source_run_dir=source_run_dir,
                        recovery_run_dir=destination_run_dir,
                        run_ids=scoped_run_ids or None,
                        verify=recovery_copy_verify,
                    )
                    entry.artifact_metadata_updated = metadata_updates > 0

                if not verify_only:
                    successful_destinations.append(destination_run_dir)
                    if working_tracker is not None:
                        logger.info(
                            "Syncing Consist state into %s",
                            destination_run_dir,
                        )
                        for successful_destination in successful_destinations:
                            _sync_consist_state(source_run_dir, successful_destination)

                entry.status = "promoted"
                logger.info(
                    "Finished promotion for %s in %.2fs",
                    destination_run_dir,
                    time.perf_counter() - root_started,
                )
            except Exception as exc:
                entry.status = "failed"
                entry.error = str(exc)
                logger.exception(
                    "Failed promoting archive run %s to recovery root %s",
                    source_run_dir,
                    recovery_root,
                )

        if (
            merge_main_db is not None
            and not dry_run
            and not verify_only
            and working_tracker is not None
            and resolved_root_run_id is not None
        ):
            if merge_db_only or result.success:
                try:
                    merge_started = time.perf_counter()
                    logger.info(
                        "Exporting and merging root run subtree %s into %s",
                        resolved_root_run_id,
                        merge_main_db,
                    )
                    result.merge_result = _merge_scoped_run_db_with_retry(
                        settings=settings,
                        source_tracker=working_tracker,
                        source_run_dir=source_run_dir,
                        root_run_id=resolved_root_run_id,
                        main_db_path=merge_main_db,
                        conflict=merge_conflict,
                        include_data=merge_include_data,
                        dry_run=merge_dry_run,
                        shard_path=merge_shard_path,
                    )
                    logger.info(
                        "Finished Consist DB merge path in %.2fs",
                        time.perf_counter() - merge_started,
                    )
                except OperationalError as exc:
                    logger.warning(
                        "Skipping Consist DB merge for archive promotion because "
                        "the export/merge transaction failed: %s",
                        exc,
                        exc_info=True,
                    )
                    result.merge_result = {
                        "skipped": True,
                        "reason": "consist_db_merge_failed_after_retry",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "root_run_id": resolved_root_run_id,
                        "main_db_path": str(merge_main_db),
                        "retry_attempted": True,
                    }
            else:
                result.merge_result = {
                    "skipped": True,
                    "reason": "archive promotion did not complete for all roots",
                }

        if not dry_run and not verify_only and not merge_db_only:
            marker_path = _write_promotion_marker(
                source_run_dir=source_run_dir,
                result=result,
            )
            result.marker_path = str(marker_path)
            if successful_destinations:
                _sync_marker_to_destinations(
                    source_run_dir=source_run_dir,
                    destinations=successful_destinations,
                )
        logger.info(
            "Archive promotion complete in %.2fs (success=%s)",
            time.perf_counter() - overall_started,
            result.success,
        )
        log_consist_cli_instructions(
            logger=logger,
            archive_run_dir=str(source_run_dir),
            run_db_path=str(db_path) if db_path is not None else None,
            main_db_path=str(merge_main_db) if merge_main_db is not None else None,
            success=result.success,
        )
    finally:
        if opened_tracker is not None:
            _dispose_tracker(opened_tracker)

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote a completed PILATES archive run to one or more recovery roots."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the PILATES YAML config used for the run.",
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "Explicit completed archive run directory. When omitted, the helper "
            "uses run.output_directory plus run.output_run_name or the latest "
            "matching timestamped run."
        ),
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Additional recovery archive root. May be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview promotion targets without copying files or updating DB metadata.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing promoted copies without copying files or updating DB metadata.",
    )
    parser.add_argument(
        "--root-run-id",
        help=(
            "Root run ID to promote and, when --merge-main-db is set, export before "
            "merge. When omitted, the helper resolves exactly one non-inherited "
            "root run from the run-local DB."
        ),
    )
    parser.add_argument(
        "--merge-main-db",
        help=(
            "Optional central Consist DB to merge into after archive promotion. "
            "The helper exports only the resolved root run subtree from the "
            "run-local DB before merging."
        ),
    )
    parser.add_argument(
        "--merge-conflict",
        choices=("error", "skip"),
        default="error",
        help="Conflict policy passed to Consist DB merge after filtered export.",
    )
    parser.add_argument(
        "--merge-dry-run",
        action="store_true",
        help=(
            "Export the resolved root run subtree to a temporary shard and dry-run "
            "the merge without mutating the main DB."
        ),
    )
    parser.add_argument(
        "--no-recovery-copy-verify",
        action="store_false",
        dest="recovery_copy_verify",
        default=True,
        help=(
            "Skip byte-level verification when registering promoted recovery copies "
            "for this archive promotion."
        ),
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help=(
            "Reuse an already-copied promoted archive and run only the Consist DB "
            "export/merge path."
        ),
    )
    parser.add_argument(
        "--merge-db-only",
        action="store_true",
        help=(
            "Run only the Consist DB export/merge path and skip archive promotion "
            "to recovery roots."
        ),
    )
    parser.add_argument(
        "--merge-shard-path",
        help=(
            "Optional path for the filtered export shard used for a real merge. "
            "Defaults to .consist/promotion-shard-<root-run-id>.duckdb inside "
            "the source run directory."
        ),
    )
    parser.add_argument(
        "--no-merge-include-data",
        action="store_false",
        dest="merge_include_data",
        default=True,
        help=(
            "Do not include run-scoped Consist global-table data in the filtered "
            "export shard."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    settings = load_config(args.config)
    result = promote_run_to_recovery_roots(
        settings,
        archive_run_dir=args.run_dir,
        roots=args.root,
        root_run_id=args.root_run_id,
        merge_main_db=args.merge_main_db,
        merge_conflict=args.merge_conflict,
        merge_include_data=bool(args.merge_include_data),
        merge_dry_run=bool(args.merge_dry_run),
        merge_only=bool(args.merge_only),
        merge_db_only=bool(args.merge_db_only),
        recovery_copy_verify=bool(args.recovery_copy_verify),
        merge_shard_path=args.merge_shard_path,
        dry_run=bool(args.dry_run),
        verify_only=bool(args.verify_only),
    )

    print(json.dumps(_json_safe(result.to_dict()), indent=2, sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
