from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Optional, Sequence

from pilates.config import PilatesConfig, load_config
from pilates.runtime.consist_audit import emit_artifact_lifecycle_audit_event
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_db_snapshot import resolve_consist_db_paths

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
    db_path: Optional[str]
    marker_path: Optional[str]
    roots: list[RootPromotionResult] = field(default_factory=list)

    @property
    def succeeded_roots(self) -> list[RootPromotionResult]:
        return [root for root in self.roots if root.status == "promoted"]

    @property
    def failed_roots(self) -> list[RootPromotionResult]:
        return [root for root in self.roots if root.status == "failed"]

    @property
    def success(self) -> bool:
        return bool(self.roots) and not self.failed_roots

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
        (
            path for path in archive_root.glob(pattern)
            if path.is_dir()
        ),
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


def _copy_run_tree(source_run_dir: Path, destination_run_dir: Path) -> None:
    destination_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, destination_run_dir, dirs_exist_ok=True)


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


def _all_output_artifacts(tracker: Any) -> list[Any]:
    artifacts: list[Any] = []
    seen_artifact_ids: set[str] = set()
    for run in tracker.find_runs(limit=100_000):
        run_artifacts = tracker.get_artifacts_for_run(str(run.id))
        run_outputs = (
            run_artifacts.outputs if run_artifacts.outputs is not None else {}
        )
        for artifact in list(run_outputs.values()):
            artifact_id = str(getattr(artifact, "id", ""))
            if artifact_id and artifact_id in seen_artifact_ids:
                continue
            if artifact_id:
                seen_artifact_ids.add(artifact_id)
            artifacts.append(artifact)
    return artifacts


def _update_artifact_recovery_roots(tracker: Any, recovery_run_dir: Path) -> int:
    updated = 0
    for artifact in _all_output_artifacts(tracker):
        tracker.set_artifact_recovery_roots(
            artifact,
            [str(recovery_run_dir)],
            append=True,
        )
        updated += 1
    return updated


def _sync_consist_state(source_run_dir: Path, destination_run_dir: Path) -> None:
    source_consist_dir = source_run_dir / ".consist"
    if not source_consist_dir.exists():
        return
    destination_consist_dir = destination_run_dir / ".consist"
    destination_consist_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_consist_dir, destination_consist_dir, dirs_exist_ok=True)


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
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def promote_run_to_recovery_roots(
    settings: PilatesConfig,
    archive_run_dir: Optional[str | os.PathLike[str]] = None,
    tracker: Any = None,
    roots: Optional[Iterable[str | os.PathLike[str]]] = None,
    verify: bool = True,
    *,
    dry_run: bool = False,
    verify_only: bool = False,
) -> PromotionResult:
    source_run_dir = _source_archive_run_dir(settings, archive_run_dir)
    recovery_roots = _recovery_roots(settings, roots=roots)
    db_path = _archive_db_path(settings, archive_run_dir=source_run_dir)
    require_consist_state = db_path is not None

    result = PromotionResult(
        source_run_dir=str(source_run_dir),
        dry_run=dry_run,
        verify_only=verify_only,
        db_path=str(db_path) if db_path is not None else None,
        marker_path=None,
    )

    opened_tracker = None
    working_tracker = tracker or cr.current_tracker()
    if working_tracker is None and not dry_run and not verify_only and db_path is not None:
        opened_tracker = _open_archive_tracker(settings, archive_run_dir=source_run_dir)
        working_tracker = opened_tracker

    successful_destinations: list[Path] = []
    try:
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
                if dry_run:
                    entry.status = "dry_run"
                    continue

                if not verify_only:
                    _copy_run_tree(source_run_dir, destination_run_dir)
                    entry.copy_performed = True

                if verify:
                    _verify_promoted_run_dir(
                        source_run_dir=source_run_dir,
                        destination_run_dir=destination_run_dir,
                        require_consist_state=require_consist_state,
                    )
                    entry.verified = True

                if not verify_only and working_tracker is not None:
                    _update_artifact_recovery_roots(
                        working_tracker,
                        destination_run_dir,
                    )
                    entry.artifact_metadata_updated = True

                if not verify_only:
                    successful_destinations.append(destination_run_dir)
                    if working_tracker is not None:
                        for successful_destination in successful_destinations:
                            _sync_consist_state(source_run_dir, successful_destination)

                entry.status = "promoted"
            except Exception as exc:
                entry.status = "failed"
                entry.error = str(exc)
                logger.exception(
                    "Failed promoting archive run %s to recovery root %s",
                    source_run_dir,
                    recovery_root,
                )

        if not dry_run and not verify_only:
            marker_path = _write_promotion_marker(source_run_dir=source_run_dir, result=result)
            result.marker_path = str(marker_path)
            if successful_destinations:
                _sync_marker_to_destinations(
                    source_run_dir=source_run_dir,
                    destinations=successful_destinations,
                )
        for entry in result.roots:
            emit_artifact_lifecycle_audit_event(
                run_dir=source_run_dir,
                event_type="promotion_status",
                recovery_root=entry.recovery_root,
                destination_run_dir=entry.destination_run_dir,
                status=entry.status,
                copy_performed=entry.copy_performed,
                verified=entry.verified,
                artifact_metadata_updated=entry.artifact_metadata_updated,
                db_path=str(db_path) if db_path is not None else None,
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_config(args.config)
    result = promote_run_to_recovery_roots(
        settings,
        archive_run_dir=args.run_dir,
        roots=args.root,
        dry_run=bool(args.dry_run),
        verify_only=bool(args.verify_only),
    )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if not result.failed_roots else 1


if __name__ == "__main__":
    raise SystemExit(main())
