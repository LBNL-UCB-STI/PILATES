import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_initial_linkstats_path(settings, workspace) -> Optional[str]:
    """
    Resolve the initial BEAM warm-start linkstats path from settings.

    Initial warmstart is opt-in. When ``beam.warmstart_linkstats_path`` is unset,
    no router-directory probing occurs. Relative paths are resolved against the
    mutable BEAM region input root, and ``{router_directory}`` is expanded from
    ``beam.router_directory`` when present.
    """
    beam_settings = getattr(settings, "beam", None)
    run_settings = getattr(settings, "run", None)
    if beam_settings is None or run_settings is None:
        return None

    region = getattr(run_settings, "region", None)
    if not region:
        return None

    configured_path = getattr(beam_settings, "warmstart_linkstats_path", None)
    if not configured_path:
        return None

    beam_region_root = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        region,
    )

    candidate = os.path.expanduser(os.fspath(configured_path))
    router_directory = getattr(beam_settings, "router_directory", None)
    if "{router_directory}" in candidate:
        if not router_directory:
            logger.warning(
                "[BEAM warmstart] warmstart_linkstats_path uses {router_directory} "
                "but beam.router_directory is not configured."
            )
            return None
        candidate = candidate.replace("{router_directory}", router_directory)

    if not os.path.isabs(candidate):
        candidate = os.path.join(beam_region_root, candidate)
    candidate = os.path.normpath(candidate)

    if os.path.exists(candidate):
        return candidate

    logger.warning(
        "[BEAM warmstart] Configured warmstart_linkstats_path not found: %s",
        candidate,
    )
    return None


def find_last_run_output_plans(
    output_path: Path,
    dir_prefix: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Mirror BEAM's LastRunOutputSource.findLastRunOutputPlans selection logic.

    Returns
    -------
    tuple
        (plans_path, experienced_plans_path) or (None, None) if not found.
    """
    plans_paths: List[Path] = []
    experienced_paths: List[Path] = []

    latest_output_dir = _find_latest_output_directory(output_path, dir_prefix)
    last_iteration_dirs = _find_all_last_iteration_directories(output_path, dir_prefix)

    for it_dir, it_number in last_iteration_dirs:
        if latest_output_dir is not None:
            output_plans = latest_output_dir / "output_plans.xml.gz"
            if output_plans.exists():
                plans_paths.append(output_plans)
            else:
                fallback = _find_file(it_dir, it_number, "plans.csv.gz")
                if fallback is not None:
                    plans_paths.append(fallback)
        else:
            fallback = _find_file(it_dir, it_number, "plans.csv.gz")
            if fallback is not None:
                plans_paths.append(fallback)

        if latest_output_dir is not None:
            output_experienced = latest_output_dir / "experienced_plans.xml.gz"
            if output_experienced.exists():
                experienced_paths.append(output_experienced)
            else:
                fallback = _find_file(it_dir, it_number, "experienced_plans.xml.gz")
                if fallback is not None:
                    experienced_paths.append(fallback)
        else:
            fallback = _find_file(it_dir, it_number, "experienced_plans.xml.gz")
            if fallback is not None:
                experienced_paths.append(fallback)

    plans_path = plans_paths[0] if plans_paths else None
    experienced_path = experienced_paths[0] if experienced_paths else None

    logger.debug(
        "[BEAM warmstart] selected plans=%s experienced=%s",
        plans_path,
        experienced_path,
    )
    return plans_path, experienced_path


def _find_file(iteration_dir: Path, iteration_number: int, file_name: str) -> Optional[Path]:
    file_path = iteration_dir / f"{iteration_number}.{file_name}"
    return file_path if file_path.exists() else None


def _find_latest_output_directory(output_path: Path, dir_prefix: str) -> Optional[Path]:
    dirs = [
        path
        for path in _find_dirs(output_path, dir_prefix)
        if (path / "ITERS").exists()
    ]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name, reverse=True)[0]


def _find_all_last_iteration_directories(
    output_path: Path, dir_prefix: str
) -> List[Tuple[Path, int]]:
    output_dirs = [
        path
        for path in _find_dirs(output_path, dir_prefix)
        if (path / "ITERS").exists()
    ]
    output_dirs = sorted(output_dirs, key=lambda p: p.name, reverse=True)

    it_dirs: List[Tuple[Path, int]] = []
    for output_dir in output_dirs:
        iters_root = output_dir / "ITERS"
        if not iters_root.exists():
            continue
        for it_dir in sorted(iters_root.iterdir(), key=lambda p: p.name, reverse=True):
            if not it_dir.is_dir():
                continue
            if not it_dir.name.startswith("it."):
                continue
            try:
                it_number = int(it_dir.name.split(".", 1)[1])
            except (IndexError, ValueError):
                logger.debug(
                    "[BEAM warmstart] skipping non-numeric iteration dir: %s",
                    it_dir,
                )
                continue
            it_dirs.append((it_dir, it_number))
    it_dirs.sort(key=lambda item: item[1], reverse=True)
    return it_dirs


def _find_dirs(parent_dir: Path, prefix: str) -> List[Path]:
    try:
        return [
            path
            for path in parent_dir.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ]
    except FileNotFoundError:
        logger.debug("[BEAM warmstart] output path not found: %s", parent_dir)
        return []
