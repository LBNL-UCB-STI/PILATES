from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def save_step_manifest(manifest: Dict[str, Any], manifest_path: Path) -> None:
    """
    Save a step manifest YAML file to disk.

    Parameters
    ----------
    manifest : dict
        Manifest content to serialize.
    manifest_path : Path
        Destination path for the YAML manifest.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    try:
        from pilates.utils.coupler_helpers import enqueue_archive_copy

        enqueue_archive_copy(key="workflow_manifest", path=manifest_path)
    except Exception as exc:
        logger.warning(
            "Failed to enqueue workflow manifest for archive copy (%s): %s",
            manifest_path,
            exc,
        )


def load_step_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a step manifest YAML file from disk.

    Parameters
    ----------
    manifest_path : Path
        Path to the YAML manifest.

    Returns
    -------
    dict or None
        Parsed manifest dict, or None if the file does not exist.
    """
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.error("Failed to load step manifest at %s: %s", manifest_path, exc)
        raise
    if not isinstance(data, dict):
        raise ValueError(f"Step manifest at {manifest_path} is not a dict")
    return data
