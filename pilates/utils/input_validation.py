import logging
import os
from typing import Iterable, Optional


logger = logging.getLogger(__name__)


def resolve_input_path(value):
    if value is None:
        return None
    return getattr(value, "path", None) or getattr(value, "uri", None) or value


def _is_uri(path) -> bool:
    return isinstance(path, str) and "://" in path


def validate_inputs(
    inputs: dict,
    required_keys: Optional[Iterable[str]] = None,
    optional_keys: Optional[Iterable[str]] = None,
    *,
    context: str = "",
    min_mtime: Optional[float] = None,
) -> None:
    required_keys = list(required_keys or [])
    optional_keys = list(optional_keys or [])

    for key in required_keys:
        if key not in inputs or inputs[key] is None:
            raise FileNotFoundError(
                f"Missing required input '{key}'{f' for {context}' if context else ''}."
            )
        path = resolve_input_path(inputs[key])
        if path is None:
            raise FileNotFoundError(
                f"Missing required input '{key}'{f' for {context}' if context else ''}."
            )
        if _is_uri(path):
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input '{key}' not found at {path}.")
        if min_mtime is not None and os.path.getmtime(path) < min_mtime:
            raise RuntimeError(f"Required input '{key}' appears stale at {path}.")

    for key in optional_keys:
        if key not in inputs or inputs[key] is None:
            logger.warning(
                "Optional input '%s' missing%s.",
                key,
                f" for {context}" if context else "",
            )
            continue
        path = resolve_input_path(inputs[key])
        if path is None:
            logger.warning(
                "Optional input '%s' missing%s.",
                key,
                f" for {context}" if context else "",
            )
            continue
        if _is_uri(path):
            continue
        if not os.path.exists(path):
            logger.warning("Optional input '%s' not found at %s.", key, path)
            continue
        if min_mtime is not None and os.path.getmtime(path) < min_mtime:
            logger.warning("Optional input '%s' appears stale at %s.", key, path)
