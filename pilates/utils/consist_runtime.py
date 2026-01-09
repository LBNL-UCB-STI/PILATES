"""
pilates/utils/consist_runtime.py

Optional Consist runtime wrappers.
These helpers provide no-op fallbacks when Consist is unavailable.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

try:
    import consist
    from consist import Tracker
except ImportError:  # Consist optional dependency
    consist = None
    Tracker = None

_warned_disabled = False


def _consist_enabled(settings) -> bool:
    return bool(getattr(settings.shared.database, "use_consist", False))


def consist_available(settings: Optional[Any] = None) -> bool:
    if consist is None:
        return False
    if settings is None:
        return True
    return _consist_enabled(settings)


def _warn_disabled() -> None:
    global _warned_disabled
    if not _warned_disabled:
        _warned_disabled = True
        logger.warning("Consist disabled/unavailable; provenance is skipped.")


@contextmanager
def use_tracker(tracker):
    if consist is None:
        _warn_disabled()
        yield None
        return
    with consist.use_tracker(tracker) as tr:
        yield tr


@contextmanager
def scenario(name: str, tracker=None, *, enabled: Optional[bool] = None, **kwargs):
    if enabled is False or consist is None:
        _warn_disabled()
        yield _NoopScenario()
        return
    with consist.scenario(name, tracker=tracker, **kwargs) as sc:
        yield sc


def log_input(path, key=None, **meta):
    if consist is None:
        _warn_disabled()
        return None
    return consist.log_input(path, key=key, **meta)


def log_output(path, key=None, **meta):
    if consist is None:
        _warn_disabled()
        return None
    return consist.log_output(path, key=key, **meta)


def log_artifacts(mapping, **meta):
    if consist is None:
        _warn_disabled()
        return None
    return consist.log_artifacts(mapping, **meta)


def log_meta(**meta):
    if consist is None:
        _warn_disabled()
        return None
    return consist.log_meta(**meta)


def current_run():
    if consist is None:
        return None
    return consist.current_run()


def set_tracker(tracker):
    """Set a global tracker for Consist API entrypoints."""
    if consist is None:
        return
    consist.use_tracker(tracker).__enter__()


def current_tracker():
    """Get the current global tracker (or None if not set)."""
    if consist is None:
        return None
    return consist.current_tracker() if hasattr(consist, "current_tracker") else None


class _NoopCoupler:
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._store.get(key, default)

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        return self._store.pop(key, default)

    def update(self, mapping: Dict[str, Any]) -> None:
        self._store.update(mapping)

    def declare_outputs(self, *args, **kwargs) -> None:
        return None

    def adopt_cached_output(self, *args, **kwargs) -> None:
        return None


class _NoopScenario:
    def __init__(self):
        self.coupler = _NoopCoupler()

    @contextmanager
    def trace(self, *args, **kwargs):
        yield None

    def declare_outputs(self, *args, **kwargs) -> None:
        return None

    def coupler_schema(self, schema):
        return schema(self.coupler)

    def run(self, fn=None, name: Optional[str] = None, **kwargs):
        if fn is None:
            return None
        runtime_kwargs = kwargs.get("runtime_kwargs") or {}
        return fn(**runtime_kwargs)
