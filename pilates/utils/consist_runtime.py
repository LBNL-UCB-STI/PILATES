"""
pilates/utils/consist_runtime.py

Optional Consist runtime wrappers.
These helpers provide no-op fallbacks when Consist is unavailable.
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import contextmanager
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from pilates.utils.consist_types import (
    ArtifactLike,
    RunResultLike,
    ScenarioLike,
    TrackerLike,
)

logger = logging.getLogger(__name__)

consist: Optional[ModuleType]
try:
    consist = import_module("consist")
except ImportError:  # Consist optional dependency
    consist = None

_warned_disabled = False
_enabled_override: Optional[bool] = None


def _consist_enabled(settings: Any) -> bool:
    return bool(getattr(settings.shared.database, "use_consist", False))


def consist_available(settings: Optional[Any] = None) -> bool:
    if consist is None:
        return False
    if settings is not None:
        return _consist_enabled(settings)
    if _enabled_override is not None:
        return _enabled_override
    return True


def _warn_disabled() -> None:
    global _warned_disabled
    if not _warned_disabled:
        _warned_disabled = True
        logger.warning("Consist disabled/unavailable; provenance is skipped.")


def _is_enabled(enabled: Optional[bool] = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    if _enabled_override is not None:
        return _enabled_override
    return True


def set_enabled(enabled: Optional[bool]) -> None:
    global _enabled_override
    _enabled_override = bool(enabled) if enabled is not None else None


def create_tracker(
    *,
    settings: Optional[Any] = None,
    enabled: Optional[bool] = None,
    tracker_factory: Optional[Callable[[], TrackerLike]] = None,
    **tracker_kwargs: Any,
) -> Optional[TrackerLike]:
    resolved_enabled = _consist_enabled(settings) if enabled is None else enabled
    set_enabled(resolved_enabled)

    if consist is None:
        if resolved_enabled:
            _warn_disabled()
        return None

    if tracker_factory is None:
        tracker_factory = lambda: consist.Tracker(**tracker_kwargs)

    return consist.create_tracker(
        enabled=bool(resolved_enabled),
        tracker_factory=tracker_factory,
    )


@contextmanager
def use_tracker(tracker: Optional[TrackerLike]) -> Iterator[Optional[TrackerLike]]:
    if consist is None:
        _warn_disabled()
        yield None
        return
    with consist.use_tracker(tracker) as tr:
        yield tr


@contextmanager
def scenario(
    name: str,
    tracker: Optional[TrackerLike] = None,
    *,
    enabled: Optional[bool] = None,
    **kwargs,
) -> Iterator[ScenarioLike]:
    if consist is None:
        _warn_disabled()
        yield _NoopScenario()
        return
    resolved_enabled = _is_enabled(enabled)
    if not resolved_enabled:
        _warn_disabled()
    with consist.scenario(
        name, tracker=tracker, enabled=resolved_enabled, **kwargs
    ) as sc:
        yield sc


def log_input(
    path: Any,
    key: Optional[str] = None,
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    resolved_enabled = _is_enabled(enabled)
    if consist is None:
        _warn_disabled()
        return _NoopArtifact(path)
    if not resolved_enabled:
        _warn_disabled()
    return consist.log_input(path, key=key, enabled=resolved_enabled, **meta)


def log_output(
    path: Any,
    key: Optional[str] = None,
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    resolved_enabled = _is_enabled(enabled)
    if not resolved_enabled or consist is None:
        resolved_path = _resolve_artifact_path(path)
        if resolved_path and not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Output path does not exist: {resolved_path}")
        if consist is None:
            _warn_disabled()
            return _NoopArtifact(resolved_path or path)
        if not resolved_enabled:
            _warn_disabled()
    return consist.log_output(path, key=key, enabled=resolved_enabled, **meta)


def log_artifacts(
    mapping: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[Dict[str, ArtifactLike]]:
    resolved_enabled = _is_enabled(enabled)
    if consist is None:
        _warn_disabled()
        return {
            key: _NoopArtifact(_resolve_artifact_path(value) or value)
            for key, value in mapping.items()
        }
    if not resolved_enabled:
        _warn_disabled()
    return consist.log_artifacts(mapping, enabled=resolved_enabled, **meta)


def log_meta(**meta: Any) -> None:
    if not _is_enabled() or consist is None:
        _warn_disabled()
        return None
    return consist.log_meta(**meta)


def current_run() -> Optional[Any]:
    if not _is_enabled() or consist is None:
        return None
    return consist.current_run()


def set_tracker(tracker: Optional[TrackerLike]) -> None:
    """Set a global tracker for Consist API entrypoints."""
    if consist is None:
        return
    if hasattr(consist, "set_current_tracker"):
        consist.set_current_tracker(tracker)
    else:
        consist.use_tracker(tracker).__enter__()


def current_tracker() -> Optional[TrackerLike]:
    """Get the current global tracker (or None if not set)."""
    if not _is_enabled() or consist is None:
        return None
    try:
        return consist.current_tracker()
    except Exception:
        return None


def require_runtime_kwargs(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if consist is None or not hasattr(consist, "require_runtime_kwargs"):
        return _noop_require_runtime_kwargs(*names)
    return consist.require_runtime_kwargs(*names)


def _noop_require_runtime_kwargs(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if not names:
        raise ValueError("require_runtime_kwargs requires at least one name.")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("require_runtime_kwargs expects non-empty string names.")

    def decorator(func):
        func.__consist_runtime_required__ = tuple(names)
        return func

    return decorator


def _resolve_artifact_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, str):
        return value
    if hasattr(value, "path"):
        return os.fspath(getattr(value, "path"))
    if hasattr(value, "uri"):
        return str(getattr(value, "uri"))
    return None


class _NoopArtifact:
    def __init__(self, path: Any) -> None:
        self._path = str(path)

    @property
    def path(self) -> str:
        return self._path

    @property
    def uri(self) -> str:
        return self._path


class _NoopCoupler:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._store.get(key, default)

    def update(self, mapping: Dict[str, Any]) -> None:
        self._store.update(mapping)

    def declare_outputs(self, *args: Any, **kwargs: Any) -> None:
        return None

    def collect_by_keys(
        self,
        artifacts: Dict[str, Any],
        *keys: str,
        prefix: str = "",
    ) -> Dict[str, Any]:
        collected: Dict[str, Any] = {}
        for key in keys:
            if key not in artifacts:
                raise KeyError(f"Missing artifact for key {key!r}.")
            coupler_key = f"{prefix}{key}"
            artifact = artifacts[key]
            self.set(coupler_key, artifact)
            collected[coupler_key] = artifact
        return collected


class _NoopScenario:
    def __init__(self) -> None:
        self.coupler = _NoopCoupler()

    @contextmanager
    def trace(self, *args: Any, **kwargs: Any) -> Iterator[None]:
        yield None

    def declare_outputs(self, *args, **kwargs) -> None:
        return None

    def coupler_schema(self, schema: Any) -> Any:
        return schema(self.coupler)

    def run(
        self,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> RunResultLike:
        if fn is None:
            return _NoopRunResult()
        runtime_kwargs = kwargs.get("runtime_kwargs") or {}
        required = getattr(fn, "__consist_runtime_required__", None)
        if required:
            missing = [name for name in required if name not in runtime_kwargs]
            if missing:
                raise ValueError(
                    "Missing runtime_kwargs for noop scenario: " f"{', '.join(missing)}"
                )
        sig = inspect.signature(fn)
        try:
            sig.bind_partial(**runtime_kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Noop scenario could not bind arguments for {fn.__name__!r}: {exc}"
            ) from exc
        result = fn(**runtime_kwargs)

        outputs: Dict[str, Any] = {}
        output_paths = kwargs.get("output_paths")
        if output_paths:
            outputs = {
                key: _NoopArtifact(value) for key, value in dict(output_paths).items()
            }
        elif hasattr(result, "to_mapping") and callable(result.to_mapping):
            outputs = {
                key: _NoopArtifact(value)
                for key, value in dict(result.to_mapping()).items()
            }

        return _NoopRunResult(outputs=outputs)

    def collect_by_keys(
        self, artifacts: Dict[str, Any], *keys: str, prefix: str = ""
    ) -> Dict[str, Any]:
        return self.coupler.collect_by_keys(artifacts, *keys, prefix=prefix)


class _NoopRunResult:
    def __init__(self, outputs: Optional[Dict[str, Any]] = None) -> None:
        self.outputs = outputs or {}
        self.cache_hit = False
