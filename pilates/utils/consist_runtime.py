"""
pilates/utils/consist_runtime.py

Consist runtime wrappers.

Consist is a mandatory dependency for PILATES. This module keeps a thin,
test-friendly layer around the public Consist API while preserving local
helpers (path normalization and schema hints).
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, TypeVar, cast

from pilates.utils.consist_types import (
    ArtifactLike,
    CouplerProtocol,
    RunResultLike,
    ScenarioWithCoupler,
    TrackerLike,
)

logger = logging.getLogger(__name__)

import consist

_enabled_override: Optional[bool] = None


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
    resolved_enabled = _is_enabled(enabled)
    set_enabled(resolved_enabled)
    if "hashing_strategy" not in tracker_kwargs:
        # Avoid expensive full-directory hashing by default.
        tracker_kwargs["hashing_strategy"] = "fast"

    if tracker_factory is None:
        tracker_factory = lambda: consist.Tracker(**tracker_kwargs)

    return consist.create_tracker(
        enabled=resolved_enabled,
        tracker_factory=tracker_factory,
    )


@contextmanager
def use_tracker(tracker: Optional[TrackerLike]) -> Iterator[Optional[TrackerLike]]:
    with consist.use_tracker(tracker) as tr:
        yield tr


@contextmanager
def scenario(
    name: str,
    tracker: Optional[TrackerLike] = None,
    *,
    enabled: Optional[bool] = None,
    **kwargs: Any,
) -> Iterator[ScenarioWithCoupler]:
    resolved_enabled = _is_enabled(enabled)
    if not resolved_enabled:
        yield _NoopScenario()
        return
    with consist.scenario(name, tracker=tracker, enabled=True, **kwargs) as sc:
        # Narrow for callers: PILATES requires a coupler-capable scenario.
        yield cast(ScenarioWithCoupler, sc)


def log_input(
    path: Any,
    key: Optional[str] = None,
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    resolved_enabled = _is_enabled(enabled)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    path = _normalize_path(path)
    meta = _maybe_fast_hash_h5(path, meta)
    if not resolved_enabled:
        return _NoopArtifact(_resolve_artifact_path(path) or path)
    try:
        return consist.log_input(path, key=key, enabled=True, **meta)
    except RuntimeError:
        logger.debug("Skipping input artifact logging outside active Consist run.")
        return _NoopArtifact(_resolve_artifact_path(path) or path)


def log_output(
    path: Any,
    key: Optional[str] = None,
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    resolved_enabled = _is_enabled(enabled)
    if key and "schema" not in meta:
        schema = _schema_for_key(key)
        if schema is not None:
            meta = {**meta, "schema": schema}
    path = _normalize_path(path)
    meta = _maybe_fast_hash_h5(path, meta)
    if not resolved_enabled:
        resolved_path = _resolve_artifact_path(path)
        if resolved_path and not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Output path does not exist: {resolved_path}")
        return _NoopArtifact(resolved_path or path)
    try:
        return consist.log_output(path, key=key, enabled=True, **meta)
    except RuntimeError:
        logger.debug("Skipping output artifact logging outside active Consist run.")
        return _NoopArtifact(_resolve_artifact_path(path) or path)


def log_h5_container(
    path: Any,
    key: Optional[str] = None,
    *,
    direction: str = "input",
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    """
    Log an HDF5 container using the Tracker API when available.

    Falls back to log_input/log_output if the Tracker method is unavailable.
    """
    resolved_enabled = _is_enabled(enabled)
    if not resolved_enabled:
        return (
            log_output(path, key=key, enabled=False, **meta)
            if direction == "output"
            else log_input(path, key=key, enabled=False, **meta)
        )
    tracker = current_tracker()
    if tracker is None or not hasattr(tracker, "log_h5_container"):
        if direction == "output":
            return log_output(path, key=key, enabled=resolved_enabled, **meta)
        return log_input(path, key=key, enabled=resolved_enabled, **meta)
    return tracker.log_h5_container(path, key=key, direction=direction, **meta)


def log_h5_table(
    path: Any,
    key: Optional[str] = None,
    *,
    table_path: str,
    direction: str = "input",
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[ArtifactLike]:
    """
    Log a single HDF5 dataset using the Tracker API when available.

    Requires table_path for schema profiling. Falls back to log_input/log_output
    if the Tracker method is unavailable.
    """
    resolved_enabled = _is_enabled(enabled)
    if not resolved_enabled:
        return (
            log_output(path, key=key, enabled=False, **meta)
            if direction == "output"
            else log_input(path, key=key, enabled=False, **meta)
        )
    tracker = current_tracker()
    if tracker is None or not hasattr(tracker, "log_h5_table"):
        if direction == "output":
            return log_output(path, key=key, enabled=resolved_enabled, **meta)
        return log_input(path, key=key, enabled=resolved_enabled, **meta)
    artifact = tracker.log_h5_table(
        path, key=key, table_path=table_path, direction=direction, **meta
    )
    return _ensure_legacy_table_path_meta(artifact, table_path=table_path)


def log_artifacts(
    mapping: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    **meta: Any,
) -> Optional[Dict[str, ArtifactLike]]:
    resolved_enabled = _is_enabled(enabled)
    normalized = {key: _normalize_path(value) for key, value in mapping.items()}
    if not resolved_enabled:
        return {
            key: _NoopArtifact(_resolve_artifact_path(value) or value)
            for key, value in normalized.items()
        }
    try:
        try:
            return consist.log_artifacts(outputs=normalized, enabled=True, **meta)
        except TypeError:
            return consist.log_artifacts(normalized, enabled=True, **meta)
    except RuntimeError:
        logger.debug("Skipping artifact logging outside active Consist run.")
        return {
            key: _NoopArtifact(_resolve_artifact_path(value) or value)
            for key, value in normalized.items()
        }


def _schema_for_key(key: str) -> Optional[Any]:
    try:
        from pilates.database.schema.registry import get_schema_for_key
    except Exception:
        return None
    return get_schema_for_key(key)


def log_meta(**meta: Any) -> None:
    if not _is_enabled():
        return None
    try:
        return consist.log_meta(**meta)
    except RuntimeError:
        logger.debug("Skipping metadata logging outside active Consist run.")
        return None


def current_run() -> Optional[Any]:
    if not _is_enabled():
        return None
    try:
        return consist.current_run()
    except RuntimeError:
        return None


def set_tracker(tracker: Optional[TrackerLike]) -> None:
    """Set a global tracker for Consist API entrypoints."""
    if hasattr(consist, "set_current_tracker"):
        consist.set_current_tracker(tracker)
    else:
        consist.use_tracker(tracker).__enter__()


def current_tracker() -> Optional[TrackerLike]:
    """Get the current global tracker (or None if not set)."""
    try:
        return consist.current_tracker()
    except Exception:
        return None


def require_runtime_kwargs(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return consist.require_runtime_kwargs(*names)


F = TypeVar("F", bound=Callable[..., Any])


def _noop_require_runtime_kwargs(
    *names: str,
) -> Callable[[F], F]:
    if not names:
        raise ValueError("require_runtime_kwargs requires at least one name.")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("require_runtime_kwargs expects non-empty string names.")

    def decorator(func: F) -> F:
        func.__consist_runtime_required__ = tuple(names)
        return func

    return decorator


def _normalize_path(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        path = os.fspath(value)
    elif isinstance(value, str):
        path = value
    else:
        return value
    if "://" in path:
        return value
    if os.path.isabs(path):
        return os.path.realpath(path)
    return value


def _maybe_fast_hash_h5(path: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    if "hashing_strategy" in meta:
        return meta
    resolved = _resolve_artifact_path(path)
    if not resolved:
        return meta
    lowered = resolved.lower()
    if lowered.endswith((".h5", ".hdf5")):
        return {**meta, "hashing_strategy": "fast"}
    return meta


def _resolve_artifact_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        return os.fspath(_normalize_path(value))
    if isinstance(value, str):
        return _normalize_path(value)
    if hasattr(value, "path"):
        return os.fspath(getattr(value, "path"))
    if hasattr(value, "container_uri"):
        return str(getattr(value, "container_uri"))
    if hasattr(value, "uri"):
        return str(getattr(value, "uri"))
    return None


def _ensure_legacy_table_path_meta(
    artifact: Optional[ArtifactLike], *, table_path: str
) -> Optional[ArtifactLike]:
    """
    Backward-compatibility shim for callers/tests that read table_path from meta.

    Newer Consist builds may keep table_path as a first-class field without
    duplicating it in artifact.meta.
    """
    if artifact is None:
        return None
    artifact_meta = getattr(artifact, "meta", None)
    if isinstance(artifact_meta, dict):
        artifact_meta.setdefault("table_path", table_path)
        return artifact
    if hasattr(artifact, "model_copy"):
        try:
            normalized_meta = {}
            if isinstance(artifact_meta, Mapping):
                normalized_meta = dict(artifact_meta)
            normalized_meta.setdefault("table_path", table_path)
            return artifact.model_copy(update={"meta": normalized_meta})
        except Exception:
            return artifact
    try:
        setattr(artifact, "meta", {"table_path": table_path})
    except Exception:
        pass
    return artifact


class _NoopArtifact:
    def __init__(self, path: Any) -> None:
        self._path = str(path)
        self.meta: Dict[str, Any] = {}

    @property
    def path(self) -> str:
        return self._path

    @property
    def container_uri(self) -> str:
        return self._path

    @property
    def uri(self) -> str:
        return self._path

    @property
    def table_path(self) -> Optional[str]:
        return None

    @property
    def array_path(self) -> Optional[str]:
        return None


class _NoopCoupler:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._store.get(key, default)

    def update(self, mapping: Mapping[str, Any]) -> None:
        self._store.update(mapping)

    def declare_outputs(self, *args: Any, **kwargs: Any) -> None:
        return None

    def view(self, namespace: str) -> "_NoopCouplerView":
        normalized = namespace.strip("/")
        if not normalized:
            raise ValueError("namespace must be non-empty")
        return _NoopCouplerView(self, normalized)

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


class _NoopCouplerView:
    def __init__(self, coupler: _NoopCoupler, namespace: str) -> None:
        self._coupler = coupler
        self._namespace = namespace

    def _qualify(self, key: str) -> str:
        local = str(key).strip("/")
        return f"{self._namespace}/{local}"

    def set(self, key: str, value: Any) -> None:
        self._coupler.set(self._qualify(key), value)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._coupler.get(self._qualify(key), default)

    def update(self, mapping: Mapping[str, Any]) -> None:
        self._coupler.update({self._qualify(key): value for key, value in mapping.items()})

    def declare_outputs(self, *args: Any, **kwargs: Any) -> None:
        return None

    def view(self, namespace: str) -> "_NoopCouplerView":
        normalized = namespace.strip("/")
        if not normalized:
            raise ValueError("namespace must be non-empty")
        return _NoopCouplerView(self._coupler, f"{self._namespace}/{normalized}")


class _NoopScenario:
    def __init__(self) -> None:
        self.coupler: CouplerProtocol = _NoopCoupler()

    @contextmanager
    def trace(self, *args: Any, **kwargs: Any) -> Iterator[None]:
        yield None

    def declare_outputs(self, *args: Any, **kwargs: Any) -> None:
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
