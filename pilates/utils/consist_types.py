from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

try:
    from consist.protocols import (  # type: ignore[assignment]
        ArtifactLike,
        RunResultLike,
        ScenarioLike,
        TrackerLike,
    )
except Exception:  # pragma: no cover - optional Consist dependency

    @runtime_checkable
    class ArtifactLike(Protocol):
        @property
        def path(self) -> Any: ...

        @property
        def uri(self) -> str: ...

    @runtime_checkable
    class RunResultLike(Protocol):
        outputs: Dict[str, Any]
        cache_hit: bool

    @runtime_checkable
    class ScenarioLike(Protocol):
        def run(
            self,
            fn=None,
            name: Optional[str] = None,
            *,
            output_paths: Optional[Mapping[str, Any]] = None,
            runtime_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> RunResultLike: ...

        def trace(self, *args: Any, **kwargs: Any): ...

    @runtime_checkable
    class TrackerLike(Protocol):
        def log_output(
            self, path: Any, key: Optional[str] = None, **metadata: Any
        ) -> Optional[ArtifactLike]: ...

        def log_input(
            self, path: Any, key: Optional[str] = None, **metadata: Any
        ) -> Optional[ArtifactLike]: ...

        def log_artifacts(
            self, outputs: Mapping[str, Any], **metadata: Any
        ) -> Mapping[str, ArtifactLike]: ...

        def scenario(self, name: str, **kwargs: Any): ...


@runtime_checkable
class ScenarioWithCoupler(ScenarioLike, Protocol):
    coupler: "CouplerProtocol"

    def declare_outputs(self, *names: str, **kwargs: Any) -> None: ...

    def coupler_schema(self, schema: Any) -> Any: ...


@runtime_checkable
class CouplerProtocol(Protocol):
    """
    Protocol for artifact registry used to pass inputs/outputs between steps.
    """

    def set(self, key: str, value: Any) -> None: ...

    def get(self, key: str, default: Optional[Any] = None) -> Any: ...

    def pop(self, key: str, default: Optional[Any] = None) -> Any: ...

    def update(self, mapping: Mapping[str, Any]) -> None: ...

    def declare_outputs(self, *names: str, **kwargs: Any) -> None: ...

    def adopt_cached_output(self, *args: Any, **kwargs: Any) -> None: ...

    def collect_by_keys(
        self,
        artifacts: Dict[str, Any],
        *keys: str,
        prefix: str = "",
    ) -> Dict[str, Any]: ...
