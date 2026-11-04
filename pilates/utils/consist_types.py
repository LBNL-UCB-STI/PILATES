from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from consist.protocols import (
    ArtifactLike as ConsistArtifactLike,
    ScenarioLike as ConsistScenarioLike,
    TrackerLike as ConsistTrackerLike,
)
from consist.core.identity import IdentityManager


@runtime_checkable
class ArtifactLike(ConsistArtifactLike, Protocol):
    pass


@runtime_checkable
class ScenarioLike(ConsistScenarioLike, Protocol):
    def collect_by_keys(
        self,
        artifacts: Dict[str, Any],
        *keys: str,
        prefix: str = "",
    ) -> Dict[str, Any]: ...


@runtime_checkable
class TrackerLike(ConsistTrackerLike, Protocol):
    identity: IdentityManager

    def log_h5_container(
        self,
        path: Any,
        key: Optional[str] = None,
        *,
        direction: str = "input",
        **metadata: Any,
    ) -> Optional[ArtifactLike]: ...

    def log_h5_table(
        self,
        path: Any,
        key: Optional[str] = None,
        *,
        table_path: str,
        direction: str = "input",
        **metadata: Any,
    ) -> Optional[ArtifactLike]: ...


@runtime_checkable
class ScenarioWithCoupler(ScenarioLike, Protocol):
    coupler: "CouplerProtocol"
    tracker: TrackerLike

    def declare_outputs(self, *names: str, **kwargs: Any) -> None: ...

    def coupler_schema(self, schema: Any) -> Any: ...


@runtime_checkable
class CouplerProtocol(Protocol):
    """
    Protocol for artifact registry used to pass inputs/outputs between steps.
    """

    def set(self, key: str, value: Any) -> None: ...

    def get(self, key: str, default: Optional[Any] = None) -> Any: ...

    def keys(self) -> Iterable[str]: ...

    def update(self, mapping: Mapping[str, Any]) -> None: ...

    def view(self, namespace: str) -> "CouplerProtocol": ...

    def declare_outputs(self, *names: str, **kwargs: Any) -> None: ...

    def collect_by_keys(
        self,
        artifacts: Dict[str, Any],
        *keys: str,
        prefix: str = "",
    ) -> Dict[str, Any]: ...
