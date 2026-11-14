from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
)
from typing_extensions import Protocol, runtime_checkable

from consist.protocols import (
    ArtifactRecordLike,
    ScenarioLike as ConsistScenarioLike,
    TrackerLike as ConsistTrackerLike,
)

if TYPE_CHECKING:
    from consist.core.identity import IdentityManager


@runtime_checkable
class ArtifactLike(ArtifactRecordLike, Protocol):
    @property
    def run_id(self) -> Optional[str]: ...


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

    def close(self) -> None: ...

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

    def __getitem__(self, key: str) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...

    def __delitem__(self, key: str) -> None: ...

    def __iter__(self) -> Iterator[str]: ...

    def __len__(self) -> int: ...

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

    def path(self, key: str) -> Path: ...

    def to_dict(self) -> Dict[str, Any]: ...
