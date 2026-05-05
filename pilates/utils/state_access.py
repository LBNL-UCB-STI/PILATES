from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class YearStateLike(Protocol):
    current_year: Optional[int]
    year: Optional[int]
    forecast_year: Optional[int]

    def is_start_year(self) -> bool: ...


@runtime_checkable
class IterationStateLike(Protocol):
    current_inner_iter: Optional[int]
    iteration: Optional[int]


def current_year(state: object) -> Optional[int]:
    """Return the runtime year using the canonical/current alias first."""
    return getattr(state, "current_year", getattr(state, "year", None))


def iteration_index(state: object, default: Optional[int] = None) -> Optional[int]:
    """Return the current iteration using the canonical/current alias first."""
    value = getattr(state, "current_inner_iter", getattr(state, "iteration", default))
    return default if value is None else value


def uses_input_datastore(state: object) -> bool:
    """
    Return whether the current state should read the interval input datastore.
    """
    is_start_year = getattr(state, "is_start_year", None)
    if callable(is_start_year):
        return bool(is_start_year())

    resolved_year = current_year(state)
    forecast_year = getattr(state, "forecast_year", None)
    start_year = getattr(state, "start_year", None)
    return (
        resolved_year is not None
        and forecast_year is not None
        and start_year is not None
        and resolved_year == forecast_year == start_year
    )
