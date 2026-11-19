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


def is_start_year(state: object, settings: object = None) -> bool:
    """Return whether the current runtime state represents the start year."""
    state_fn = getattr(state, "is_start_year", None)
    if callable(state_fn):
        return bool(state_fn())

    state_start_year = getattr(state, "start_year", None)
    if state_start_year is None and settings is not None:
        run_settings = getattr(settings, "run", None)
        if run_settings is not None:
            state_start_year = getattr(run_settings, "start_year", None)
        if state_start_year is None and isinstance(settings, dict):
            state_start_year = settings.get("start_year")
        elif state_start_year is None:
            state_start_year = getattr(settings, "start_year", None)

    resolved_year = current_year(state)
    if resolved_year is not None and state_start_year is not None:
        return resolved_year == state_start_year

    forecast_year = getattr(state, "forecast_year", None)
    if forecast_year is not None and state_start_year is not None:
        return forecast_year == state_start_year

    return False


def uses_input_datastore(state: object) -> bool:
    """
    Return whether the current state should read the interval input datastore.
    """
    state_fn = getattr(state, "is_start_year", None)
    if callable(state_fn):
        return bool(state_fn())

    resolved_year = current_year(state)
    forecast_year = getattr(state, "forecast_year", None)
    start_year = getattr(state, "start_year", None)
    return (
        resolved_year is not None
        and forecast_year is not None
        and start_year is not None
        and resolved_year == forecast_year == start_year
    )
