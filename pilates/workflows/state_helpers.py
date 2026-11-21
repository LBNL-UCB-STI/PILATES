"""Pure workflow state and authority predicates."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from pilates.utils.io import get_traffic_assignment_model


@runtime_checkable
class _HasForecastYear(Protocol):
    forecast_year: int | None


@runtime_checkable
class _HasYear(Protocol):
    year: int | None


@runtime_checkable
class _HasCurrentYear(Protocol):
    current_year: int | None


@runtime_checkable
class _HasAtlasIntervalStartYear(Protocol):
    atlas_interval_start_year: int | None


@runtime_checkable
class _HasIteration(Protocol):
    iteration: int | None


@runtime_checkable
class _HasCurrentInnerIter(Protocol):
    current_inner_iter: int | None


@runtime_checkable
class _HasRunStartYear(Protocol):
    start_year: int | None


@runtime_checkable
class _HasRun(Protocol):
    run: Any


@runtime_checkable
class _HasBeam(Protocol):
    beam: Any


def resolve_forecast_year(state: Any) -> Optional[int]:
    """Return ``state.forecast_year``, falling back to ``state.year``.

    The fallback supports restart paths where ``forecast_year`` may not yet be
    populated. Normal forward execution should have ``forecast_year`` set.
    """
    if isinstance(state, _HasForecastYear) and state.forecast_year is not None:
        return state.forecast_year
    if isinstance(state, _HasYear):
        return state.year
    return None


def _frontier_year(state: Any) -> int | None:
    if isinstance(state, _HasAtlasIntervalStartYear):
        return state.atlas_interval_start_year
    if isinstance(state, _HasCurrentYear) and state.current_year is not None:
        return state.current_year
    if isinstance(state, _HasYear):
        return state.year
    return None


def _iteration_value(state: Any) -> int:
    if isinstance(state, _HasIteration) and state.iteration is not None:
        return int(state.iteration)
    if isinstance(state, _HasCurrentInnerIter) and state.current_inner_iter is not None:
        return int(state.current_inner_iter)
    return 0


def _start_year_value(settings: Any) -> int | None:
    if isinstance(settings, _HasRun) and isinstance(settings.run, _HasRunStartYear):
        return settings.run.start_year
    return None


def requires_prior_beam_skim_handoff(*, settings: Any, state: Any) -> bool:
    """Whether this invocation must consume a prior BEAM-produced OMX skim.

    The start-year, first-iteration invocation is bootstrap: it may use the
    configured initial skim. Every later BEAM frontier instead requires the
    explicit producer handoff, including restart hydration.
    """
    if get_traffic_assignment_model(settings) != "beam":
        return False

    frontier_year = _frontier_year(state)
    start_year = _start_year_value(settings)
    iteration = _iteration_value(state)
    if (
        start_year is not None
        and frontier_year is not None
        and frontier_year > start_year
    ):
        return True
    return iteration > 0


def requires_prior_beam_linkstats_handoff(*, settings: Any, state: Any) -> bool:
    """Whether this invocation must consume a prior BEAM-produced linkstats handoff.

    The start-year, first-iteration invocation is bootstrap: it may use the
    configured initial linkstats file. Every later BEAM iteration or year
    instead requires the explicit producer handoff from a previous BEAM run.
    """
    model = get_traffic_assignment_model(settings)
    if model is not None and model != "beam":
        return False
    if model is None and (not isinstance(settings, _HasBeam) or settings.beam is None):
        return False

    frontier_year = _frontier_year(state)
    start_year = _start_year_value(settings)
    iteration = _iteration_value(state)
    if (
        start_year is not None
        and frontier_year is not None
        and frontier_year > start_year
    ):
        return True
    return iteration > 0
