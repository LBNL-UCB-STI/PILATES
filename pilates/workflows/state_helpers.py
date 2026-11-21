"""Pure workflow state and authority predicates."""

from __future__ import annotations

from typing import Any, Optional

from pilates.utils.io import get_traffic_assignment_model


def resolve_forecast_year(state: Any) -> Optional[int]:
    """Return ``state.forecast_year``, falling back to ``state.year``.

    The fallback supports restart paths where ``forecast_year`` may not yet be
    populated. Normal forward execution should have ``forecast_year`` set.
    """
    forecast_year = getattr(state, "forecast_year", None)
    if forecast_year is not None:
        return forecast_year
    return getattr(state, "year", None)


def _frontier_year(state: Any) -> int | None:
    atlas_start = getattr(state, "atlas_interval_start_year", None)
    if atlas_start is not None:
        return atlas_start
    current_yr = getattr(state, "current_year", None)
    if current_yr is not None:
        return current_yr
    return getattr(state, "year", None)


def _iteration_value(state: Any) -> int:
    iteration = getattr(state, "iteration", None)
    if iteration is not None:
        return int(iteration)
    inner_iter = getattr(state, "current_inner_iter", None)
    if inner_iter is not None:
        return int(inner_iter)
    return 0


def _start_year_value(settings: Any) -> int | None:
    run_cfg = getattr(settings, "run", None)
    return getattr(run_cfg, "start_year", None) if run_cfg is not None else None


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
    if model is None and getattr(settings, "beam", None) is None:
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
