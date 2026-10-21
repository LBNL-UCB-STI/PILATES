"""Native-step input-authority predicates.

These predicates distinguish bootstrap ingress from a workflow frontier that
must consume a previously published producer artifact.  They deliberately do
not select paths; native resolvers remain the sole selection authority.
"""

from __future__ import annotations

from typing import Any

from pilates.utils.io import get_traffic_assignment_model


def requires_prior_beam_skim_handoff(*, settings: Any, state: Any) -> bool:
    """Whether this invocation must consume a prior BEAM-produced OMX skim.

    The start-year, first-iteration invocation is bootstrap: it may use the
    configured initial skim.  Every later BEAM frontier instead requires the
    explicit producer handoff, including restart hydration.
    """

    if get_traffic_assignment_model(settings) != "beam":
        return False

    # ``AtlasSubState.current_year`` advances through ATLAS's internal
    # sub-years before the workflow has executed BEAM.  Its
    # ``atlas_interval_start_year`` preserves the parent workflow frontier,
    # which is the only year boundary that can establish a BEAM skim handoff.
    frontier_year = getattr(
        state,
        "atlas_interval_start_year",
        getattr(state, "current_year", getattr(state, "year", None)),
    )
    start_year = getattr(getattr(settings, "run", None), "start_year", None)
    iteration = getattr(state, "iteration", getattr(state, "current_inner_iter", 0))
    if (
        start_year is not None
        and frontier_year is not None
        and frontier_year > start_year
    ):
        return True
    return int(iteration or 0) > 0


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

    frontier_year = getattr(
        state,
        "atlas_interval_start_year",
        getattr(state, "current_year", getattr(state, "year", None)),
    )
    start_year = getattr(getattr(settings, "run", None), "start_year", None)
    iteration = getattr(state, "iteration", getattr(state, "current_inner_iter", 0))
    if (
        start_year is not None
        and frontier_year is not None
        and frontier_year > start_year
    ):
        return True
    return int(iteration or 0) > 0
