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
    return state.current_year > settings.run.start_year or state.iteration > 0
