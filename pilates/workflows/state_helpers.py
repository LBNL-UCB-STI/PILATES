from __future__ import annotations

from typing import Any, Optional


def resolve_forecast_year(state: Any) -> Optional[int]:
    """Return ``state.forecast_year``, falling back to ``state.year``.

    The fallback supports restart paths where ``forecast_year`` may not yet be
    populated. Normal forward execution should have ``forecast_year`` set.
    """
    forecast_year = getattr(state, "forecast_year", None)
    if forecast_year is not None:
        return forecast_year
    return getattr(state, "year", None)
