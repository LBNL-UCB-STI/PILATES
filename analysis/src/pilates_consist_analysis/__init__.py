"""Consist-enabled post-run analysis scaffolding for PILATES."""

from .keys import CANONICAL_KEY_COLUMNS, AnalysisKey
from .api import AnalysisSession, open_run
from .runset import RunSet, runset_from_query, runset_from_runs
from .epochs import (
    EpochPanel,
    SimulationEpoch,
    build_epoch_panel,
    converged_epoch,
)
from .epoch_views import ARTIFACT_FAMILIES, EpochViews, epoch_views
from .scenario_compare import ScenarioComparison, compare_scenarios

__all__ = [
    "AnalysisKey",
    "AnalysisSession",
    "CANONICAL_KEY_COLUMNS",
    "RunSet",
    "runset_from_query",
    "runset_from_runs",
    "SimulationEpoch",
    "EpochPanel",
    "build_epoch_panel",
    "converged_epoch",
    "ARTIFACT_FAMILIES",
    "EpochViews",
    "epoch_views",
    "ScenarioComparison",
    "compare_scenarios",
    "open_run",
]
