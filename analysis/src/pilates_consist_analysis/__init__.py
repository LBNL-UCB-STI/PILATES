"""Consist-enabled post-run analysis scaffolding for PILATES."""

from .keys import CANONICAL_KEY_COLUMNS, AnalysisKey
from .api import AnalysisSession, open_run
from .archive import Archive, ArchiveScenario, open_archive
from .run_index import RunIndex, build_run_index
from .runset import RunSet, runset_from_query, runset_from_runs
from .epochs import (
    EpochPanel,
    SimulationEpoch,
    build_epoch_panel,
    converged_epoch,
)
from .epoch_views import (
    ARTIFACT_FAMILIES,
    ARTIFACT_FAMILIES_ENV_VAR,
    EpochViews,
    epoch_views,
    load_artifact_families_from_json,
    resolve_artifact_families,
)
from .epoch_api import Epoch, EpochTables
from .faceted import delta, delta_change, difference, rank
from .runtime import (
    assert_run_tagging_consistent,
    get_run_tagging_issues,
    inspect_run_tagging,
    run_tagging_to_frame,
)

__all__ = [
    "AnalysisKey",
    "AnalysisSession",
    "Archive",
    "ArchiveScenario",
    "CANONICAL_KEY_COLUMNS",
    "RunIndex",
    "build_run_index",
    "RunSet",
    "runset_from_query",
    "runset_from_runs",
    "SimulationEpoch",
    "EpochPanel",
    "Epoch",
    "EpochTables",
    "delta",
    "delta_change",
    "difference",
    "rank",
    "build_epoch_panel",
    "converged_epoch",
    "ARTIFACT_FAMILIES",
    "ARTIFACT_FAMILIES_ENV_VAR",
    "EpochViews",
    "epoch_views",
    "load_artifact_families_from_json",
    "resolve_artifact_families",
    "open_archive",
    "open_run",
    "inspect_run_tagging",
    "get_run_tagging_issues",
    "assert_run_tagging_consistent",
    "run_tagging_to_frame",
]
