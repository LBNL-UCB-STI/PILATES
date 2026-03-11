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
from .epoch_views import (
    ARTIFACT_FAMILIES,
    ARTIFACT_FAMILIES_ENV_VAR,
    EpochViews,
    epoch_views,
    load_artifact_families_from_json,
    resolve_artifact_families,
)
from .scenario_compare import ScenarioComparison, compare_scenarios
from .handoff import (
    ArtifactIngestSpec,
    TableTransformSpec,
    export_activitysim_inputs,
    export_scenario_bundle,
    export_sql_query,
    ingest_artifacts,
    list_run_artifacts,
    parse_artifact_ref_arg,
)
from .runtime import (
    assert_run_tagging_consistent,
    get_run_tagging_issues,
    inspect_run_tagging,
    run_tagging_to_frame,
)

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
    "ARTIFACT_FAMILIES_ENV_VAR",
    "EpochViews",
    "epoch_views",
    "load_artifact_families_from_json",
    "resolve_artifact_families",
    "ScenarioComparison",
    "compare_scenarios",
    "ArtifactIngestSpec",
    "TableTransformSpec",
    "ingest_artifacts",
    "export_scenario_bundle",
    "export_sql_query",
    "export_activitysim_inputs",
    "list_run_artifacts",
    "parse_artifact_ref_arg",
    "open_run",
    "inspect_run_tagging",
    "get_run_tagging_issues",
    "assert_run_tagging_consistent",
    "run_tagging_to_frame",
]
