from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
from consist import RunSet as ConsistRunSet

from .activitysim_trips import ActivitySimTripsDataset, build_activitysim_trips_dataset
from .handoff import (
    ArtifactIngestSpec,
    TableTransformSpec,
    export_activitysim_inputs as export_activitysim_inputs_core,
    export_scenario_bundle,
    export_sql_query,
    ingest_artifacts as ingest_artifacts_core,
    list_run_artifacts as list_run_artifacts_core,
    resolve_urbansim_activitysim_boundary_h5s as resolve_urbansim_activitysim_boundary_h5s_core,
)
from .epochs import (
    EpochPanel,
    SimulationEpoch,
    build_epoch_panel,
    converged_epoch as resolve_converged_epoch,
)
from .scenario_compare import (
    ScenarioComparison,
    compare_scenarios as compare_scenarios_core,
    runset_from_run_ids,
)
from .skim_analysis import SkimConvergenceDataset, build_skim_convergence_dataset
from .epoch_views import (
    ARTIFACT_FAMILIES_ENV_VAR,
    EpochViews,
    epoch_views as build_epoch_views,
    resolve_artifact_families,
)
from .runset import (
    RunSet,
    runset_from_query,
    runset_from_runs,
    runset_label,
    runs_to_frame,
)
from .runtime import (
    assert_run_tagging_report,
    create_analysis_tracker,
    db_health_to_frame,
    get_db_health,
    get_db_health_issues,
    get_run_tagging_issues,
    inspect_run_tagging as inspect_run_tagging_report,
    resolve_archive_run_dir,
    resolve_db_path,
    run_tagging_to_frame,
)


class AnalysisSession:
    def __init__(
        self,
        *,
        archive_run_dir: str | Path,
        project_root: str | Path,
        db_path: Optional[str | Path] = None,
        output_root: Optional[str | Path] = None,
        extra_mounts: Optional[Mapping[str, str | Path]] = None,
        access_mode: str = "analysis",
        hashing_strategy: str = "fast",
        tracker: Optional[Any] = None,
        strict_tagging: bool = False,
        fail_on_tagging_issues: bool = False,
        artifact_families: Optional[
            Mapping[str, Mapping[str, Mapping[str, Any]]]
        ] = None,
        artifact_families_json_path: Optional[str | Path] = None,
        artifact_families_env_var: str = ARTIFACT_FAMILIES_ENV_VAR,
    ) -> None:
        self.archive_run_dir = resolve_archive_run_dir(archive_run_dir)
        self.project_root = Path(project_root).expanduser().resolve()
        self.db_path = resolve_db_path(self.archive_run_dir, db_path=db_path)
        self.artifact_families = resolve_artifact_families(
            artifact_families=artifact_families,
            artifact_families_json_path=artifact_families_json_path,
            env_var=artifact_families_env_var,
        )
        self.tracker = tracker or create_analysis_tracker(
            archive_run_dir=self.archive_run_dir,
            project_root=self.project_root,
            db_path=self.db_path,
            output_root=output_root,
            extra_mounts=extra_mounts,
            access_mode=access_mode,
            hashing_strategy=hashing_strategy,
        )
        try:
            self.tagging_report = inspect_run_tagging_report(self.tracker)
        except Exception as exc:
            message = f"run_tagging.validation_failed: {exc}"
            if strict_tagging or fail_on_tagging_issues:
                raise RuntimeError(message) from exc
            self.tagging_report = {
                "total_runs": 0,
                "missing_counts": {
                    "scenario_id": 0,
                    "year": 0,
                    "iteration": 0,
                    "model": 0,
                },
                "linkage_counts": {
                    "beam_parent_checked": 0,
                    "beam_parent_missing": 0,
                    "beam_parent_mismatch": 0,
                    "asim_parent_checked": 0,
                    "asim_parent_missing": 0,
                    "asim_parent_mismatch": 0,
                },
                "warnings": [message],
            }
        self.tagging_warnings = list(self.tagging_report.get("warnings", []) or [])
        self.tagging_issues = get_run_tagging_issues(
            self.tagging_report,
            strict=strict_tagging,
        )
        assert_run_tagging_report(
            self.tagging_report,
            strict=strict_tagging,
            raise_on_issues=fail_on_tagging_issues,
        )

    @classmethod
    def open(
        cls,
        archive_run_dir: str | Path,
        *,
        project_root: Optional[str | Path] = None,
        db_path: Optional[str | Path] = None,
        output_root: Optional[str | Path] = None,
        extra_mounts: Optional[Mapping[str, str | Path]] = None,
        access_mode: str = "analysis",
        hashing_strategy: str = "fast",
        strict_tagging: bool = False,
        fail_on_tagging_issues: bool = False,
        artifact_families: Optional[
            Mapping[str, Mapping[str, Mapping[str, Any]]]
        ] = None,
        artifact_families_json_path: Optional[str | Path] = None,
        artifact_families_env_var: str = ARTIFACT_FAMILIES_ENV_VAR,
    ) -> "AnalysisSession":
        resolved_archive = resolve_archive_run_dir(archive_run_dir)
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]
        return cls(
            archive_run_dir=resolved_archive,
            project_root=project_root,
            db_path=db_path,
            output_root=output_root,
            extra_mounts=extra_mounts,
            access_mode=access_mode,
            hashing_strategy=hashing_strategy,
            strict_tagging=strict_tagging,
            fail_on_tagging_issues=fail_on_tagging_issues,
            artifact_families=artifact_families,
            artifact_families_json_path=artifact_families_json_path,
            artifact_families_env_var=artifact_families_env_var,
        )

    def open_run(self, run_id: str) -> pd.DataFrame:
        run = self.tracker.get_run(run_id) if hasattr(self.tracker, "get_run") else None
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        return runs_to_frame([run])

    def runs(
        self,
        *,
        runset_name: str = "runs",
        tags: Optional[list[str]] = None,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        limit: int = 100,
        name: Optional[str] = None,
    ) -> RunSet:
        return runset_from_query(
            tracker=self.tracker,
            runset_name=runset_name,
            tags=tags,
            year=year,
            iteration=iteration,
            model=model,
            status=status,
            parent_id=parent_id,
            metadata=metadata,
            limit=limit,
            run_name=name,
        )

    def runset_from_ids(
        self, run_ids: Iterable[str], *, name: str = "runset"
    ) -> RunSet:
        return runset_from_run_ids(self.tracker, list(run_ids), name=name)

    def epochs(
        self,
        *,
        scenario_id: Optional[str] = None,
        models: Optional[Iterable[str]] = None,
    ) -> EpochPanel:
        return build_epoch_panel(
            self.tracker,
            scenario_id=scenario_id,
            models=list(models) if models is not None else None,
        )

    def converged_epoch(
        self,
        *,
        year: int,
        scenario_id: Optional[str] = None,
        models: Optional[Iterable[str]] = None,
    ) -> SimulationEpoch:
        return resolve_converged_epoch(
            self.tracker,
            year=year,
            scenario_id=scenario_id,
            models=list(models) if models is not None else None,
        )

    def views(self, epoch: SimulationEpoch) -> EpochViews:
        return build_epoch_views(
            epoch=epoch,
            tracker=self.tracker,
            artifact_families=self.artifact_families,
        )

    def trips(
        self,
        *,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        artifact_family: str = "trips",
        namespace: str = "activitysim",
        grouped_mode: str = "hybrid",
        grouped_missing_files: str = "warn",
        grouped_schema_id: Optional[str] = None,
        latest_per_iteration: bool = True,
        limit: int = 10000,
    ) -> ActivitySimTripsDataset:
        return build_activitysim_trips_dataset(
            self.tracker,
            year=year,
            iteration=iteration,
            artifact_family=artifact_family,
            namespace=namespace,
            grouped_mode=grouped_mode,
            grouped_missing_files=grouped_missing_files,
            grouped_schema_id=grouped_schema_id,
            latest_per_iteration=latest_per_iteration,
            limit=limit,
        )

    def skims(
        self,
        *,
        concept_keys: Optional[list[str]] = None,
        run_ids: Optional[list[str]] = None,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        key_contains: str = "skim",
        limit: int = 10000,
    ) -> SkimConvergenceDataset:
        return build_skim_convergence_dataset(
            self.tracker,
            concept_keys=concept_keys,
            run_ids=run_ids,
            year=year,
            iteration=iteration,
            key_contains=key_contains,
            limit=limit,
        )

    def config(
        self,
        run_id: str,
        *,
        namespace: Optional[str] = None,
        prefix: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        limit: int = 10000,
    ) -> dict[str, Any]:
        if not hasattr(self.tracker, "queries"):
            raise RuntimeError("Tracker does not expose queries service.")
        return self.tracker.queries.get_config_values(
            run_id,
            namespace=namespace,
            prefix=prefix,
            keys=keys,
            limit=limit,
        )

    def diff_configs(
        self,
        run_id_left: str,
        run_id_right: str,
        *,
        namespace: Optional[str] = None,
        prefix: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        limit: int = 10000,
        include_equal: bool = False,
    ) -> pd.DataFrame:
        if not hasattr(self.tracker, "queries"):
            raise RuntimeError("Tracker does not expose queries service.")
        result = self.tracker.queries.diff_runs(
            run_id_left,
            run_id_right,
            namespace=namespace,
            prefix=prefix,
            keys=keys,
            limit=limit,
            include_equal=include_equal,
        )
        namespaces = result.get("namespace", {}) or {}
        rows = []
        for key, payload in (result.get("changes", {}) or {}).items():
            rows.append(
                {
                    "key": key,
                    "left": payload.get("left"),
                    "right": payload.get("right"),
                    "status": payload.get("status"),
                    "run_id_left": run_id_left,
                    "run_id_right": run_id_right,
                    "namespace_left": namespaces.get("left"),
                    "namespace_right": namespaces.get("right"),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "key",
                    "left",
                    "right",
                    "status",
                    "run_id_left",
                    "run_id_right",
                    "namespace_left",
                    "namespace_right",
                ]
            )
        return pd.DataFrame(rows).sort_values(["status", "key"]).reset_index(drop=True)

    def inspect_db(self) -> pd.DataFrame:
        health = get_db_health(self.tracker, archive_run_dir=self.archive_run_dir)
        return db_health_to_frame(health)

    def assert_db_healthy(self, strict: bool = False) -> pd.DataFrame:
        health = get_db_health(self.tracker, archive_run_dir=self.archive_run_dir)
        issues = get_db_health_issues(health, strict=strict)
        if issues:
            mode = "strict" if strict else "standard"
            raise RuntimeError(f"DB health check failed ({mode}): {', '.join(issues)}")
        return db_health_to_frame(health)

    def run_tagging_report(self) -> dict[str, Any]:
        return dict(self.tagging_report)

    def inspect_run_tagging(self, strict: bool = False) -> pd.DataFrame:
        return run_tagging_to_frame(self.tagging_report, strict=strict)

    def assert_run_tagging(self, strict: bool = False) -> pd.DataFrame:
        assert_run_tagging_report(
            self.tagging_report, strict=strict, raise_on_issues=True
        )
        return run_tagging_to_frame(self.tagging_report, strict=strict)

    def assert_run_tagging_consistent(
        self,
        *,
        strict: bool = True,
        raise_on_issues: bool = True,
    ) -> dict[str, Any]:
        assert_run_tagging_report(
            self.tagging_report,
            strict=strict,
            raise_on_issues=raise_on_issues,
        )
        return self.run_tagging_report()

    def ingest_artifacts(
        self,
        artifact_specs: Sequence[ArtifactIngestSpec],
        *,
        run_id: Optional[str] = None,
        model: str = "analysis_ingest",
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        parent_run_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        run_config: Optional[Mapping[str, Any]] = None,
        ingest_data: bool = True,
        profile_schema: bool = True,
    ) -> dict[str, Any]:
        return ingest_artifacts_core(
            self.tracker,
            artifact_specs,
            run_id=run_id,
            model=model,
            scenario_id=scenario_id,
            seed=seed,
            year=year,
            iteration=iteration,
            parent_run_id=parent_run_id,
            tags=tags,
            run_config=run_config,
            ingest_data=ingest_data,
            profile_schema=profile_schema,
        )

    def list_run_artifacts(
        self,
        *,
        run_id: str,
        direction: str = "output",
        key_contains: Optional[str] = None,
        artifact_family_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        return list_run_artifacts_core(
            self.tracker,
            run_id=run_id,
            direction=direction,
            key_contains=key_contains,
            artifact_family_prefix=artifact_family_prefix,
        )

    def resolve_urbansim_activitysim_boundary_h5s(
        self,
        *,
        forecast_year: int,
        next_input_year: Optional[int] = None,
    ) -> pd.DataFrame:
        return resolve_urbansim_activitysim_boundary_h5s_core(
            self.archive_run_dir,
            forecast_year=forecast_year,
            next_input_year=next_input_year,
        )

    def export_scenario_db(
        self,
        *,
        out_path: str | Path,
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        status: Optional[str] = "completed",
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        limit: int = 10000,
        use_converged: bool = True,
        converged_group_by: Optional[Iterable[str]] = None,
        latest_group_by: Optional[Iterable[str]] = None,
        include_data: bool = True,
        include_snapshots: bool = False,
        include_children: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return export_scenario_bundle(
            self.tracker,
            archive_run_dir=self.archive_run_dir,
            out_path=out_path,
            scenario_id=scenario_id,
            seed=seed,
            model=model,
            status=status,
            year=year,
            iteration=iteration,
            tags=tags,
            metadata=metadata,
            limit=limit,
            use_converged=use_converged,
            converged_group_by=list(converged_group_by)
            if converged_group_by is not None
            else None,
            latest_group_by=list(latest_group_by)
            if latest_group_by is not None
            else None,
            include_data=include_data,
            include_snapshots=include_snapshots,
            include_children=include_children,
            dry_run=dry_run,
        )

    def export_sql(
        self,
        *,
        sql: str,
        output_path: str | Path,
        output_format: str = "csv",
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        return export_sql_query(
            self.tracker,
            sql=sql,
            output_path=output_path,
            output_format=output_format,
            limit=limit,
        )

    def export_activitysim_inputs(
        self,
        *,
        output_dir: str | Path,
        scenario_id: Optional[str] = None,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        use_converged: bool = True,
        trips: Optional[TableTransformSpec] = None,
        persons: Optional[TableTransformSpec] = None,
        include_trips: bool = True,
        include_persons: bool = True,
        output_format: str = "csv",
    ) -> dict[str, Any]:
        return export_activitysim_inputs_core(
            self.tracker,
            output_dir=output_dir,
            scenario_id=scenario_id,
            year=year,
            iteration=iteration,
            use_converged=use_converged,
            trips=trips,
            persons=persons,
            include_trips=include_trips,
            include_persons=include_persons,
            output_format=output_format,
        )

    def compare_scenarios(
        self,
        left: RunSet | Iterable[str],
        right: RunSet | Iterable[str],
        *,
        left_name: str = "left",
        right_name: str = "right",
        datasets: Optional[list[str]] = None,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        config_namespace: Optional[str] = None,
        config_prefix: Optional[str] = None,
        config_include_equal: bool = False,
        align_on: str = "year",
        latest_group_by: Optional[Iterable[str]] = None,
        use_converged: bool = False,
        converged_group_by: Optional[Iterable[str]] = None,
    ) -> ScenarioComparison:
        if isinstance(left, str):
            left = [left]
        if isinstance(right, str):
            right = [right]
        left_runset = (
            left
            if isinstance(left, (RunSet, ConsistRunSet))
            else self.runset_from_ids(left, name=left_name)
        )
        right_runset = (
            right
            if isinstance(right, (RunSet, ConsistRunSet))
            else self.runset_from_ids(right, name=right_name)
        )
        if not isinstance(left_runset, RunSet):
            left_runset = runset_from_runs(
                left_runset,
                name=runset_label(left_runset, default=left_name),
                tracker=self.tracker,
            )
        if not isinstance(right_runset, RunSet):
            right_runset = runset_from_runs(
                right_runset,
                name=runset_label(right_runset, default=right_name),
                tracker=self.tracker,
            )
        left_runset.label = runset_label(left_runset, default=left_name)
        right_runset.label = runset_label(right_runset, default=right_name)
        return compare_scenarios_core(
            self.tracker,
            left_runset,
            right_runset,
            datasets=datasets,
            year=year,
            iteration=iteration,
            config_namespace=config_namespace,
            config_prefix=config_prefix,
            config_include_equal=config_include_equal,
            align_on=align_on,
            latest_group_by=list(latest_group_by)
            if latest_group_by is not None
            else None,
            use_converged=bool(use_converged),
            converged_group_by=list(converged_group_by)
            if converged_group_by is not None
            else None,
        )


def open_run(
    archive_run_dir: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    db_path: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    extra_mounts: Optional[Mapping[str, str | Path]] = None,
    access_mode: str = "analysis",
    hashing_strategy: str = "fast",
    strict_tagging: bool = False,
    fail_on_tagging_issues: bool = False,
    artifact_families: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
    artifact_families_json_path: Optional[str | Path] = None,
    artifact_families_env_var: str = ARTIFACT_FAMILIES_ENV_VAR,
) -> AnalysisSession:
    return AnalysisSession.open(
        archive_run_dir=archive_run_dir,
        project_root=project_root,
        db_path=db_path,
        output_root=output_root,
        extra_mounts=extra_mounts,
        access_mode=access_mode,
        hashing_strategy=hashing_strategy,
        strict_tagging=strict_tagging,
        fail_on_tagging_issues=fail_on_tagging_issues,
        artifact_families=artifact_families,
        artifact_families_json_path=artifact_families_json_path,
        artifact_families_env_var=artifact_families_env_var,
    )
