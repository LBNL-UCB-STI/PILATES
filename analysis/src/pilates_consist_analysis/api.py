from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
from consist import RunSet as ConsistRunSet

from .epochs import (
    EpochPanel,
    SimulationEpoch,
    build_epoch_panel,
    converged_epoch as resolve_converged_epoch,
)
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
    runset_from_run_ids,
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
