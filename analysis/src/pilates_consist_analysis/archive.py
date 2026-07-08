from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .api import AnalysisSession
from .epoch_api import Epoch
from .epoch_views import ARTIFACT_FAMILIES
from .epochs import SimulationEpoch
from .run_index import RunIndex, build_run_index
from .runtime import (
    get_db_health,
    get_db_health_issues,
    resolve_archive_run_dir,
    resolve_db_path,
)


SYSTEM_FACET_COLUMNS = {
    "scenario_id": ("scenario_id", "facet_scenario_id", "consist_scenario_id"),
    "year": ("year", "facet_year", "consist_year"),
    "iteration": ("iteration", "facet_iteration", "consist_iteration"),
    "model": ("model", "facet_model", "consist_model"),
    "seed": ("seed", "facet_seed", "consist_seed"),
}


def _open_grouped_ibis_table(**kwargs: Any) -> Any:
    import consist

    context = consist.ibis_grouped_view(**kwargs)
    table = context.__enter__()
    return table, context


def _copy_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            return
        shutil.copytree(source, destination)
        return
    if destination.exists():
        return
    shutil.copy2(source, destination)


def _stable_view_name(
    name: str, facets: Sequence[str], where: Mapping[str, Any]
) -> str:
    payload = json.dumps(
        {
            "name": name,
            "facets": list(facets),
            "where": {key: str(value) for key, value in sorted(where.items())},
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"v_pilates_analysis_{digest}"


def _table_columns(table: Any) -> list[str]:
    try:
        return list(table.columns)
    except Exception:
        return []


def _alias_facets(table: Any, facets: Sequence[str]) -> Any:
    columns = _table_columns(table)
    for facet in facets:
        if facet in columns:
            continue
        candidates = SYSTEM_FACET_COLUMNS.get(
            facet, (facet, f"facet_{facet}", f"consist_{facet}")
        )
        source = next(
            (candidate for candidate in candidates if candidate in columns),
            None,
        )
        if source is None:
            continue
        table = table.mutate(**{facet: table[source]})
        columns = _table_columns(table)
    return table


def _apply_where(table: Any, where: Mapping[str, Any]) -> Any:
    for column, value in where.items():
        if isinstance(value, range):
            table = table.filter(table[column].isin(list(value)))
        elif isinstance(value, (list, tuple, set, frozenset)):
            if len(value) == 2 and isinstance(value, tuple):
                lower, upper = value
                table = table.filter(table[column] >= lower)
                table = table.filter(table[column] <= upper)
            else:
                table = table.filter(table[column].isin(list(value)))
        else:
            table = table.filter(table[column] == value)
    return table


@dataclass(frozen=True)
class ArchiveScenario:
    archive: "Archive"
    scenario_id: str

    def summary(self) -> pd.DataFrame:
        runs = self.runs()
        row = {
            "scenario_id": self.scenario_id,
            "run_count": int(len(runs)),
            "year_count": int(len(self.years())),
            "model_count": int(len(self.archive.models(scenario_id=self.scenario_id))),
        }
        return pd.DataFrame([row])

    def runs(
        self,
        *,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        completed_only: bool = False,
    ) -> pd.DataFrame:
        return self.archive.runs(
            scenario_id=self.scenario_id,
            year=year,
            iteration=iteration,
            model=model,
            status=status,
            completed_only=completed_only,
        )

    def years(self) -> list[int]:
        return self.archive.years(scenario_id=self.scenario_id)

    def models(self) -> list[str]:
        return self.archive.models(scenario_id=self.scenario_id)

    def epochs(
        self,
        *,
        converged: bool = False,
        models: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        return self.archive.epochs(
            scenario_id=self.scenario_id,
            converged=converged,
            models=models,
        )

    def epoch(
        self,
        *,
        year: int,
        converged: bool = False,
        iteration: Optional[int] = None,
        models: Optional[Iterable[str]] = None,
    ) -> Epoch:
        return self.archive.epoch(
            year=year,
            scenario_id=self.scenario_id,
            converged=converged,
            iteration=iteration,
            models=models,
        )


class Archive:
    def __init__(
        self,
        session: AnalysisSession,
        *,
        persistent_archive_run_dir: Optional[str | Path] = None,
        local_cache_run_dir: Optional[str | Path] = None,
    ) -> None:
        self.session = session
        self._run_index: Optional[RunIndex] = None
        self.persistent_archive_run_dir = (
            Path(persistent_archive_run_dir).expanduser().resolve()
            if persistent_archive_run_dir is not None
            else Path(session.archive_run_dir)
        )
        self.local_cache_run_dir = (
            Path(local_cache_run_dir).expanduser().resolve()
            if local_cache_run_dir is not None
            else None
        )
        self._localization_manifest: dict[str, Any] = {
            "persistent_archive_run_dir": str(self.persistent_archive_run_dir),
            "local_archive_run_dir": str(self.archive_run_dir),
            "db_path": str(self.db_path),
            "artifacts": [],
        }
        self._ibis_contexts: list[Any] = []

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
        artifact_families_env_var: Optional[str] = None,
        local_cache: Optional[str | Path] = None,
    ) -> "Archive":
        persistent_archive = resolve_archive_run_dir(archive_run_dir)
        local_cache_run_dir = None
        if local_cache is not None:
            local_cache_run_dir = (
                Path(local_cache).expanduser().resolve() / persistent_archive.name
            )
            local_db_path = local_cache_run_dir / ".consist" / "consist.duckdb"
            source_db_path = resolve_db_path(persistent_archive, db_path=db_path)
            _copy_path(source_db_path, local_db_path)
            archive_run_dir = local_cache_run_dir
            db_path = local_db_path

        session_kwargs: dict[str, Any] = {
            "archive_run_dir": archive_run_dir,
            "project_root": project_root,
            "db_path": db_path,
            "output_root": output_root,
            "extra_mounts": extra_mounts,
            "access_mode": access_mode,
            "hashing_strategy": hashing_strategy,
            "strict_tagging": strict_tagging,
            "fail_on_tagging_issues": fail_on_tagging_issues,
            "artifact_families": artifact_families,
            "artifact_families_json_path": artifact_families_json_path,
        }
        if artifact_families_env_var is not None:
            session_kwargs["artifact_families_env_var"] = artifact_families_env_var
        return cls(
            AnalysisSession.open(**session_kwargs),
            persistent_archive_run_dir=persistent_archive,
            local_cache_run_dir=local_cache_run_dir,
        )

    @property
    def tracker(self) -> Any:
        return self.session.tracker

    @property
    def archive_run_dir(self) -> Path:
        return Path(self.session.archive_run_dir)

    @property
    def db_path(self) -> Path:
        return Path(self.session.db_path)

    @property
    def run_index(self) -> RunIndex:
        if self._run_index is None:
            self._run_index = build_run_index(
                self.tracker,
                archive_run_dir=self.archive_run_dir,
            )
        return self._run_index

    def summary(self) -> pd.DataFrame:
        health = get_db_health(self.tracker, archive_run_dir=self.archive_run_dir)
        row = {
            "archive_run_dir": str(self.archive_run_dir),
            "db_path": str(self.db_path),
            "run_count": int(len(self.run_index.frame)),
            "scenario_count": int(len(self.scenarios())),
            "year_count": int(len(self.years())),
            "model_count": int(len(self.models())),
            "db_healthy": bool(health.get("healthy", False)),
            "db_issue_count": int(len(get_db_health_issues(health, strict=False))),
            "tagging_issue_count": int(len(self.session.tagging_issues)),
            "tagging_warning_count": int(len(self.session.tagging_warnings)),
        }
        return pd.DataFrame([row])

    def issues(self, *, strict_db: bool = False) -> dict[str, Any]:
        health = get_db_health(self.tracker, archive_run_dir=self.archive_run_dir)
        return {
            "archive_run_dir": str(self.archive_run_dir),
            "db_path": str(self.db_path),
            "db_issues": get_db_health_issues(health, strict=strict_db),
            "tagging_issues": list(self.session.tagging_issues),
            "tagging_warnings": list(self.session.tagging_warnings),
        }

    def runs(
        self,
        *,
        scenario_id: Optional[str] = None,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        completed_only: bool = False,
    ) -> pd.DataFrame:
        return self.run_index.filter(
            scenario_id=scenario_id,
            year=year,
            iteration=iteration,
            model=model,
            status=status,
            completed_only=completed_only,
        )

    def scenarios(self) -> list[str]:
        return self.run_index.scenarios()

    def years(self, *, scenario_id: Optional[str] = None) -> list[int]:
        return self.run_index.years(scenario_id=scenario_id)

    def models(self, *, scenario_id: Optional[str] = None) -> list[str]:
        return self.run_index.models(scenario_id=scenario_id)

    def scenario(self, scenario_id: str) -> ArchiveScenario:
        normalized = str(scenario_id).strip()
        if not normalized:
            raise ValueError("scenario_id must be a non-empty string.")
        return ArchiveScenario(archive=self, scenario_id=normalized)

    def epochs(
        self,
        *,
        scenario_id: Optional[str] = None,
        converged: bool = False,
        models: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        panel = self.session.epochs(
            scenario_id=scenario_id,
            models=models,
        )
        if converged:
            panel = panel.converged_epochs()
        return panel.to_frame()

    def views(self, epoch: Epoch | SimulationEpoch) -> Any:
        simulation_epoch = epoch.raw if isinstance(epoch, Epoch) else epoch
        return self.session.views(simulation_epoch)

    def table(
        self,
        name: str,
        *,
        facets: Optional[Sequence[str]] = None,
        where: Optional[Mapping[str, Any]] = None,
        mode: str = "hybrid",
        missing_files: str = "warn",
        schema_compatible: bool = False,
    ) -> Any:
        model, logical_name = self._parse_table_name(name)
        family_spec = self._artifact_family_spec(model, logical_name)
        artifact_family = family_spec["artifact_family"]
        requested_facets = list(dict.fromkeys(facets or ()))
        filters = dict(where or {})
        params = [f"artifact_family={artifact_family}"]
        seed = self._seed_artifact(model=model, artifact_family=artifact_family)
        self._localize_artifacts([seed])
        opened = _open_grouped_ibis_table(
            tracker=self.tracker,
            view_name=_stable_view_name(name, requested_facets, filters),
            artifact_id=seed.id,
            namespace=model,
            params=params,
            attach_facets=requested_facets,
            include_system_columns=True,
            mode=mode,
            if_exists="replace",
            missing_files=missing_files,
            schema_compatible=schema_compatible,
        )
        if isinstance(opened, tuple) and len(opened) == 2:
            table, context = opened
            self._ibis_contexts.append(context)
        else:
            table = opened
        table = _alias_facets(table, requested_facets)
        return _apply_where(table, filters)

    def measure(
        self,
        table: Any,
        *,
        by: Sequence[str],
        measures: Mapping[str, Any],
    ) -> Any:
        aggregations = {name: builder(table) for name, builder in measures.items()}
        return table.group_by(list(by)).agg(**aggregations)

    def localization_manifest(self) -> dict[str, Any]:
        return {
            **self._localization_manifest,
            "artifacts": list(self._localization_manifest["artifacts"]),
        }

    def close(self) -> None:
        while self._ibis_contexts:
            context = self._ibis_contexts.pop()
            context.__exit__(None, None, None)

    def __enter__(self) -> "Archive":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.close()

    def epoch(
        self,
        *,
        year: int,
        scenario_id: Optional[str] = None,
        converged: bool = False,
        iteration: Optional[int] = None,
        models: Optional[Iterable[str]] = None,
    ) -> Epoch:
        simulation_epoch: SimulationEpoch
        if converged:
            simulation_epoch = self.session.converged_epoch(
                year=year,
                scenario_id=scenario_id,
                models=models,
            )
            return Epoch(archive=self, simulation_epoch=simulation_epoch)

        panel = self.session.epochs(
            scenario_id=scenario_id,
            models=models,
        )
        candidates = [epoch for epoch in panel if int(epoch.year) == int(year)]
        if iteration is not None:
            candidates = [
                epoch
                for epoch in candidates
                if int(epoch.outer_iteration) == int(iteration)
            ]
        if not candidates:
            raise ValueError(
                "No epoch found for "
                f"scenario_id={scenario_id!r}, year={year}, iteration={iteration}."
            )
        if len(candidates) > 1:
            iterations = sorted(int(epoch.outer_iteration) for epoch in candidates)
            raise ValueError(
                f"Multiple epochs found for year={year} with iterations={iterations}. "
                "Specify iteration or use converged=True."
            )
        simulation_epoch = candidates[0]
        return Epoch(archive=self, simulation_epoch=simulation_epoch)

    def _parse_table_name(self, name: str) -> tuple[str, str]:
        model, separator, logical_name = str(name).strip().lower().partition(".")
        if not separator or not model or not logical_name:
            raise ValueError(
                f"Table name must use '<model>.<logical_name>' form, got {name!r}."
            )
        return model, logical_name

    def _artifact_family_spec(self, model: str, logical_name: str) -> Mapping[str, str]:
        families = getattr(self.session, "artifact_families", None) or ARTIFACT_FAMILIES
        model_families = families.get(model)
        if model_families is None or logical_name not in model_families:
            available = {
                model_name: sorted(logicals.keys())
                for model_name, logicals in families.items()
            }
            raise KeyError(
                f"Unknown analysis table {model}.{logical_name}. Available: {available}."
            )
        return model_families[logical_name]

    def _seed_artifact(self, *, model: str, artifact_family: str) -> Any:
        finder = self.tracker.find_artifacts_by_params
        artifacts = list(
            finder(
                params=[f"{model}.artifact_family={artifact_family}"],
                namespace=model,
                limit=1,
            )
        )
        if not artifacts:
            artifacts = list(
                finder(
                    params=[f"artifact_family={artifact_family}"],
                    namespace=model,
                    limit=1,
                )
            )
        if not artifacts:
            raise RuntimeError(
                f"No artifacts found for {model}.artifact_family={artifact_family}."
            )
        return artifacts[0]

    def _localize_artifacts(self, artifacts: Sequence[Any]) -> None:
        if self.local_cache_run_dir is None:
            return
        existing = {
            entry["artifact_id"] for entry in self._localization_manifest["artifacts"]
        }
        for artifact in artifacts:
            artifact_id = str(getattr(artifact, "id", ""))
            if artifact_id in existing:
                continue
            source = self._artifact_source_path(artifact)
            destination = self._artifact_local_path(artifact, source=source)
            _copy_path(source, destination)
            entry = {
                "artifact_id": artifact_id,
                "key": str(getattr(artifact, "key", "")),
                "persistent_path": str(source),
                "local_path": str(destination),
            }
            self._localization_manifest["artifacts"].append(entry)
        self._write_localization_manifest()

    def _artifact_source_path(self, artifact: Any) -> Path:
        raw_abs_path = str(getattr(artifact, "abs_path", "") or "").strip()
        if raw_abs_path:
            abs_path = Path(raw_abs_path).expanduser()
            if abs_path.exists():
                return abs_path.resolve()
        uri = str(
            getattr(artifact, "uri", None)
            or getattr(artifact, "container_uri", None)
            or ""
        )
        if uri.startswith("workspace://"):
            relative = uri.removeprefix("workspace://").lstrip("/")
            source = self.persistent_archive_run_dir / relative
            if source.exists():
                return source.resolve()
        key = str(getattr(artifact, "key", "<unknown>"))
        raise FileNotFoundError(f"Could not resolve artifact path for {key}.")

    def _artifact_local_path(self, artifact: Any, *, source: Path) -> Path:
        uri = str(
            getattr(artifact, "uri", None)
            or getattr(artifact, "container_uri", None)
            or ""
        )
        if uri.startswith("workspace://"):
            relative = uri.removeprefix("workspace://").lstrip("/")
            return self.archive_run_dir / relative
        try:
            relative = source.resolve().relative_to(self.persistent_archive_run_dir)
        except ValueError:
            relative = (
                Path(".pilates")
                / "localized_artifacts"
                / str(getattr(artifact, "id", "artifact"))
                / source.name
            )
        return self.archive_run_dir / relative

    def _write_localization_manifest(self) -> None:
        path = self.archive_run_dir / ".pilates" / "analysis_localization_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._localization_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session, name)


def open_archive(
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
    artifact_families_env_var: Optional[str] = None,
    local_cache: Optional[str | Path] = None,
) -> Archive:
    return Archive.open(
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
        local_cache=local_cache,
    )
