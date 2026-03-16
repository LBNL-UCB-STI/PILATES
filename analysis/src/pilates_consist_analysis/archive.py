from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from .api import AnalysisSession
from .epoch_api import Epoch
from .epochs import SimulationEpoch
from .run_index import RunIndex, build_run_index
from .runtime import get_db_health, get_db_health_issues


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
    def __init__(self, session: AnalysisSession) -> None:
        self.session = session
        self._run_index: Optional[RunIndex] = None

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
        artifact_families: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
        artifact_families_json_path: Optional[str | Path] = None,
        artifact_families_env_var: Optional[str] = None,
    ) -> "Archive":
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
        return cls(AnalysisSession.open(**session_kwargs))

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
    )
