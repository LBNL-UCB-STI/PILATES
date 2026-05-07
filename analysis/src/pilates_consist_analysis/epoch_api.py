from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable, Optional

import pandas as pd

from .epoch_views import EpochViews
from .epochs import SimulationEpoch

if TYPE_CHECKING:
    from .archive import Archive


_TABLE_VIEW_CANDIDATES = {
    "trips": ("trips",),
    "persons": ("persons", "urbansim_persons"),
    "households": ("households", "urbansim_households"),
    "land_use": ("land_use",),
    "linkstats": ("linkstats",),
    "urbansim_persons": ("urbansim_persons",),
    "urbansim_households": ("urbansim_households",),
    "urbansim_jobs": ("urbansim_jobs",),
}


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _column_sql(columns: Optional[Iterable[str]]) -> str:
    if columns is None:
        return "*"
    normalized = [str(column).strip() for column in columns if str(column).strip()]
    if not normalized:
        return "*"
    return ", ".join(_quote_ident(column) for column in normalized)


@dataclass(frozen=True)
class Epoch:
    archive: "Archive"
    simulation_epoch: SimulationEpoch

    @property
    def raw(self) -> SimulationEpoch:
        return self.simulation_epoch

    @property
    def year(self) -> int:
        return int(self.simulation_epoch.year)

    @property
    def outer_iteration(self) -> int:
        return int(self.simulation_epoch.outer_iteration)

    @property
    def scenario_id(self) -> Optional[str]:
        return self.simulation_epoch.scenario_id

    @property
    def is_complete(self) -> bool:
        return bool(self.simulation_epoch.is_complete)

    @property
    def models(self) -> list[str]:
        return list(self.simulation_epoch.models)

    @cached_property
    def views(self) -> EpochViews:
        return self.archive.session.views(self.simulation_epoch)

    @cached_property
    def tables(self) -> "EpochTables":
        return EpochTables(self)

    def summary(self) -> pd.DataFrame:
        row = {
            "scenario_id": self.scenario_id,
            "year": self.year,
            "outer_iteration": self.outer_iteration,
            "is_complete": self.is_complete,
            "model_count": int(len(self.models)),
            "models": ",".join(self.models),
            "available_tables": ",".join(self.tables.available()),
        }
        row.update(
            {f"{model}_run_id": run_id for model, run_id in self.run_ids().items()}
        )
        return pd.DataFrame([row])

    def run_ids(self) -> dict[str, str]:
        return self.simulation_epoch.run_ids()

    def model_run(self, model: str) -> Any:
        return self.simulation_epoch.model_run(model)

    def sql(self, sql: str) -> pd.DataFrame:
        return self.views.query(sql)

    def query(self, sql: str) -> pd.DataFrame:
        return self.sql(sql)

    def __repr__(self) -> str:
        return (
            "Epoch("
            f"scenario_id={self.scenario_id!r}, "
            f"year={self.year}, "
            f"outer_iteration={self.outer_iteration}, "
            f"models={self.models}, "
            f"is_complete={self.is_complete})"
        )


class EpochTables:
    def __init__(self, epoch: Epoch) -> None:
        self.epoch = epoch

    @property
    def views(self) -> EpochViews:
        return self.epoch.views

    def available(self) -> list[str]:
        available: list[str] = []
        for name in ("trips", "persons", "households", "land_use", "linkstats"):
            if self.supports(name):
                available.append(name)
        if self.supports("skim_summary"):
            available.append("skim_summary")
        for name in ("urbansim_persons", "urbansim_households", "urbansim_jobs"):
            if self.supports(name):
                available.append(name)
        return available

    def supports(self, name: str) -> bool:
        normalized = str(name).strip().lower()
        if normalized == "skim_summary":
            try:
                self.views.skim_summary
                return True
            except Exception:
                return False
        try:
            self._resolve_view_attr(normalized)
            return True
        except Exception:
            return False

    def view_name(self, name: str) -> str:
        view_attr = self._resolve_view_attr(name)
        return str(getattr(self.views, view_attr))

    def load(
        self,
        name: str,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        normalized = str(name).strip().lower()
        if normalized == "skim_summary":
            frame = self.views.skim_summary
            return frame.copy()

        view_attr = self._resolve_view_attr(normalized)
        select_sql = _column_sql(columns)
        sql = f"SELECT {select_sql} FROM {{views.{view_attr}}}"
        if where is not None and str(where).strip():
            sql += f" WHERE {str(where).strip()}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.epoch.sql(sql)

    def trips(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load("trips", columns=columns, where=where, limit=limit)

    def persons(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load("persons", columns=columns, where=where, limit=limit)

    def households(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load("households", columns=columns, where=where, limit=limit)

    def land_use(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load("land_use", columns=columns, where=where, limit=limit)

    def linkstats(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load("linkstats", columns=columns, where=where, limit=limit)

    def skim_summary(self) -> pd.DataFrame:
        return self.load("skim_summary")

    def urbansim_persons(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load("urbansim_persons", columns=columns, where=where, limit=limit)

    def urbansim_households(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load(
            "urbansim_households",
            columns=columns,
            where=where,
            limit=limit,
        )

    def urbansim_jobs(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load("urbansim_jobs", columns=columns, where=where, limit=limit)

    def _resolve_view_attr(self, name: str) -> str:
        normalized = str(name).strip().lower()
        candidates = _TABLE_VIEW_CANDIDATES.get(normalized)
        if candidates is None:
            raise AttributeError(f"Unknown epoch table '{name}'.")

        errors: list[str] = []
        for candidate in candidates:
            try:
                getattr(self.views, candidate)
                return candidate
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue

        raise AttributeError(
            f"Epoch table '{normalized}' is not available for this epoch. "
            f"Tried {list(candidates)}. Errors: {'; '.join(errors)}"
        )
