from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import inspect
from typing import Any, Mapping, Optional

import pandas as pd

from .epochs import SimulationEpoch


ARTIFACT_FAMILIES: dict[str, dict[str, dict[str, str]]] = {
    "activitysim": {
        "trips": {"artifact_family": "trips", "concept_key": "trips_asim_out"},
        "persons": {"artifact_family": "persons", "concept_key": "persons"},
        "households": {"artifact_family": "households", "concept_key": "households"},
        "land_use": {"artifact_family": "land_use", "concept_key": "land_use"},
        "skims": {"artifact_family": "omx_skims", "concept_key": "omx_skims"},
    },
    "beam": {
        "linkstats": {
            "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet",
            "concept_key": "linkstats",
        }
    },
    "urbansim": {
        "households": {"artifact_family": "households", "concept_key": "households"},
        "persons": {"artifact_family": "persons", "concept_key": "persons"},
        "jobs": {"artifact_family": "jobs", "concept_key": "jobs"},
    },
}


@dataclass
class EpochViews:
    epoch: SimulationEpoch
    tracker: Any
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @cached_property
    def trips(self) -> str:
        return self._view("activitysim", "trips")

    @cached_property
    def persons(self) -> str:
        return self._view("activitysim", "persons")

    @cached_property
    def households(self) -> str:
        return self._view("activitysim", "households")

    @cached_property
    def land_use(self) -> str:
        return self._view("activitysim", "land_use")

    @cached_property
    def linkstats(self) -> str:
        return self._view("beam", "linkstats")

    @cached_property
    def urbansim_households(self) -> str:
        return self._view("urbansim", "households")

    @cached_property
    def urbansim_persons(self) -> str:
        return self._view("urbansim", "persons")

    @cached_property
    def urbansim_jobs(self) -> str:
        return self._view("urbansim", "jobs")

    @cached_property
    def skim_summary(self) -> pd.DataFrame:
        if not hasattr(self.tracker, "openmatrix_metadata"):
            raise RuntimeError("Tracker does not expose openmatrix metadata views.")

        run_ids = sorted(self.epoch.run_ids().values())
        concept_keys = self._discover_skim_concept_keys(run_ids)
        if not concept_keys:
            return pd.DataFrame(
                columns=[
                    "concept_key",
                    "run_id",
                    "year",
                    "iteration",
                    "matrix_name",
                    "n_rows",
                    "n_cols",
                ]
            )

        frames: list[pd.DataFrame] = []
        openmatrix_metadata = self.tracker.openmatrix_metadata
        for concept_key in concept_keys:
            try:
                metadata_view = openmatrix_metadata(concept_key)
            except Exception:
                continue
            matrices = self._load_matrices(
                metadata_view=metadata_view,
                concept_key=concept_key,
                run_ids=run_ids,
            )
            if matrices is None or matrices.empty:
                continue
            frame = matrices.copy()
            if "concept_key" not in frame.columns:
                frame["concept_key"] = concept_key
            frames.append(frame)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "concept_key",
                    "run_id",
                    "year",
                    "iteration",
                    "matrix_name",
                    "n_rows",
                    "n_cols",
                ]
            )

        output = pd.concat(frames, ignore_index=True)
        sort_cols = [
            col
            for col in ("concept_key", "run_id", "year", "iteration", "matrix_name")
            if col in output.columns
        ]
        if sort_cols:
            output = output.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        return output

    def query(self, sql: str) -> pd.DataFrame:
        db = getattr(self.tracker, "db", None)
        if db is None or not hasattr(db, "query"):
            raise RuntimeError("Tracker DB query interface is not available.")
        result = db.query(sql.format(views=self))
        if not hasattr(result, "df"):
            raise RuntimeError("Tracker DB query result does not expose .df().")
        return result.df()

    def _view(self, model: str, logical_name: str) -> str:
        model_key = str(model).strip().lower()
        logical_key = str(logical_name).strip().lower()

        run = self.epoch.runs.get(model_key)
        if run is None:
            available = ", ".join(self.epoch.models) or "<none>"
            raise AttributeError(
                f"Model '{model_key}' is not present in epoch "
                f"(year={self.epoch.year}, iteration={self.epoch.outer_iteration}, "
                f"scenario_id={self.epoch.scenario_id}). Available models: {available}."
            )

        family_spec = _family_spec(model_key, logical_key)
        artifact_family = family_spec["artifact_family"]

        run_id = str(getattr(run, "id", "") or "").strip()
        if not run_id:
            raise RuntimeError(
                f"Epoch run for model '{model_key}' does not expose a valid run id."
            )

        cache_key = f"view:{model_key}:{logical_key}"
        cached = self._cache.get(cache_key)
        if isinstance(cached, str) and cached:
            return cached

        view_name = _stable_view_name(
            epoch=self.epoch,
            model=model_key,
            logical_name=logical_key,
            run_id=run_id,
            artifact_family=artifact_family,
        )
        schema_id = self._resolve_schema_id(
            model=model_key,
            run_id=run_id,
            artifact_family=artifact_family,
        )
        if not schema_id:
            raise RuntimeError(
                "Could not resolve schema_id for epoch-scoped view creation "
                f"(model={model_key}, logical_name={logical_key}, run_id={run_id}, "
                f"artifact_family={artifact_family})."
            )

        params = [f"{model_key}.artifact_family={artifact_family}"]
        predicates = _build_artifact_predicates(self.tracker, params)
        attach_facets = ["artifact_family"]

        create_hybrid = _resolve_grouped_hybrid_creator(self.tracker)
        if callable(create_hybrid):
            kwargs = {
                "view_name": view_name,
                "schema_id": schema_id,
                "predicates": predicates,
                "namespace": model_key,
                "attach_facets": attach_facets,
                "include_system_columns": True,
                "mode": "hybrid",
                "if_exists": "replace",
                "missing_files": "warn",
                "run_id": run_id,
                "model": model_key,
            }
            _call_with_supported_kwargs(create_hybrid, kwargs)
            self._cache[cache_key] = view_name
            return view_name

        create_grouped_view = getattr(self.tracker, "create_grouped_view", None)
        if callable(create_grouped_view):
            kwargs = {
                "view_name": view_name,
                "schema_id": schema_id,
                "namespace": model_key,
                "params": params,
                "attach_facets": attach_facets,
                "include_system_columns": True,
                "mode": "hybrid",
                "if_exists": "replace",
                "missing_files": "warn",
                "run_id": run_id,
                "model": model_key,
            }
            _call_with_supported_kwargs(create_grouped_view, kwargs)
            self._cache[cache_key] = view_name
            return view_name

        raise RuntimeError(
            "Tracker does not expose grouped view creation methods "
            "(expected view_factory.create_grouped_hybrid_view or create_grouped_view)."
        )

    def _resolve_schema_id(
        self,
        *,
        model: str,
        run_id: str,
        artifact_family: str,
    ) -> Optional[str]:
        artifacts = self._artifacts_for_run_family(
            model=model,
            run_id=run_id,
            artifact_family=artifact_family,
        )
        for artifact in artifacts:
            schema_id = _artifact_schema_id(artifact)
            if schema_id:
                return schema_id

        selector = getattr(self.tracker, "select_artifact_schema_for_artifact", None)
        if callable(selector):
            for artifact in artifacts:
                artifact_id = getattr(artifact, "id", None)
                if artifact_id is None:
                    continue
                try:
                    selected = selector(artifact_id=artifact_id)
                except Exception:
                    continue
                schema_id = str(getattr(selected, "schema_id", "") or "").strip()
                if schema_id:
                    return schema_id
        return None

    def _artifacts_for_run_family(
        self,
        *,
        model: str,
        run_id: str,
        artifact_family: str,
        limit: int = 1000,
    ) -> list[Any]:
        collected: list[Any] = []

        by_params = getattr(self.tracker, "find_artifacts_by_params", None)
        if callable(by_params):
            try:
                matches = _call_with_supported_kwargs(
                    by_params,
                    {
                        "params": [f"{model}.artifact_family={artifact_family}"],
                        "namespace": model,
                        "limit": limit,
                    },
                )
                if matches:
                    collected.extend(list(matches))
            except Exception:
                pass

        find_artifacts = getattr(self.tracker, "find_artifacts", None)
        if callable(find_artifacts):
            for kwargs in (
                {"creator": run_id, "limit": limit},
                {"consumer": run_id, "limit": limit},
                {"limit": limit},
            ):
                try:
                    matches = _call_with_supported_kwargs(find_artifacts, kwargs)
                except Exception:
                    continue
                if matches:
                    collected.extend(list(matches))
                    if kwargs.get("creator") == run_id:
                        break

        deduped: list[Any] = []
        seen: set[str] = set()
        for artifact in collected:
            artifact_id = str(getattr(artifact, "id", "") or "").strip()
            if not artifact_id:
                artifact_id = str(getattr(artifact, "key", "") or "").strip()
            if not artifact_id or artifact_id in seen:
                continue
            seen.add(artifact_id)
            deduped.append(artifact)

        run_filtered: list[Any] = []
        for artifact in deduped:
            artifact_run_id = str(getattr(artifact, "run_id", "") or "").strip()
            if artifact_run_id and artifact_run_id != run_id:
                continue
            run_filtered.append(artifact)

        family_filtered = [
            artifact
            for artifact in run_filtered
            if _artifact_matches_family(artifact, model=model, artifact_family=artifact_family)
        ]
        return family_filtered

    def _load_matrices(
        self,
        *,
        metadata_view: Any,
        concept_key: str,
        run_ids: list[str],
    ) -> Optional[pd.DataFrame]:
        get_matrices = getattr(metadata_view, "get_matrices", None)
        if not callable(get_matrices):
            return None
        calls = [
            {"run_ids": run_ids or None, "year": int(self.epoch.year)},
            {"run_ids": run_ids or None},
            {},
        ]
        for kwargs in calls:
            try:
                return get_matrices(concept_key, **kwargs)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    def _discover_skim_concept_keys(self, run_ids: list[str]) -> list[str]:
        concept_keys: list[str] = []

        configured = (
            ARTIFACT_FAMILIES.get("activitysim", {})
            .get("skims", {})
            .get("concept_key")
        )
        if configured:
            concept_keys.append(str(configured))

        find_artifacts = getattr(self.tracker, "find_artifacts", None)
        if callable(find_artifacts):
            for run_id in run_ids:
                try:
                    artifacts = _call_with_supported_kwargs(
                        find_artifacts,
                        {"creator": run_id, "limit": 5000},
                    )
                except Exception:
                    continue
                for artifact in artifacts or []:
                    driver = str(getattr(artifact, "driver", "") or "").strip().lower()
                    key = str(getattr(artifact, "key", "") or "").strip()
                    if not key:
                        continue
                    if driver != "openmatrix" and "skim" not in key.lower():
                        continue
                    concept_keys.append(key)

        output: list[str] = []
        seen: set[str] = set()
        for key in concept_keys:
            normalized = str(key).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output


def epoch_views(epoch: SimulationEpoch, tracker: Any) -> EpochViews:
    return EpochViews(epoch=epoch, tracker=tracker)


def _family_spec(model: str, logical_name: str) -> Mapping[str, str]:
    by_model = ARTIFACT_FAMILIES.get(model)
    if by_model is None:
        available = ", ".join(sorted(ARTIFACT_FAMILIES.keys())) or "<none>"
        raise AttributeError(
            f"Model '{model}' is not configured in ARTIFACT_FAMILIES. "
            f"Configured models: {available}."
        )
    spec = by_model.get(logical_name)
    if spec is None:
        available = ", ".join(sorted(by_model.keys())) or "<none>"
        raise AttributeError(
            f"Logical artifact '{logical_name}' is not configured for model '{model}'. "
            f"Configured logical names: {available}."
        )
    return spec


def _stable_view_name(
    *,
    epoch: SimulationEpoch,
    model: str,
    logical_name: str,
    run_id: str,
    artifact_family: str,
) -> str:
    payload = "|".join(
        [
            str(epoch.year),
            str(epoch.outer_iteration),
            str(epoch.scenario_id or ""),
            model,
            logical_name,
            run_id,
            artifact_family,
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"v_epoch_{model}_{logical_name}_{digest}"


def _artifact_schema_id(artifact: Any) -> Optional[str]:
    direct = str(getattr(artifact, "schema_id", "") or "").strip()
    if direct:
        return direct

    meta = getattr(artifact, "meta", None)
    if not isinstance(meta, Mapping):
        return None

    raw = meta.get("schema_id")
    if raw:
        return str(raw).strip() or None

    schema = meta.get("schema")
    if isinstance(schema, Mapping):
        schema_id = schema.get("id") or schema.get("schema_id")
        if schema_id:
            return str(schema_id).strip() or None
    return None


def _artifact_matches_family(artifact: Any, *, model: str, artifact_family: str) -> bool:
    observed = _artifact_family(artifact, model=model)
    if observed:
        return observed == artifact_family
    key = str(getattr(artifact, "key", "") or "").strip().lower()
    return artifact_family.lower() in key if key else False


def _artifact_family(artifact: Any, *, model: str) -> Optional[str]:
    direct = str(getattr(artifact, "artifact_family", "") or "").strip()
    if direct:
        return direct

    meta = getattr(artifact, "meta", None)
    if not isinstance(meta, Mapping):
        return None

    facet = meta.get("facet")
    if isinstance(facet, Mapping):
        family = facet.get("artifact_family")
        if family:
            return str(family).strip() or None

    namespaced = meta.get(model)
    if isinstance(namespaced, Mapping):
        family = namespaced.get("artifact_family")
        if family:
            return str(family).strip() or None

    dotted = meta.get(f"{model}.artifact_family")
    if dotted:
        return str(dotted).strip() or None

    fallback = meta.get("artifact_family")
    if fallback:
        return str(fallback).strip() or None

    return None


def _resolve_grouped_hybrid_creator(tracker: Any) -> Optional[Any]:
    view_factory = getattr(tracker, "view_factory", None)
    creator = getattr(view_factory, "create_grouped_hybrid_view", None)
    if callable(creator):
        return creator
    try:
        from consist.core.views import ViewFactory
    except Exception:
        return None
    try:
        fallback = getattr(ViewFactory(tracker), "create_grouped_hybrid_view", None)
    except Exception:
        return None
    return fallback if callable(fallback) else None


def _build_artifact_predicates(tracker: Any, params: list[str]) -> list[dict[str, Any]]:
    if not params:
        return []
    parser = getattr(tracker, "_parse_artifact_param_expression", None)
    if callable(parser):
        return [parser(param) for param in params]
    output: list[dict[str, Any]] = []
    for param in params:
        lhs, _, rhs = str(param).partition("=")
        namespace = None
        key_path = lhs
        if "." in lhs:
            namespace, key_path = lhs.split(".", 1)
        output.append(
            {
                "namespace": namespace,
                "key_path": key_path,
                "op": "=",
                "kind": "str",
                "value": rhs,
            }
        )
    return output


def _call_with_supported_kwargs(func: Any, kwargs: Mapping[str, Any]) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**dict(kwargs))
    accepts_var_kw = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kw:
        return func(**dict(kwargs))
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(**filtered)


__all__ = ["ARTIFACT_FAMILIES", "EpochViews", "epoch_views"]
