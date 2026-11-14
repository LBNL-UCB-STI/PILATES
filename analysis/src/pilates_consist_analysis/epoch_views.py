from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
import os
from pathlib import Path
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
ARTIFACT_FAMILIES_ENV_VAR = "PILATES_ANALYSIS_ARTIFACT_FAMILIES_JSON"


def load_artifact_families_from_json(
    path: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    json_path = Path(path).expanduser()
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Artifact families JSON file not found: {json_path}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not read artifact families JSON file: {json_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid artifact families JSON in {json_path}: {exc.msg} (line {exc.lineno})"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"Artifact families JSON must contain an object: {json_path}"
        )
    return _normalize_artifact_families_mapping(
        payload,
        source=f"artifact families JSON '{json_path}'",
    )


def resolve_artifact_families(
    *,
    artifact_families: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
    artifact_families_json_path: Optional[str | Path] = None,
    env_var: str = ARTIFACT_FAMILIES_ENV_VAR,
) -> dict[str, dict[str, dict[str, str]]]:
    resolved = _normalize_artifact_families_mapping(
        ARTIFACT_FAMILIES,
        source="ARTIFACT_FAMILIES",
    )

    override_path = artifact_families_json_path
    if override_path is None:
        env_value = str(os.environ.get(env_var, "") or "").strip()
        if env_value:
            override_path = env_value
    if override_path is not None:
        resolved = _merge_artifact_families(
            resolved,
            load_artifact_families_from_json(override_path),
        )

    if artifact_families is not None:
        resolved = _merge_artifact_families(
            resolved,
            _normalize_artifact_families_mapping(
                artifact_families,
                source="artifact_families override",
            ),
        )

    return resolved


def _normalize_artifact_families_mapping(
    mapping: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, dict[str, dict[str, str]]]:
    normalized: dict[str, dict[str, dict[str, str]]] = {}
    for raw_model, raw_logicals in mapping.items():
        model_key = str(raw_model).strip().lower()
        if not model_key:
            raise RuntimeError(f"Invalid empty model key in {source}.")
        if not isinstance(raw_logicals, Mapping):
            raise RuntimeError(
                f"Artifact families for model '{model_key}' in {source} must be an object."
            )
        model_entry: dict[str, dict[str, str]] = {}
        for raw_logical, raw_spec in raw_logicals.items():
            logical_key = str(raw_logical).strip().lower()
            if not logical_key:
                raise RuntimeError(
                    f"Invalid empty logical name for model '{model_key}' in {source}."
                )
            if not isinstance(raw_spec, Mapping):
                raise RuntimeError(
                    f"Artifact family spec for '{model_key}.{logical_key}' in {source} must be an object."
                )
            spec_entry: dict[str, str] = {}
            for raw_field, raw_value in raw_spec.items():
                field = str(raw_field).strip()
                if not field:
                    raise RuntimeError(
                        f"Artifact family spec for '{model_key}.{logical_key}' in {source} "
                        "contains an empty field name."
                    )
                spec_entry[field] = str(raw_value).strip()
            model_entry[logical_key] = spec_entry
        normalized[model_key] = model_entry
    return normalized


def _merge_artifact_families(
    base: Mapping[str, Mapping[str, Mapping[str, str]]],
    override: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> dict[str, dict[str, dict[str, str]]]:
    merged: dict[str, dict[str, dict[str, str]]] = {
        model: {logical: dict(spec) for logical, spec in logicals.items()}
        for model, logicals in base.items()
    }
    for model, logicals in override.items():
        target_model = merged.setdefault(model, {})
        for logical_name, spec in logicals.items():
            target_spec = dict(target_model.get(logical_name, {}))
            target_spec.update(spec)
            target_model[logical_name] = target_spec
    return merged


@dataclass
class EpochViews:
    epoch: SimulationEpoch
    tracker: Any
    artifact_families: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None
    artifact_families_json_path: Optional[str | Path] = None
    artifact_families_env_var: str = ARTIFACT_FAMILIES_ENV_VAR
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _resolved_artifact_families: dict[str, dict[str, dict[str, str]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._resolved_artifact_families = resolve_artifact_families(
            artifact_families=self.artifact_families,
            artifact_families_json_path=self.artifact_families_json_path,
            env_var=self.artifact_families_env_var,
        )

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
        try:
            openmatrix_metadata = self.tracker.openmatrix_metadata
        except AttributeError as exc:
            raise RuntimeError(
                "Tracker does not expose openmatrix metadata views."
            ) from exc

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
            output = output.sort_values(sort_cols, na_position="last").reset_index(
                drop=True
            )
        return output

    def query(self, sql: str) -> pd.DataFrame:
        formatted_sql = sql.format(views=self)
        try:
            return self.tracker.query_df(formatted_sql)
        except AttributeError:
            pass
        try:
            return self.tracker.db.query_df(formatted_sql)
        except AttributeError as exc:
            raise RuntimeError("Tracker DB query interface is not available.") from exc

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

        family_spec = _family_spec(
            model_key,
            logical_key,
            artifact_families=self._resolved_artifact_families,
        )
        artifact_family = family_spec["artifact_family"]

        try:
            run_id = str(run.id or "").strip()
        except AttributeError:
            run_id = ""
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

        try:
            create_grouped_view = self.tracker.create_grouped_view
        except AttributeError as exc:
            raise RuntimeError(
                "Tracker does not expose grouped view creation methods "
                "(expected create_grouped_view)."
            ) from exc

        create_grouped_view(
            view_name=view_name,
            schema_id=schema_id,
            namespace=model_key,
            params=[f"{model_key}.artifact_family={artifact_family}"],
            attach_facets=["artifact_family"],
            include_system_columns=True,
            mode="hybrid",
            if_exists="replace",
            missing_files="warn",
            run_id=run_id,
            model=model_key,
        )
        self._cache[cache_key] = view_name
        return view_name

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

        try:
            matches = self.tracker.find_artifacts_by_params(
                params=[f"{model}.artifact_family={artifact_family}"],
                namespace=model,
                limit=limit,
            )
            if matches:
                collected.extend(list(matches))
        except Exception:
            pass

        for kwargs in (
            {"creator": run_id, "limit": limit},
            {"consumer": run_id, "limit": limit},
            {"limit": limit},
        ):
            try:
                matches = self.tracker.find_artifacts(**kwargs)
            except Exception:
                continue
            if matches:
                collected.extend(list(matches))
                if kwargs.get("creator") == run_id:
                    break

        deduped: list[Any] = []
        seen: set[str] = set()
        for artifact in collected:
            artifact_id = str(artifact.id or artifact.key or "").strip()
            if not artifact_id or artifact_id in seen:
                continue
            seen.add(artifact_id)
            deduped.append(artifact)

        run_filtered: list[Any] = []
        for artifact in deduped:
            artifact_run_id = str(artifact.run_id or "").strip()
            if artifact_run_id and artifact_run_id != run_id:
                continue
            run_filtered.append(artifact)

        family_filtered = [
            artifact
            for artifact in run_filtered
            if _artifact_matches_family(
                artifact, model=model, artifact_family=artifact_family
            )
        ]
        return family_filtered

    def _load_matrices(
        self,
        *,
        metadata_view: Any,
        concept_key: str,
        run_ids: list[str],
    ) -> Optional[pd.DataFrame]:
        try:
            get_matrices = metadata_view.get_matrices
        except AttributeError:
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
            self._resolved_artifact_families.get("activitysim", {})
            .get("skims", {})
            .get("concept_key")
        )
        if configured:
            concept_keys.append(str(configured))

        for run_id in run_ids:
            try:
                artifacts = self.tracker.find_artifacts(creator=run_id, limit=5000)
            except Exception:
                continue
            for artifact in artifacts or []:
                driver = str(artifact.driver or "").strip().lower()
                key = str(artifact.key or "").strip()
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


def epoch_views(
    epoch: SimulationEpoch,
    tracker: Any,
    *,
    artifact_families: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
    artifact_families_json_path: Optional[str | Path] = None,
    artifact_families_env_var: str = ARTIFACT_FAMILIES_ENV_VAR,
) -> EpochViews:
    return EpochViews(
        epoch=epoch,
        tracker=tracker,
        artifact_families=artifact_families,
        artifact_families_json_path=artifact_families_json_path,
        artifact_families_env_var=artifact_families_env_var,
    )


def _family_spec(
    model: str,
    logical_name: str,
    *,
    artifact_families: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> Mapping[str, str]:
    by_model = artifact_families.get(model)
    if by_model is None:
        available = ", ".join(sorted(artifact_families.keys())) or "<none>"
        raise AttributeError(
            f"Model '{model}' is not configured in artifact families mapping. "
            f"Configured models: {available}."
        )
    spec = by_model.get(logical_name)
    if spec is None:
        available = ", ".join(sorted(by_model.keys())) or "<none>"
        raise AttributeError(
            f"Logical artifact '{logical_name}' is not configured for model '{model}'. "
            f"Configured logical names: {available}."
        )
    artifact_family = str(spec.get("artifact_family", "") or "").strip()
    if not artifact_family:
        raise RuntimeError(
            f"Artifact family spec for '{model}.{logical_name}' is missing required "
            "'artifact_family'."
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
    try:
        direct = str(artifact.schema_id or "").strip()
    except AttributeError:
        direct = ""
    if direct:
        return direct

    meta = artifact.meta
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


def _artifact_matches_family(
    artifact: Any, *, model: str, artifact_family: str
) -> bool:
    observed = _artifact_family(artifact, model=model)
    if observed:
        return observed == artifact_family
    key = str(artifact.key or "").strip().lower()
    return artifact_family.lower() in key if key else False


def _artifact_family(artifact: Any, *, model: str) -> Optional[str]:
    meta = artifact.meta
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


__all__ = [
    "ARTIFACT_FAMILIES",
    "ARTIFACT_FAMILIES_ENV_VAR",
    "EpochViews",
    "epoch_views",
    "load_artifact_families_from_json",
    "resolve_artifact_families",
]
