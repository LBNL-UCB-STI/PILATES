from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .epoch_views import epoch_views
from .epochs import build_epoch_panel
from .packaging import export_bundle
from .runset import runset_from_query, runset_run_ids


@dataclass
class ArtifactIngestSpec:
    """Specification for logging and optional ingestion of one artifact file."""

    path: str | Path
    key: Optional[str] = None
    direction: str = "output"
    driver: Optional[str] = None
    artifact_family: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableTransformSpec:
    """Simple table transform for export workflows."""

    columns: Optional[list[str]] = None
    rename: dict[str, str] = field(default_factory=dict)
    where_sql: Optional[str] = None


def ingest_artifacts(
    tracker: Any,
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
    """
    Log files as artifacts in a dedicated analysis run and optionally ingest them.

    This is intended for ad-hoc HPC workflows where you ingest archived outputs
    into the DB before downstream analysis queries.
    """
    if not artifact_specs:
        raise ValueError("artifact_specs must contain at least one artifact.")
    access_mode = str(getattr(tracker, "access_mode", "standard") or "standard")
    if access_mode != "standard":
        raise RuntimeError(
            "ingest_artifacts requires tracker access_mode='standard' because it creates a new run. "
            f"Current mode is '{access_mode}'."
        )

    resolved_run_id = run_id or _default_run_id(model)
    resolved_tags = _build_ingest_tags(tags=tags, scenario_id=scenario_id, seed=seed)
    run_facet = _build_run_facet(
        scenario_id=scenario_id,
        seed=seed,
        year=year,
        iteration=iteration,
    )

    artifact_rows: list[dict[str, Any]] = []
    with _start_run_context(
        tracker,
        run_id=resolved_run_id,
        model=model,
        tags=resolved_tags,
        config=dict(run_config or {}),
        year=year,
        iteration=iteration,
        parent_run_id=parent_run_id,
        facet=run_facet or None,
    ):
        for spec in artifact_specs:
            resolved_spec = _resolve_spec(spec)
            path = Path(resolved_spec.path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Artifact path does not exist: {path}")

            meta = dict(resolved_spec.meta)
            if resolved_spec.artifact_family:
                meta.setdefault("artifact_family", resolved_spec.artifact_family)

            artifact = tracker.log_artifact(
                str(path),
                key=resolved_spec.key,
                direction=resolved_spec.direction,
                driver=resolved_spec.driver,
                **meta,
            )

            ingest_result_type = None
            if ingest_data:
                ingest_result = tracker.ingest(artifact, profile_schema=profile_schema)
                ingest_result_type = type(ingest_result).__name__

            artifact_rows.append(
                {
                    "artifact_id": str(getattr(artifact, "id", "") or ""),
                    "key": str(getattr(artifact, "key", resolved_spec.key) or resolved_spec.key),
                    "uri": str(getattr(artifact, "uri", "") or ""),
                    "direction": resolved_spec.direction,
                    "artifact_family": resolved_spec.artifact_family,
                    "driver": resolved_spec.driver,
                    "ingested": bool(getattr(artifact, "meta", {}).get("is_ingested", False))
                    if isinstance(getattr(artifact, "meta", None), dict)
                    else False,
                    "ingest_result_type": ingest_result_type,
                }
            )

    return {
        "run_id": resolved_run_id,
        "model": model,
        "artifact_count": len(artifact_rows),
        "ingest_data": bool(ingest_data),
        "profile_schema": bool(profile_schema),
        "artifacts": artifact_rows,
    }


def export_scenario_bundle(
    tracker: Any,
    *,
    archive_run_dir: str | Path,
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
    converged_group_by: Optional[Sequence[str]] = None,
    latest_group_by: Optional[Sequence[str]] = None,
    include_data: bool = True,
    include_snapshots: bool = False,
    include_children: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resolve scenario run IDs via RunSet filters and export a standalone DB shard."""

    runset = runset_from_query(
        tracker=tracker,
        runset_name="scenario-export",
        tags=tags,
        year=year,
        iteration=iteration,
        model=model,
        status=status,
        metadata=metadata,
        limit=limit,
    )

    if scenario_id is not None:
        runset = runset.filter(scenario_id=scenario_id)
    if seed is not None:
        runset = runset.filter(seed=seed)

    if use_converged:
        runset = runset.converged(group_by=list(converged_group_by) if converged_group_by else None)
    if latest_group_by:
        runset = runset.latest(group_by=list(latest_group_by))

    run_ids = runset_run_ids(runset)
    if not run_ids:
        raise ValueError("No runs matched export filters.")

    export_payload = export_bundle(
        tracker,
        archive_run_dir=archive_run_dir,
        run_ids=run_ids,
        out_path=out_path,
        include_data=include_data,
        include_snapshots=include_snapshots,
        include_children=include_children,
        dry_run=dry_run,
    )
    export_payload["selected_run_ids"] = run_ids
    return export_payload


def export_sql_query(
    tracker: Any,
    *,
    sql: str,
    output_path: str | Path,
    output_format: str = "csv",
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Execute arbitrary SQL against the tracker DB and export rows to disk."""
    if not hasattr(tracker, "db"):
        raise RuntimeError("Tracker does not expose db query interface.")

    relation = tracker.db.query(sql)
    frame = relation.df()
    if limit is not None and limit >= 0:
        frame = frame.head(int(limit))

    resolved_path = _write_table(frame, output_path=output_path, output_format=output_format)
    return {
        "output_path": str(resolved_path),
        "output_format": output_format,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
    }


def export_activitysim_inputs(
    tracker: Any,
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
    """Export ActivitySim trips/persons tables for one epoch with optional transforms."""
    if not include_trips and not include_persons:
        raise ValueError("At least one of include_trips/include_persons must be True.")

    epoch = _resolve_activitysim_epoch(
        tracker,
        scenario_id=scenario_id,
        year=year,
        iteration=iteration,
        use_converged=use_converged,
    )
    views = epoch_views(epoch, tracker)

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "year": int(epoch.year),
        "iteration": int(epoch.outer_iteration),
        "scenario_id": epoch.scenario_id,
        "run_id": str(epoch.model_run("activitysim").id),
        "files": {},
        "row_counts": {},
    }

    if include_trips:
        trips_frame = _read_view_table(tracker, views.trips, transform=trips)
        trips_path = _write_table(
            trips_frame,
            output_path=resolved_output_dir / _table_filename("trips", output_format),
            output_format=output_format,
        )
        payload["files"]["trips"] = str(trips_path)
        payload["row_counts"]["trips"] = int(len(trips_frame))

    if include_persons:
        persons_frame = _read_view_table(tracker, views.persons, transform=persons)
        persons_path = _write_table(
            persons_frame,
            output_path=resolved_output_dir / _table_filename("persons", output_format),
            output_format=output_format,
        )
        payload["files"]["persons"] = str(persons_path)
        payload["row_counts"]["persons"] = int(len(persons_frame))

    return payload


def parse_artifact_arg(raw: str, *, direction: str = "output", driver: Optional[str] = None, artifact_family: Optional[str] = None) -> ArtifactIngestSpec:
    """Parse CLI artifact arg in KEY=PATH or PATH form into ArtifactIngestSpec."""
    value = str(raw).strip()
    if not value:
        raise ValueError("Artifact argument cannot be empty.")

    key: Optional[str]
    path_text: str
    if "=" in value:
        maybe_key, maybe_path = value.split("=", 1)
        if maybe_key.strip() and maybe_path.strip():
            key = maybe_key.strip()
            path_text = maybe_path.strip()
        else:
            key = None
            path_text = value
    else:
        key = None
        path_text = value

    return ArtifactIngestSpec(
        path=path_text,
        key=key,
        direction=direction,
        driver=driver,
        artifact_family=artifact_family,
    )


def parse_columns_arg(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    return values or None


def parse_rename_args(raw_values: Optional[Sequence[str]]) -> dict[str, str]:
    rename: dict[str, str] = {}
    for raw in raw_values or []:
        value = str(raw).strip()
        if not value:
            continue
        if ":" not in value:
            raise ValueError(f"Invalid rename mapping '{value}'. Use old:new format.")
        old, new = value.split(":", 1)
        old_key = old.strip()
        new_key = new.strip()
        if not old_key or not new_key:
            raise ValueError(f"Invalid rename mapping '{value}'. Use old:new format.")
        rename[old_key] = new_key
    return rename


def _resolve_activitysim_epoch(
    tracker: Any,
    *,
    scenario_id: Optional[str],
    year: Optional[int],
    iteration: Optional[int],
    use_converged: bool,
):
    panel = build_epoch_panel(tracker, scenario_id=scenario_id, models=["activitysim"])
    if use_converged:
        panel = panel.converged_epochs()

    if not panel.epochs:
        raise ValueError("No ActivitySim epochs available for the requested filters.")

    resolved_year = int(year) if year is not None else max(panel.years())
    candidates = [epoch for epoch in panel.epochs if int(epoch.year) == resolved_year]

    if iteration is not None:
        candidates = [epoch for epoch in candidates if int(epoch.outer_iteration) == int(iteration)]

    if not candidates:
        raise ValueError(
            f"No ActivitySim epoch found for year={resolved_year}"
            + (f", iteration={int(iteration)}" if iteration is not None else "")
            + "."
        )
    if len(candidates) > 1:
        iterations = sorted({int(epoch.outer_iteration) for epoch in candidates})
        raise ValueError(
            f"Multiple ActivitySim epochs found for year={resolved_year} iterations={iterations}. "
            "Pass iteration or enable converged selection."
        )
    return candidates[0]


def _read_view_table(tracker: Any, view_name: str, *, transform: Optional[TableTransformSpec]) -> pd.DataFrame:
    where_sql = (transform.where_sql.strip() if transform and transform.where_sql else "")
    sql = f"SELECT * FROM {view_name}"
    if where_sql:
        sql = f"{sql} WHERE {where_sql}"

    frame = tracker.db.query(sql).df()
    if transform is None:
        return frame

    if transform.columns:
        missing = [column for column in transform.columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Columns not found in source table {view_name}: {missing}")
        frame = frame.loc[:, transform.columns]

    if transform.rename:
        frame = frame.rename(columns=transform.rename)

    return frame


def _resolve_spec(spec: ArtifactIngestSpec) -> ArtifactIngestSpec:
    path = Path(spec.path).expanduser().resolve()
    key = spec.key or path.stem
    return ArtifactIngestSpec(
        path=path,
        key=key,
        direction=spec.direction,
        driver=spec.driver,
        artifact_family=spec.artifact_family,
        meta=dict(spec.meta),
    )


def _build_ingest_tags(
    *,
    tags: Optional[Sequence[str]],
    scenario_id: Optional[str],
    seed: Optional[int],
) -> list[str]:
    output = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    output.append("analysis:ingest-artifacts")
    if scenario_id:
        output.append(f"scenario_id:{scenario_id}")
    if seed is not None:
        output.append(f"seed:{int(seed)}")
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in output:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def _build_run_facet(
    *,
    scenario_id: Optional[str],
    seed: Optional[int],
    year: Optional[int],
    iteration: Optional[int],
) -> dict[str, Any]:
    facet: dict[str, Any] = {}
    if scenario_id is not None:
        facet["scenario_id"] = scenario_id
    if seed is not None:
        facet["seed"] = int(seed)
    if year is not None:
        facet["year"] = int(year)
    if iteration is not None:
        facet["iteration"] = int(iteration)
    return facet


@contextmanager
def _start_run_context(
    tracker: Any,
    *,
    run_id: str,
    model: str,
    tags: Sequence[str],
    config: Mapping[str, Any],
    year: Optional[int],
    iteration: Optional[int],
    parent_run_id: Optional[str],
    facet: Optional[Mapping[str, Any]],
):
    kwargs: dict[str, Any] = {
        "config": dict(config),
        "tags": list(tags),
        "year": year,
        "iteration": iteration,
        "parent_run_id": parent_run_id,
    }
    if facet:
        kwargs["facet"] = dict(facet)

    with tracker.start_run(run_id=run_id, model=model, **kwargs):
        yield


def _default_run_id(model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in model)
    safe_model = safe_model.strip("_") or "analysis"
    return f"{safe_model}__{stamp}"


def _table_filename(table_name: str, output_format: str) -> str:
    normalized = output_format.strip().lower()
    if normalized == "csv":
        return f"{table_name}.csv"
    if normalized == "parquet":
        return f"{table_name}.parquet"
    raise ValueError(f"Unsupported output format: {output_format}")


def _write_table(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    output_format: str,
) -> Path:
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = output_format.strip().lower()
    if normalized == "csv":
        frame.to_csv(resolved_path, index=False)
        return resolved_path
    if normalized == "parquet":
        frame.to_parquet(resolved_path, index=False)
        return resolved_path
    raise ValueError(f"Unsupported output format: {output_format}")
