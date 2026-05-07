from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd

from .keys import CANONICAL_KEY_COLUMNS, ensure_canonical_key_columns
from .manifest import DatasetManifest


@dataclass
class LinkstatsDataset:
    artifacts: pd.DataFrame
    summary: pd.DataFrame
    deltas: pd.DataFrame


def _load_pilates_linkstats_module() -> Any:
    try:
        from pilates.utils import consist_analysis as ca
    except Exception as exc:
        raise RuntimeError(
            "Linkstats dataset builders require PILATES module "
            "`pilates.utils.consist_analysis` to be importable."
        ) from exc
    return ca


def discover_linkstats_artifacts(
    tracker: Any,
    *,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    artifact_family: str = "linkstats_unmodified_phys_sim_iter_parquet",
    namespace: str = "beam",
    limit: int = 10000,
) -> pd.DataFrame:
    ca = _load_pilates_linkstats_module()
    artifacts = ca.find_linkstats_artifacts(
        tracker,
        year=year,
        iteration=iteration,
        artifact_family=artifact_family,
        namespace=namespace,
        limit=limit,
    )
    if artifacts.empty:
        return artifacts
    return ca.assign_effective_beam_sub_iteration(artifacts)


def build_linkstats_dataset(
    tracker: Any,
    *,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    artifact_family: str = "linkstats_unmodified_phys_sim_iter_parquet",
    namespace: str = "beam",
    grouped_mode: str = "hybrid",
    grouped_missing_files: str = "warn",
    grouped_schema_id: Optional[str] = None,
    traveltime_weighting: Literal["unweighted", "volume_weighted"] = "unweighted",
    limit: int = 10000,
) -> LinkstatsDataset:
    ca = _load_pilates_linkstats_module()
    artifacts_df = discover_linkstats_artifacts(
        tracker,
        year=year,
        iteration=iteration,
        artifact_family=artifact_family,
        namespace=namespace,
        limit=limit,
    )
    if artifacts_df.empty:
        return LinkstatsDataset(
            artifacts=artifacts_df, summary=pd.DataFrame(), deltas=pd.DataFrame()
        )

    summarize_kwargs: Dict[str, Any] = {
        "tracker": tracker,
        "namespace": namespace,
        "grouped_mode": grouped_mode,
        "grouped_missing_files": grouped_missing_files,
        "traveltime_weighting": traveltime_weighting,
    }
    if grouped_schema_id:
        summarize_kwargs["grouped_schema_id"] = grouped_schema_id

    summary_df = ca.summarize_linkstats_artifacts(artifacts_df, **summarize_kwargs)
    if summary_df.empty:
        return LinkstatsDataset(
            artifacts=artifacts_df, summary=summary_df, deltas=pd.DataFrame()
        )
    deltas_df = ca.summarize_linkstats_deltas(summary_df, tracker=tracker)
    return LinkstatsDataset(
        artifacts=artifacts_df, summary=summary_df, deltas=deltas_df
    )


def write_linkstats_dataset(
    dataset: LinkstatsDataset,
    *,
    output_dir: str | Path,
    archive_run_dir: str,
    db_path: str,
    query: Dict[str, Any],
) -> DatasetManifest:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    artifacts_path = out / "linkstats_artifacts.csv"
    summary_path = out / "linkstats_summary.csv"
    deltas_path = out / "linkstats_deltas.csv"

    dataset.artifacts.to_csv(artifacts_path, index=False)
    ensure_canonical_key_columns(dataset.summary).to_csv(summary_path, index=False)
    ensure_canonical_key_columns(dataset.deltas).to_csv(deltas_path, index=False)

    manifest = DatasetManifest(
        dataset_name="beam_linkstats_dataset",
        archive_run_dir=str(archive_run_dir),
        db_path=str(db_path),
        query=query,
        files={
            "artifacts": str(artifacts_path),
            "summary": str(summary_path),
            "deltas": str(deltas_path),
        },
        row_counts={
            "artifacts": int(len(dataset.artifacts)),
            "summary": int(len(dataset.summary)),
            "deltas": int(len(dataset.deltas)),
        },
        key_columns=list(CANONICAL_KEY_COLUMNS),
    )
    manifest.write_json(out / "dataset_manifest.json")
    return manifest
