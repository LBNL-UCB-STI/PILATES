from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class ActivitySimEquilibriumMetrics:
    pair_count: int
    mean_mode_share_tvd: float
    p90_mode_share_tvd: float
    mean_total_trips_delta_abs: float
    p90_total_trips_delta_abs: float
    mean_mode_share_delta_abs: float
    stabilized_pair_share: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_count": self.pair_count,
            "mean_mode_share_tvd": self.mean_mode_share_tvd,
            "p90_mode_share_tvd": self.p90_mode_share_tvd,
            "mean_total_trips_delta_abs": self.mean_total_trips_delta_abs,
            "p90_total_trips_delta_abs": self.p90_total_trips_delta_abs,
            "mean_mode_share_delta_abs": self.mean_mode_share_delta_abs,
            "stabilized_pair_share": self.stabilized_pair_share,
        }


def _to_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")


def compute_activitysim_equilibrium_metrics(
    equilibrium_pairs_df: pd.DataFrame,
    *,
    mode_deltas_df: pd.DataFrame | None = None,
) -> ActivitySimEquilibriumMetrics:
    if equilibrium_pairs_df.empty:
        return ActivitySimEquilibriumMetrics(
            pair_count=0,
            mean_mode_share_tvd=0.0,
            p90_mode_share_tvd=0.0,
            mean_total_trips_delta_abs=0.0,
            p90_total_trips_delta_abs=0.0,
            mean_mode_share_delta_abs=0.0,
            stabilized_pair_share=0.0,
        )

    tvd = _to_numeric(
        equilibrium_pairs_df, "mode_share_total_variation_distance"
    ).dropna()
    trip_delta_abs = _to_numeric(equilibrium_pairs_df, "total_trips_delta_abs").dropna()

    stabilized_share = 0.0
    frame = equilibrium_pairs_df.copy()
    frame["comparison_group"] = frame.get("comparison_group")
    frame["year"] = _to_numeric(frame, "year")
    frame["iteration_prev"] = _to_numeric(frame, "iteration_prev")
    frame["iteration"] = _to_numeric(frame, "iteration")
    frame["tvd"] = _to_numeric(frame, "mode_share_total_variation_distance")
    frame = frame.sort_values(
        ["comparison_group", "year", "iteration_prev", "iteration"],
        na_position="last",
    )
    flags = []
    for _, group in frame.groupby(["comparison_group", "year"], dropna=False):
        group = group.dropna(subset=["tvd"])
        prev_tvd = group["tvd"].shift(1)
        curr_tvd = group["tvd"]
        valid = prev_tvd.notna() & curr_tvd.notna()
        flags.extend((curr_tvd[valid] <= prev_tvd[valid]).tolist())
    if flags:
        stabilized_share = float(sum(bool(v) for v in flags)) / float(len(flags))

    mean_mode_share_delta_abs = 0.0
    if mode_deltas_df is not None and not mode_deltas_df.empty:
        mode_share_delta_abs = _to_numeric(
            mode_deltas_df, "mode_share_delta_abs"
        ).dropna()
        if not mode_share_delta_abs.empty:
            mean_mode_share_delta_abs = float(mode_share_delta_abs.mean())

    return ActivitySimEquilibriumMetrics(
        pair_count=int(len(equilibrium_pairs_df)),
        mean_mode_share_tvd=float(tvd.mean()) if not tvd.empty else 0.0,
        p90_mode_share_tvd=float(tvd.quantile(0.90)) if not tvd.empty else 0.0,
        mean_total_trips_delta_abs=float(trip_delta_abs.mean())
        if not trip_delta_abs.empty
        else 0.0,
        p90_total_trips_delta_abs=float(trip_delta_abs.quantile(0.90))
        if not trip_delta_abs.empty
        else 0.0,
        mean_mode_share_delta_abs=mean_mode_share_delta_abs,
        stabilized_pair_share=stabilized_share,
    )


def write_activitysim_equilibrium_metrics(
    metrics: ActivitySimEquilibriumMetrics, path: str | Path
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return out
