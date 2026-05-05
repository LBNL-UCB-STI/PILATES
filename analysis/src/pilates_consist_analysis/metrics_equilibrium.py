from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class EquilibriumMetrics:
    pair_count: int
    traveltime_delta_abs_mean: float
    traveltime_delta_abs_p90: float
    volume_delta_abs_mean: float
    volume_delta_abs_p90: float
    traveltime_temperature: float
    volume_temperature: float
    monotone_abs_delta_share: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_count": self.pair_count,
            "traveltime_delta_abs_mean": self.traveltime_delta_abs_mean,
            "traveltime_delta_abs_p90": self.traveltime_delta_abs_p90,
            "volume_delta_abs_mean": self.volume_delta_abs_mean,
            "volume_delta_abs_p90": self.volume_delta_abs_p90,
            "traveltime_temperature": self.traveltime_temperature,
            "volume_temperature": self.volume_temperature,
            "monotone_abs_delta_share": self.monotone_abs_delta_share,
        }


def _to_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")


def compute_equilibrium_metrics(deltas_df: pd.DataFrame) -> EquilibriumMetrics:
    if deltas_df.empty:
        return EquilibriumMetrics(
            pair_count=0,
            traveltime_delta_abs_mean=0.0,
            traveltime_delta_abs_p90=0.0,
            volume_delta_abs_mean=0.0,
            volume_delta_abs_p90=0.0,
            traveltime_temperature=0.0,
            volume_temperature=0.0,
            monotone_abs_delta_share=0.0,
        )

    tt_abs = _to_numeric(deltas_df, "traveltime_delta_abs_mean").dropna()
    vol_abs = _to_numeric(deltas_df, "volume_delta_abs_mean").dropna()
    tt_signed = _to_numeric(deltas_df, "traveltime_delta_mean").dropna()
    vol_signed = _to_numeric(deltas_df, "volume_delta_mean").dropna()

    monotone_share = 0.0
    if not tt_abs.empty:
        df = deltas_df.copy()
        df["_tt_abs"] = _to_numeric(df, "traveltime_delta_abs_mean")
        df["_phys_curr"] = _to_numeric(df, "phys_sim_iteration_curr")
        group_cols = ["year", "iteration", "beam_sub_iteration"]
        existing_group_cols = [column for column in group_cols if column in df.columns]
        if existing_group_cols and "_phys_curr" in df.columns:
            decreasing_flags = []
            for _, group in df.dropna(subset=["_tt_abs", "_phys_curr"]).groupby(existing_group_cols):
                ordered = group.sort_values("_phys_curr")
                prev = ordered["_tt_abs"].shift(1)
                curr = ordered["_tt_abs"]
                valid = prev.notna() & curr.notna()
                if valid.any():
                    decreasing_flags.extend((curr[valid] <= prev[valid]).tolist())
            if decreasing_flags:
                monotone_share = float(sum(bool(v) for v in decreasing_flags)) / float(
                    len(decreasing_flags)
                )

    return EquilibriumMetrics(
        pair_count=int(len(deltas_df)),
        traveltime_delta_abs_mean=float(tt_abs.mean()) if not tt_abs.empty else 0.0,
        traveltime_delta_abs_p90=float(tt_abs.quantile(0.90)) if not tt_abs.empty else 0.0,
        volume_delta_abs_mean=float(vol_abs.mean()) if not vol_abs.empty else 0.0,
        volume_delta_abs_p90=float(vol_abs.quantile(0.90)) if not vol_abs.empty else 0.0,
        traveltime_temperature=float(tt_signed.std(ddof=0)) if not tt_signed.empty else 0.0,
        volume_temperature=float(vol_signed.std(ddof=0)) if not vol_signed.empty else 0.0,
        monotone_abs_delta_share=monotone_share,
    )


def write_equilibrium_metrics(metrics: EquilibriumMetrics, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out
