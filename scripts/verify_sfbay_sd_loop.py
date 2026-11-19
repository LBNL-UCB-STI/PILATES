#!/usr/bin/env python3
"""Post-run verification and sensitivity analysis for SFBay supply-demand loop.

Verifies:
1. Ingestion of skims and linkstats warmstart with Consist provenance tracking.
2. Sensitivity of ActivitySim outcomes against baseline outputs.
3. Skim drift across BEAM iterations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tables

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def verify_provenance(run_dir: Path, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Verify admission reports and Consist provenance records in run_dir."""
    logger.info("=== 1. Verifying Consist Provenance & Ingestion ===")
    results: Dict[str, Any] = {"admission": {}, "steps": []}

    # Check admission reports
    admission_dir = run_dir / "admission"
    if not admission_dir.is_dir():
        # Check subdirectories if nested
        candidates = list(run_dir.glob("**/admission"))
        if candidates:
            admission_dir = candidates[0]

    if admission_dir.is_dir():
        logger.info("Found admission directory: %s", admission_dir)
        for report_file in sorted(admission_dir.glob("*.json")):
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
                results["admission"][report_file.stem] = data
                verdict = data.get("verdict", data.get("status", "unknown"))
                role = data.get("role", report_file.stem)
                digest = data.get("actual_digest", data.get("identity", "unknown"))
                logger.info(
                    "  Admission [%s]: verdict=%s, digest=%s", role, verdict, digest
                )
            except Exception as e:
                logger.warning(
                    "  Could not read admission report %s: %s", report_file, e
                )
    else:
        logger.warning("Admission directory not found in %s", run_dir)

    # Check duckdb if available
    effective_db = db_path or run_dir / "provenance.duckdb"
    if effective_db.is_file():
        try:
            import duckdb

            con = duckdb.connect(str(effective_db), read_only=True)
            tables_list = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
            logger.info(
                "DuckDB provenance database found: %s (tables: %s)",
                effective_db,
                tables_list,
            )
            if "runs" in tables_list:
                runs = con.execute(
                    "SELECT run_id, scenario, model_name, year, iteration FROM runs"
                ).fetchdf()
                logger.info("Tracked runs in DuckDB:\n%s", runs.to_string(index=False))
                results["runs"] = runs.to_dict(orient="records")
            con.close()
        except Exception as e:
            logger.warning("Could not query DuckDB %s: %s", effective_db, e)

    return results


def compare_activitysim_outcomes(
    baseline_trips_path: Path,
    new_trips_paths: List[Path],
) -> pd.DataFrame:
    """Compare trip mode shares and purposes between baseline and new run iterations."""
    logger.info("=== 2. ActivitySim Outcome Drift Comparison ===")
    if not baseline_trips_path.is_file():
        raise FileNotFoundError(f"Baseline trips file not found: {baseline_trips_path}")

    base_df = pd.read_parquet(baseline_trips_path)
    logger.info(
        "Baseline trips loaded: %d rows from %s", len(base_df), baseline_trips_path
    )

    base_shares = base_df["trip_mode"].value_counts(normalize=True).rename("Baseline")
    shares_df = pd.DataFrame(base_shares)

    for i, trips_path in enumerate(new_trips_paths):
        col_name = f"Iter_{i}"
        if trips_path.is_file():
            iter_df = pd.read_parquet(trips_path)
            logger.info(
                "Iteration %d trips loaded: %d rows from %s",
                i,
                len(iter_df),
                trips_path,
            )
            iter_shares = (
                iter_df["trip_mode"].value_counts(normalize=True).rename(col_name)
            )
            shares_df = shares_df.join(iter_shares, how="outer").fillna(0.0)
            shares_df[f"Delta_{col_name}"] = shares_df[col_name] - shares_df["Baseline"]
        else:
            logger.warning("Trip file for iteration %d not found: %s", i, trips_path)

    shares_df = shares_df.sort_values(by="Baseline", ascending=False)
    # Format as percentages
    formatted = shares_df.applymap(
        lambda x: f"{x * 100:.2f}%" if isinstance(x, float) else x
    )
    logger.info("Mode Share Comparison:\n%s", formatted.to_string())
    return shares_df


def compare_skims(
    initial_skims_path: Path,
    iter_skims_paths: List[Path],
    matrices_to_compare: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare matrix values across initial and subsequent iteration skims."""
    logger.info("=== 3. Skim Drift Across BEAM Iterations ===")
    if not initial_skims_path.is_file():
        raise FileNotFoundError(f"Initial skims file not found: {initial_skims_path}")

    if matrices_to_compare is None:
        matrices_to_compare = [
            "SOV_TIME__AM",
            "SOV_TIME__PM",
            "SOV_TIME__MD",
            "SOV_TIME__EA",
            "SOV_TIME__EV",
            "SOV_DIST__AM",
            "SOV_DIST__PM",
            "WLK_LOC_WLK_TOTIVT__AM",
            "DRV_COM_WLK_TOTIVT__AM",
            "DIST",
        ]

    stats: List[Dict[str, Any]] = []

    with tables.open_file(str(initial_skims_path), mode="r") as f_init:
        available_init = set(f_init.root.data._v_children.keys())
        for mat_name in matrices_to_compare:
            if mat_name not in available_init:
                continue
            arr_init = f_init.root.data[mat_name].read()
            # Replace NaNs or extreme values for stats
            valid_mask = np.isfinite(arr_init) & (arr_init > 0)
            init_mean = (
                float(np.mean(arr_init[valid_mask])) if valid_mask.any() else 0.0
            )
            init_max = float(np.max(arr_init[valid_mask])) if valid_mask.any() else 0.0

            row: Dict[str, Any] = {
                "matrix": mat_name,
                "initial_mean": init_mean,
                "initial_max": init_max,
            }

            for i, skims_path in enumerate(iter_skims_paths):
                if not skims_path.is_file():
                    continue
                with tables.open_file(str(skims_path), mode="r") as f_iter:
                    if mat_name in f_iter.root.data._v_children:
                        arr_iter = f_iter.root.data[mat_name].read()
                        iter_mask = np.isfinite(arr_iter) & (arr_iter > 0)
                        iter_mean = (
                            float(np.mean(arr_iter[iter_mask]))
                            if iter_mask.any()
                            else 0.0
                        )
                        row[f"iter_{i}_mean"] = iter_mean
                        # Delta mean
                        row[f"delta_mean_it{i}"] = iter_mean - init_mean
                        # Mean absolute error across matrix cells
                        common_mask = valid_mask & iter_mask
                        if common_mask.any():
                            mad = float(
                                np.mean(
                                    np.abs(
                                        arr_iter[common_mask] - arr_init[common_mask]
                                    )
                                )
                            )
                            row[f"mad_it{i}"] = mad
            stats.append(row)

    stats_df = pd.DataFrame(stats)
    logger.info("Skim Matrix Comparison Summary:\n%s", stats_df.to_string(index=False))
    return stats_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to PILATES run directory (e.g. /global/scratch/users/$USER/pilates-outputs/consist-sfbay-smoke-sd-loop-2019)",
    )
    parser.add_argument(
        "--baseline-trips",
        type=Path,
        default=Path("pilates/activitysim/output/final_pipeline/trips/final.parquet"),
        help="Path to reference baseline ActivitySim trips parquet",
    )
    parser.add_argument(
        "--baseline-skims",
        type=Path,
        default=Path("pilates/activitysim/data/sfbay/skims.omx"),
        help="Path to initial input skims.omx",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional path to provenance.duckdb",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    logger.info("Analyzing PILATES run directory: %s", run_dir)

    # 1. Provenance Verification
    verify_provenance(run_dir, args.db_path)

    # 2. ActivitySim Mode Share Comparison
    trips_candidates = sorted(run_dir.glob("**/activitysim/**/trips/final.parquet"))
    if not trips_candidates:
        trips_candidates = sorted(run_dir.glob("**/trips/final.parquet"))
    logger.info("Found ActivitySim trips outputs: %s", trips_candidates)
    if trips_candidates and args.baseline_trips.is_file():
        compare_activitysim_outcomes(args.baseline_trips, trips_candidates)
    else:
        logger.warning("Skipping ActivitySim comparison: trips files not located.")

    # 3. Skim Drift Comparison
    skims_candidates = sorted(run_dir.glob("**/final_skims.omx"))
    if not skims_candidates:
        skims_candidates = sorted(run_dir.glob("**/*skims*.omx"))
        # exclude initial baseline if found inside
        skims_candidates = [
            p for p in skims_candidates if p.resolve() != args.baseline_skims.resolve()
        ]
    logger.info("Found BEAM output skims: %s", skims_candidates)
    if skims_candidates and args.baseline_skims.is_file():
        compare_skims(args.baseline_skims, skims_candidates)
    else:
        logger.warning("Skipping skims comparison: output skims files not located.")


if __name__ == "__main__":
    main()
