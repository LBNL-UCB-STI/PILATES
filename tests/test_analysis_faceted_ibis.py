from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from pilates_consist_analysis import delta, delta_change, difference, rank


def test_faceted_helpers_execute_on_real_ibis_tables():
    import ibis

    measured = ibis.memtable(
        pd.DataFrame(
            [
                {
                    "scenario_id": "baseline",
                    "pricing_policy": "none",
                    "year": 2020,
                    "iteration": 0,
                    "trips": 10,
                },
                {
                    "scenario_id": "baseline",
                    "pricing_policy": "none",
                    "year": 2030,
                    "iteration": 1,
                    "trips": 14,
                },
                {
                    "scenario_id": "policy-a",
                    "pricing_policy": "cordon",
                    "year": 2030,
                    "iteration": 1,
                    "trips": 20,
                },
                {
                    "scenario_id": "policy-b",
                    "pricing_policy": "mileage",
                    "year": 2030,
                    "iteration": 1,
                    "trips": 18,
                },
            ]
        )
    )

    year_delta = delta(
        measured,
        value="trips",
        over="year",
        by=["scenario_id", "pricing_policy"],
    ).to_pandas()
    baseline = year_delta.loc[
        (year_delta["scenario_id"] == "baseline") & (year_delta["year"] == 2030)
    ].iloc[0]
    assert int(baseline["trips_delta"]) == 4

    sweep = difference(
        measured,
        value="trips",
        compare="pricing_policy",
        baseline="none",
        at={"year": 2030},
    ).to_pandas()
    cordon = sweep.loc[sweep["pricing_policy"] == "cordon"].iloc[0]
    mileage = sweep.loc[sweep["pricing_policy"] == "mileage"].iloc[0]
    assert int(cordon["trips_difference"]) == 6
    assert int(mileage["trips_difference"]) == 4

    ranked = rank(
        measured,
        value="trips",
        by=["year"],
        descending=True,
    ).to_pandas()
    ordered = ranked.loc[ranked["year"] == 2030].sort_values("trips_rank")
    assert list(ordered["pricing_policy"]) == ["cordon", "mileage", "none"]


def test_delta_change_executes_on_real_ibis_table():
    import ibis

    measured = ibis.memtable(
        pd.DataFrame(
            [
                {"scenario_id": "baseline", "year": 2030, "iteration": 0, "trips": 100},
                {"scenario_id": "baseline", "year": 2030, "iteration": 1, "trips": 130},
                {"scenario_id": "baseline", "year": 2030, "iteration": 2, "trips": 145},
            ]
        )
    )

    changed = delta_change(
        measured,
        value="trips",
        over="iteration",
        by=["scenario_id", "year"],
    ).to_pandas()
    row = changed.loc[changed["iteration"] == 2].iloc[0]
    assert int(row["trips_delta"]) == 15
    assert int(row["trips_delta_change"]) == -15
