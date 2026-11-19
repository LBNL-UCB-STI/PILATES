from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _load_example_module(relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_hit_example_builds_boundary_overlap_summary():
    module = _load_example_module("examples/consist/cache_hit_inspection.py")
    baseline = pd.DataFrame(
        [
            {
                "scenario_id": "baseline",
                "year": 2030,
                "iteration": 0,
                "model": "urbansim",
                "run_id": "u-1",
            },
            {
                "scenario_id": "baseline",
                "year": 2030,
                "iteration": 0,
                "model": "beam",
                "run_id": "b-1",
            },
        ]
    )
    rerun = pd.DataFrame(
        [
            {
                "scenario_id": "baseline",
                "year": 2030,
                "iteration": 0,
                "model": "urbansim",
                "run_id": "u-2",
            },
        ]
    )

    summary = module.build_boundary_overlap_summary(baseline, rerun)

    urbansim_row = summary.loc[summary["model"] == "urbansim"].iloc[0]
    beam_row = summary.loc[summary["model"] == "beam"].iloc[0]
    assert urbansim_row["status"] == "both"
    assert beam_row["status"] == "baseline_only"


def test_run_comparison_example_builds_mode_share_comparison():
    import ibis

    module = _load_example_module("examples/consist/run_comparison.py")
    calls = {}

    class ArchiveStub:
        def table(self, name, *, facets, where):
            calls["table"] = {"name": name, "facets": facets, "where": where}
            frame = pd.DataFrame(
                [
                    {
                        "pricing_policy": "none",
                        "year": 2030,
                        "iteration": 0,
                        "trip_mode": "DRIVE",
                    },
                    {
                        "pricing_policy": "none",
                        "year": 2030,
                        "iteration": 0,
                        "trip_mode": "DRIVE",
                    },
                    {
                        "pricing_policy": "none",
                        "year": 2030,
                        "iteration": 0,
                        "trip_mode": "WALK",
                    },
                    {
                        "pricing_policy": "cordon",
                        "year": 2030,
                        "iteration": 0,
                        "trip_mode": "DRIVE",
                    },
                    {
                        "pricing_policy": "cordon",
                        "year": 2030,
                        "iteration": 0,
                        "trip_mode": "WALK",
                    },
                    {
                        "pricing_policy": "cordon",
                        "year": 2030,
                        "iteration": 0,
                        "trip_mode": "WALK",
                    },
                ]
            )
            return ibis.memtable(frame)

        def measure(self, table, *, by, measures):
            calls["measure"] = {"table": table, "by": by, "measures": measures}
            aggregations = {name: builder(table) for name, builder in measures.items()}
            return table.group_by(list(by)).agg(**aggregations)

    archive = ArchiveStub()

    result = module.build_mode_share_comparison(
        archive,
        table_name="activitysim.trips",
        compare="pricing_policy",
        baseline="none",
        group_by=["year", "iteration"],
        filters={"year": 2030},
    ).to_pandas()

    cordon_drive = result.loc[
        (result["pricing_policy"] == "cordon") & (result["trip_mode"] == "DRIVE")
    ].iloc[0]
    cordon_walk = result.loc[
        (result["pricing_policy"] == "cordon") & (result["trip_mode"] == "WALK")
    ].iloc[0]

    assert cordon_drive["trip_count"] == 1
    assert cordon_drive["total_trips"] == 3
    assert round(float(cordon_drive["mode_share"]), 6) == 0.333333
    assert round(float(cordon_drive["mode_share_difference"]), 6) == -0.333333
    assert round(float(cordon_walk["mode_share_difference"]), 6) == 0.333333
    assert calls["table"] == {
        "name": "activitysim.trips",
        "facets": ["year", "iteration", "pricing_policy", "trip_mode"],
        "where": {"year": 2030},
    }
    assert calls["measure"]["by"] == [
        "year",
        "iteration",
        "pricing_policy",
        "trip_mode",
    ]
    assert list(calls["measure"]["measures"]) == ["trip_count"]


def test_run_comparison_example_ranks_mode_share_without_baseline():
    import ibis

    module = _load_example_module("examples/consist/run_comparison.py")

    class ArchiveStub:
        def table(self, name, *, facets, where):
            del name, facets, where
            return ibis.memtable(
                pd.DataFrame(
                    [
                        {"year": 2030, "iteration": 0, "trip_mode": "DRIVE"},
                        {"year": 2030, "iteration": 0, "trip_mode": "DRIVE"},
                        {"year": 2030, "iteration": 0, "trip_mode": "WALK"},
                    ]
                )
            )

        def measure(self, table, *, by, measures):
            return table.group_by(list(by)).agg(
                **{name: builder(table) for name, builder in measures.items()}
            )

    result = module.build_mode_share_comparison(
        ArchiveStub(),
        table_name="activitysim.trips",
        compare="trip_mode",
        baseline=None,
        group_by=["year", "iteration"],
        filters={"year": 2030},
    )

    ranked = result.to_pandas().sort_values("mode_share_rank")
    assert list(ranked["trip_mode"]) == ["DRIVE", "WALK"]


def test_restart_replay_example_summarizes_run_outputs():
    module = _load_example_module("examples/consist/restart_replay_inspection.py")
    outputs = {
        "beam_output": SimpleNamespace(
            path=Path("/tmp/beam/output"),
            recovery_roots=["/archive/a", "/archive/b"],
            hash="abc123",
        ),
        "asim_output": SimpleNamespace(
            path=None,
            recovery_roots=[],
            hash=None,
        ),
    }

    frame = module.summarize_run_outputs(outputs)

    assert list(frame["key"]) == ["asim_output", "beam_output"]
    beam_row = frame.loc[frame["key"] == "beam_output"].iloc[0]
    assert beam_row["recovery_root_count"] == 2
    assert beam_row["hash"] == "abc123"
