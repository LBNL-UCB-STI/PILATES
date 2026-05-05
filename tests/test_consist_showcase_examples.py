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
            {"scenario_id": "baseline", "year": 2030, "iteration": 0, "model": "urbansim", "run_id": "u-1"},
            {"scenario_id": "baseline", "year": 2030, "iteration": 0, "model": "beam", "run_id": "b-1"},
        ]
    )
    rerun = pd.DataFrame(
        [
            {"scenario_id": "baseline", "year": 2030, "iteration": 0, "model": "urbansim", "run_id": "u-2"},
        ]
    )

    summary = module.build_boundary_overlap_summary(baseline, rerun)

    urbansim_row = summary.loc[summary["model"] == "urbansim"].iloc[0]
    beam_row = summary.loc[summary["model"] == "beam"].iloc[0]
    assert urbansim_row["status"] == "both"
    assert beam_row["status"] == "baseline_only"


def test_run_comparison_example_builds_selection_from_run_ids():
    module = _load_example_module("examples/consist/run_comparison.py")
    archive = SimpleNamespace(compare=lambda left, right, **kwargs: (left, right, kwargs))

    left, right, kwargs = module.run_comparison(
        archive,
        left_run_ids=["run-a"],
        right_run_ids=["run-b"],
        year=2030,
    )

    assert left == ["run-a"]
    assert right == ["run-b"]
    assert kwargs["year"] == 2030
    assert kwargs["align_on"] == "year"


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
