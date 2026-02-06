from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pilates.utils.provenance_report import build_provenance_report, write_provenance_report


class _FakeTracker:
    def __init__(self) -> None:
        self._run = SimpleNamespace(
            id="scenario-1",
            status="completed",
            model_name="orchestrator",
            meta={
                "steps": [
                    {"id": "step-a", "name": "activitysim_preprocess"},
                    {"id": "step-b", "name": "activitysim_run"},
                ]
            },
        )
        self._artifacts = {
            "step-a": SimpleNamespace(
                inputs={"usim_datastore_h5": SimpleNamespace(key="usim_datastore_h5", path="/tmp/usim.h5")},
                outputs={"asim_households_in": SimpleNamespace(key="asim_households_in", path="/tmp/households.csv")},
            ),
            "step-b": SimpleNamespace(
                inputs={"asim_households_in": SimpleNamespace(key="asim_households_in", path="/tmp/households.csv")},
                outputs={"zarr_skims": SimpleNamespace(key="zarr_skims", path="/tmp/skims.zarr")},
            ),
        }

    def get_run(self, run_id: str):
        if run_id == self._run.id:
            return self._run
        return None

    def get_artifacts_for_run(self, run_id: str):
        return self._artifacts[run_id]


def test_build_provenance_report_contains_chain_and_dag():
    tracker = _FakeTracker()
    report = build_provenance_report(tracker, "scenario-1")
    assert "# Provenance Report" in report
    assert "activitysim_preprocess" in report
    assert "activitysim_run" in report
    assert "`zarr_skims`" in report
    assert "```mermaid" in report


def test_write_provenance_report_writes_markdown(tmp_path: Path):
    tracker = _FakeTracker()
    output = tmp_path / "report.md"
    report = write_provenance_report(tracker, "scenario-1", output)
    assert output.exists()
    assert output.read_text(encoding="utf-8") == report
