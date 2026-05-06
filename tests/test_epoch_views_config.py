from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pytest

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from pilates_consist_analysis.epoch_views import (
    EpochViews,
    _family_spec,
    resolve_artifact_families,
)
from pilates_consist_analysis.epochs import SimulationEpoch


epoch_views_module = importlib.import_module("pilates_consist_analysis.epoch_views")


@dataclass
class _RunStub:
    id: str


class _TrackerStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_grouped_view(self, **kwargs):
        self.calls.append(kwargs)


def _epoch_for_beam() -> SimulationEpoch:
    return SimulationEpoch(
        year=2030,
        outer_iteration=1,
        scenario_id="scenario-a",
        runs={"beam": _RunStub(id="beam-run-1")},
    )


def test_default_mapping_resolves_expected_entries():
    resolved = resolve_artifact_families()

    linkstats = _family_spec(
        "beam",
        "linkstats",
        artifact_families=resolved,
    )

    assert linkstats["artifact_family"] == "linkstats_unmodified_phys_sim_iter_parquet"
    assert resolved["activitysim"]["trips"]["concept_key"] == "trips_asim_out"


def test_mapping_dict_override_merges_override_and_new_logical_name():
    resolved = resolve_artifact_families(
        artifact_families={
            "beam": {
                "linkstats": {"artifact_family": "linkstats_custom"},
                "events": {
                    "artifact_family": "events_parquet",
                    "concept_key": "events",
                },
            }
        }
    )

    assert resolved["beam"]["linkstats"]["artifact_family"] == "linkstats_custom"
    assert resolved["beam"]["linkstats"]["concept_key"] == "linkstats"
    assert resolved["beam"]["events"]["artifact_family"] == "events_parquet"


def test_mapping_json_file_override_loads_and_merges(tmp_path):
    override_path = tmp_path / "artifact_family_override.json"
    override_path.write_text(
        json.dumps(
            {
                "activitysim": {
                    "tours": {
                        "artifact_family": "tours",
                        "concept_key": "tours_asim_out",
                    }
                },
                "beam": {
                    "linkstats": {
                        "artifact_family": "linkstats_from_json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_artifact_families(artifact_families_json_path=override_path)

    assert resolved["activitysim"]["tours"]["concept_key"] == "tours_asim_out"
    assert resolved["beam"]["linkstats"]["artifact_family"] == "linkstats_from_json"
    assert resolved["activitysim"]["persons"]["artifact_family"] == "persons"


def test_mapping_env_var_path_fallback_works(monkeypatch, tmp_path):
    override_path = tmp_path / "artifact_family_env_override.json"
    override_path.write_text(
        json.dumps(
            {
                "beam": {
                    "linkstats": {
                        "artifact_family": "linkstats_from_env",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(epoch_views_module.ARTIFACT_FAMILIES_ENV_VAR, str(override_path))

    resolved = resolve_artifact_families()

    assert resolved["beam"]["linkstats"]["artifact_family"] == "linkstats_from_env"


def test_epoch_views_uses_overridden_family_spec(monkeypatch):
    monkeypatch.setattr(
        epoch_views_module, "_resolve_grouped_hybrid_creator", lambda _t: None
    )

    captured_schema_args: dict[str, Any] = {}

    def _fake_resolve_schema_id(self, *, model: str, run_id: str, artifact_family: str):
        captured_schema_args.update(
            {"model": model, "run_id": run_id, "artifact_family": artifact_family}
        )
        return "schema-1"

    monkeypatch.setattr(EpochViews, "_resolve_schema_id", _fake_resolve_schema_id)

    tracker = _TrackerStub()
    views = EpochViews(
        epoch=_epoch_for_beam(),
        tracker=tracker,
        artifact_families={
            "beam": {
                "linkstats": {
                    "artifact_family": "linkstats_override_for_view",
                }
            }
        },
    )

    created_view = views._view("beam", "linkstats")

    assert created_view == tracker.calls[0]["view_name"]
    assert captured_schema_args["artifact_family"] == "linkstats_override_for_view"
    assert tracker.calls[0]["params"] == [
        "beam.artifact_family=linkstats_override_for_view"
    ]


def test_missing_model_or_logical_mapping_raises_clear_attribute_error():
    resolved = resolve_artifact_families()

    with pytest.raises(
        AttributeError,
        match="Model 'not_a_model' is not configured in artifact families mapping",
    ):
        _family_spec("not_a_model", "anything", artifact_families=resolved)

    with pytest.raises(
        AttributeError,
        match="Logical artifact 'not_a_logical_name' is not configured for model 'beam'",
    ):
        _family_spec("beam", "not_a_logical_name", artifact_families=resolved)


def test_schema_resolution_failure_raises_contextual_runtime_error(monkeypatch):
    monkeypatch.setattr(EpochViews, "_resolve_schema_id", lambda self, **_kwargs: None)

    views = EpochViews(epoch=_epoch_for_beam(), tracker=_TrackerStub())

    with pytest.raises(RuntimeError) as excinfo:
        views._view("beam", "linkstats")

    message = str(excinfo.value)
    assert "Could not resolve schema_id for epoch-scoped view creation" in message
    assert "model=beam" in message
    assert "logical_name=linkstats" in message
    assert "run_id=beam-run-1" in message
    assert "artifact_family=linkstats_unmodified_phys_sim_iter_parquet" in message
