import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Optional dependency stubs (for lightweight test environments)
# ---------------------------------------------------------------------------
import sys
import types


def _stub_module(module_name: str, attrs=None):
    mod = sys.modules.get(module_name)
    if mod is None:
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod
    if attrs:
        for key, value in attrs.items():
            setattr(mod, key, value)
    return mod


try:
    import openlineage  # noqa: F401
except ModuleNotFoundError:
    openlineage_mod = _stub_module("openlineage")
    openlineage_client_mod = _stub_module(
        "openlineage.client",
        attrs={"set_producer": lambda *a, **k: None, "OpenLineageClient": object},
    )
    setattr(openlineage_mod, "client", openlineage_client_mod)
    _stub_module(
        "openlineage.client.facet",
        attrs={
            "SchemaField": object,
            "SchemaDatasetFacet": object,
            "DocumentationJobFacet": object,
            "SourceCodeLocationJobFacet": object,
        },
    )
    _stub_module(
        "openlineage.client.run",
        attrs={
            "Dataset": object,
            "InputDataset": object,
            "OutputDataset": object,
            "RunEvent": object,
            "Run": object,
            "Job": object,
            "RunState": object,
        },
    )
    _stub_module("openlineage.client.transport")
    _stub_module("openlineage.client.transport.http", attrs={"HttpTransport": object, "HttpConfig": object})
    _stub_module("openlineage.client.transport.file", attrs={"FileTransport": object, "FileConfig": object})
    _stub_module(
        "openlineage.client.transport.composite",
        attrs={"CompositeTransport": object, "CompositeConfig": object},
    )

from pilates.beam.preprocessor import BeamPreprocessor
from pilates.generic.records import FileRecord, RecordStore


@pytest.fixture
def simple_settings(tmp_path):
    return SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        beam=SimpleNamespace(router_directory="r5/router"),
    )


@pytest.fixture
def simple_workspace(tmp_path):
    workspace = MagicMock()
    workspace.full_path = str(tmp_path)
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")
    return workspace


def _make_preprocessor(settings, *, forecast_year=2020, inner_iter=0):
    state = SimpleNamespace(
        full_settings=settings,
        forecast_year=forecast_year,
        current_inner_iter=inner_iter,
        year=forecast_year,
    )
    tracker = MagicMock()
    return BeamPreprocessor(model_name="beam", state=state, provenance_tracker=tracker)


def test_first_beam_iteration_uses_initial_linkstats(simple_settings, simple_workspace, tmp_path):
    pre = _make_preprocessor(simple_settings, forecast_year=2020, inner_iter=0)
    store = RecordStore()

    init_path = (
        tmp_path
        / "beam"
        / "input"
        / "seattle"
        / "r5"
        / "router"
        / "init.linkstats.csv.gz"
    )
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text("dummy")

    pre._handle_linkstats(simple_workspace, previous_beam_records=[], store=store)

    rec = next((r for r in store.all_records() if r.short_name == "linkstats_warmstart"), None)
    assert rec is not None
    assert isinstance(rec, FileRecord)
    assert rec.short_name == "linkstats_warmstart"
    assert rec.file_path == os.path.relpath(str(init_path), str(tmp_path))
    assert rec.description and "source=initial_inputs" in rec.description


def test_subsequent_beam_iteration_uses_previous_beam_output_linkstats(
    simple_settings, simple_workspace, tmp_path
):
    pre = _make_preprocessor(simple_settings, forecast_year=2020, inner_iter=1)
    store = RecordStore()

    init_path = (
        tmp_path
        / "beam"
        / "input"
        / "seattle"
        / "r5"
        / "router"
        / "init.linkstats.csv.gz"
    )
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text("init")

    prev_path = (
        tmp_path
        / "beam"
        / "beam_output"
        / "seattle"
        / "year-2020-iteration-0"
        / "ITERS"
        / "it.2"
        / "2.linkstats.csv.gz"
    )
    prev_path.parent.mkdir(parents=True, exist_ok=True)
    prev_path.write_text("prev")

    misleading_unversioned = FileRecord(
        file_path=os.path.relpath(str(init_path), str(tmp_path)),
        short_name="linkstats",
        models=["beam"],
    )
    prev_final = FileRecord(
        file_path=os.path.relpath(str(prev_path), str(tmp_path)),
        short_name="linkstats_2020_0",
        models=["beam"],
    )

    pre._handle_linkstats(simple_workspace, previous_beam_records=[misleading_unversioned, prev_final], store=store)

    rec = next((r for r in store.all_records() if r.short_name == "linkstats_warmstart"), None)
    assert rec is not None
    assert rec.file_path == os.path.relpath(str(prev_path), str(tmp_path))
    assert rec.description and "source=previous_beam_output" in rec.description


def test_unversioned_previous_linkstats_logs_not_ideal(caplog, simple_settings, simple_workspace, tmp_path):
    pre = _make_preprocessor(simple_settings, forecast_year=2020, inner_iter=1)
    store = RecordStore()

    init_path = (
        tmp_path
        / "beam"
        / "input"
        / "seattle"
        / "r5"
        / "router"
        / "init.linkstats.csv.gz"
    )
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text("init")

    unversioned = FileRecord(
        file_path=os.path.relpath(str(init_path), str(tmp_path)),
        short_name="linkstats",
        models=["beam"],
    )

    with caplog.at_level("WARNING"):
        pre._handle_linkstats(simple_workspace, previous_beam_records=[unversioned], store=store)

    assert any("[NOT IDEAL]" in r.message and "unversioned" in r.message for r in caplog.records)
