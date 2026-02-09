from types import SimpleNamespace
from unittest.mock import MagicMock

from pilates.beam.preprocessor import BeamPreprocessor
from pilates.generic.records import FileRecord, RecordStore


def _make_preprocessor():
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        beam=SimpleNamespace(router_directory="router"),
    )
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        forecast_year=2018,
        current_inner_iter=1,
    )
    return BeamPreprocessor("beam", state)


def test_handle_linkstats_accepts_versioned_parquet_warmstart(tmp_path):
    preprocessor = _make_preprocessor()
    workspace = MagicMock()
    workspace.full_path = str(tmp_path)
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")

    parquet_path = tmp_path / "beam_output" / "2.linkstats.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_text("stub")

    ignored_path = tmp_path / "beam_output" / "2.events.PathTraversal.links.parquet"
    ignored_path.write_text("stub")

    previous_records = [
        FileRecord(
            file_path=str(ignored_path),
            short_name="path_traversal_links_2018_0",
        ),
        FileRecord(
            file_path=str(parquet_path),
            short_name="linkstats_parquet_2018_0",
        ),
    ]

    store = RecordStore()
    preprocessor._handle_linkstats(workspace, previous_records, store)

    warmstart = [
        rec for rec in store.all_records() if getattr(rec, "short_name", "") == "linkstats_warmstart"
    ]
    assert len(warmstart) == 1
    assert warmstart[0].file_path.endswith("beam_output/2.linkstats.parquet")

