from types import SimpleNamespace

from pilates.generic.records import FileRecord, RecordStore
from pilates.utils import coupler_helpers


class DummyCoupler:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value


def test_update_coupler_from_beam_outputs_updates_keys(tmp_path, monkeypatch):
    zarr_path = tmp_path / "skims.zarr"
    omx_path = tmp_path / "final.omx"
    zarr_path.mkdir()
    omx_path.write_text("ok")

    output_store = RecordStore(
        recordList=[
            FileRecord(file_path=str(zarr_path), short_name="zarr_skims"),
            FileRecord(file_path=str(omx_path), short_name="final_skims_omx"),
        ]
    )

    coupler = DummyCoupler()
    typed_coupler = SimpleNamespace()
    workspace = SimpleNamespace(full_path=str(tmp_path))

    monkeypatch.setattr(coupler_helpers.cr, "log_output", lambda *args, **kwargs: None)

    coupler_helpers.update_coupler_from_beam_outputs(
        output_store, coupler, typed_coupler, workspace
    )

    assert coupler.values["zarr_skims"] == str(zarr_path)
    assert coupler.values["final_skims_omx"] == str(omx_path)
    assert not hasattr(typed_coupler, "zarr_skims")
    assert not hasattr(typed_coupler, "final_skims_omx")
