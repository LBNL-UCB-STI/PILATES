import os
import sys
import types
from types import SimpleNamespace

if "openmatrix" not in sys.modules:
    openmatrix_stub = types.ModuleType("openmatrix")
    openmatrix_stub.File = object
    sys.modules["openmatrix"] = openmatrix_stub

if "geopandas" not in sys.modules:
    geopandas_stub = types.ModuleType("geopandas")
    geopandas_stub.GeoDataFrame = object
    sys.modules["geopandas"] = geopandas_stub

from pilates.beam.runner import BeamRunner
from pilates.generic.records import FileRecord, RecordStore
from pilates.urbansim.runner import UrbansimRunner
from pilates.utils.coupler_helpers import update_coupler_from_beam_outputs


class _StubState:
    def __init__(self, forecast_year=2017, start_year=True):
        self.forecast_year = forecast_year
        self._start_year = start_year

    def is_start_year(self):
        return self._start_year


class _StubWorkspace:
    def __init__(self, root):
        self._root = root

    def get_usim_mutable_data_dir(self):
        return os.path.join(self._root, "usim")

    def get_asim_output_dir(self):
        return os.path.join(self._root, "asim")

    def get_beam_mutable_data_dir(self):
        return os.path.join(self._root, "beam")


def _urbansim_settings():
    return SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            input_file_template="input_{region_id}.h5",
            output_file_template="output_{year}.h5",
            region_mappings={"region_to_region_id": {"test": "123"}},
        ),
    )


def test_urbansim_expected_inputs_resolve_existing_input(tmp_path):
    settings = _urbansim_settings()
    state = _StubState(start_year=True, forecast_year=2017)
    workspace = _StubWorkspace(str(tmp_path))
    os.makedirs(workspace.get_usim_mutable_data_dir(), exist_ok=True)

    input_name = settings.urbansim.input_file_template.format(region_id="123")
    input_path = os.path.join(workspace.get_usim_mutable_data_dir(), input_name)
    with open(input_path, "w") as handle:
        handle.write("stub")

    inputs = UrbansimRunner.expected_inputs(settings, state, workspace)

    assert inputs["usim_mutable_data_dir"] == workspace.get_usim_mutable_data_dir()
    assert inputs["usim_datastore_h5"] == input_path


def test_beam_expected_inputs_resolve_optional_zarr(tmp_path):
    settings = SimpleNamespace(activitysim=SimpleNamespace())
    state = _StubState()
    workspace = _StubWorkspace(str(tmp_path))
    os.makedirs(os.path.join(workspace.get_asim_output_dir(), "cache"), exist_ok=True)

    inputs = BeamRunner.expected_inputs(settings, state, workspace)
    assert inputs["beam_mutable_data_dir"] == workspace.get_beam_mutable_data_dir()
    assert inputs["zarr_skims"] is None

    zarr_path = os.path.join(workspace.get_asim_output_dir(), "cache", "skims.zarr")
    with open(zarr_path, "w") as handle:
        handle.write("stub")

    inputs = BeamRunner.expected_inputs(settings, state, workspace)
    assert inputs["zarr_skims"] == zarr_path


def test_update_coupler_from_beam_outputs_sets_outputs(monkeypatch, tmp_path):
    zarr_path = tmp_path / "skims.zarr"
    omx_path = tmp_path / "final_skims.omx"
    zarr_path.write_text("stub")
    omx_path.write_text("stub")

    records = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(zarr_path),
                short_name="zarr_skims",
                description="zarr",
            ),
            FileRecord(
                file_path=str(omx_path),
                short_name="final_skims_omx",
                description="omx",
            ),
        ]
    )

    class _Coupler:
        def __init__(self):
            self._store = {}

        def set(self, key, value):
            self._store[key] = value

    coupler = _Coupler()

    logged = {}

    def _log_output(path, key=None, description=None):
        logged[key] = path
        return f"artifact:{key}"

    monkeypatch.setattr("pilates.utils.coupler_helpers.cr.log_output", _log_output)

    update_coupler_from_beam_outputs(
        records, coupler, workspace=SimpleNamespace(full_path=str(tmp_path))
    )

    assert coupler._store["zarr_skims"] == "artifact:zarr_skims"
    assert coupler._store["final_skims_omx"] == "artifact:final_skims_omx"
