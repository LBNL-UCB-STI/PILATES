from types import SimpleNamespace

from pilates.activitysim.inputs import build_activitysim_inputs


class DummyWorkspace:
    def __init__(self, root):
        self.full_path = str(root)
        self._root = root

    def get_asim_mutable_data_dir(self):
        return str(self._root / "activitysim" / "data")

    def get_usim_mutable_data_dir(self):
        return str(self._root / "urbansim" / "data")

    def get_asim_output_dir(self):
        return str(self._root / "activitysim" / "output")


def test_build_activitysim_inputs_merges_coupler_and_usim(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)

    coupler = {"zarr_skims": "skims.zarr"}
    usim_inputs = {"usim_datastore_h5": "/tmp/usim.h5"}

    inputs, descriptions = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        year=2018,
        iteration=0,
        coupler=coupler,
        usim_inputs=usim_inputs,
    )

    assert inputs["asim_mutable_data_dir"] == str(asim_dir)
    assert inputs["usim_datastore_h5"] == "/tmp/usim.h5"
    assert inputs["zarr_skims"] == "skims.zarr"
    assert "asim_mutable_data_dir" in descriptions
