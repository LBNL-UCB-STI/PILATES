from types import SimpleNamespace

from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.beam.postprocessor import BeamPostprocessor


class DummyWorkspace:
    def __init__(self, root):
        self._root = root

    def get_asim_output_dir(self):
        return str(self._root / "activitysim" / "output")

    def get_beam_output_dir(self):
        return str(self._root / "beam" / "output")

    def get_atlas_output_dir(self):
        return str(self._root / "atlas" / "output")

    def get_usim_mutable_data_dir(self):
        return str(self._root / "urbansim" / "data")


def test_activitysim_postprocessor_expected_inputs_skip_missing(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    inputs = ActivitysimPostprocessor.expected_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
    )
    assert inputs["asim_output_dir"] is None


def test_beam_postprocessor_expected_inputs_skip_missing(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    inputs = BeamPostprocessor.expected_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
    )
    assert inputs["beam_output_dir"] is None
    assert inputs["asim_output_dir"] is None


def test_atlas_postprocessor_expected_inputs_skip_missing(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template="usim_{year}.h5"),
        run=SimpleNamespace(region="test"),
    )
    state = SimpleNamespace(forecast_year=2030)
    inputs = AtlasPostprocessor.expected_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    assert inputs["atlas_output_dir"] is None
    assert inputs["usim_datastore_h5"] is None
