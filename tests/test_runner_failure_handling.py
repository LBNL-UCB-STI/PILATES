import os

from pilates.utils.failure_handling import persist_state_on_error


class _State:
    def __init__(self, path):
        self.path = path
        self.written = False

    def write_state(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("state")
        self.written = True


def test_persist_state_on_error_writes_state(tmp_path, caplog):
    caplog.set_level("ERROR")
    state_path = tmp_path / "state.yaml"
    state = _State(str(state_path))

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        persist_state_on_error(state, "test-step")

    assert state.written is True
    assert os.path.exists(state_path)
