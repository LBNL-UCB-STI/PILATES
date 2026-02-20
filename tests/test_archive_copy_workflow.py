from pathlib import Path
from types import SimpleNamespace
import logging

import pytest

from pilates.utils import coupler_helpers as ch
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.steps import StepOutputsHolder


class DummyCoupler:
    def __init__(self) -> None:
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def get(self, key, default=None):
        return self.values.get(key, default)

    def update(self, mapping):
        self.values.update(mapping)


class DummyWorkspace:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def full_path(self) -> str:
        return str(self._root)


class ExecutingScenario:
    def run(self, **kwargs):
        fn = kwargs["fn"]
        execution_options = kwargs.get("execution_options")
        runtime_kwargs = kwargs.get("runtime_kwargs") or getattr(
            execution_options, "runtime_kwargs", None
        )
        runtime_kwargs = dict(runtime_kwargs or {})
        fn(**runtime_kwargs)
        return SimpleNamespace(cache_hit=False)


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture(autouse=True)
def _reset_archive_state(monkeypatch):
    ch.stop_archive_worker(timeout=1)
    ch._archive_queue = None
    ch._archive_thread = None
    monkeypatch.delenv("PILATES_ENABLE_ARCHIVE_COPY", raising=False)
    monkeypatch.delenv("PILATES_LOCAL_RUN_DIR", raising=False)
    monkeypatch.delenv("PILATES_ARCHIVE_RUN_DIR", raising=False)
    yield
    ch.stop_archive_worker(timeout=1)
    ch._archive_queue = None
    ch._archive_thread = None


def test_archive_copy_copies_file_and_preserves_relative_path(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "beam" / "output" / "linkstats.csv.gz"
    _write_file(source, "linkstats")

    ch._enqueue_archive_copy("linkstats", str(source))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / "beam" / "output" / "linkstats.csv.gz"
    assert archived.exists()
    assert archived.read_text() == "linkstats"


def test_resolve_existing_path_materializes_local_from_archive(
    monkeypatch, tmp_path, caplog
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    local_path = local_root / "activitysim" / "data" / "households.csv"
    archive_path = archive_root / "activitysim" / "data" / "households.csv"
    _write_file(archive_path, "archived-households")

    with caplog.at_level(logging.INFO):
        resolved = ch.resolve_existing_path(
            str(local_path), materialize_from_archive=True
        )
    assert resolved == str(local_path)
    assert local_path.exists()
    assert local_path.read_text() == "archived-households"
    assert "materializing from archive" in caplog.text
    assert "Materialized local path from archive" in caplog.text


def test_archive_copy_rejects_non_allowlisted_directory(monkeypatch, tmp_path, caplog):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = local_root / "beam" / "output" / "raw_dir"
    _write_file(directory / "file.txt", "data")

    ch._enqueue_archive_copy("beam_output_dir", str(directory))

    assert "not allowlisted" in caplog.text
    assert not (archive_root / "beam" / "output" / "raw_dir").exists()


def test_archive_copy_allows_zarr_directories(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = local_root / "activitysim" / "cache" / "skims.zarr"
    _write_file(directory / "0" / "values", "zarr")

    ch._enqueue_archive_copy("asim_input_skims_zarr_archived", str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / "activitysim" / "cache" / "skims.zarr" / "0" / "values"
    assert archived.exists()
    assert archived.read_text() == "zarr"


def test_log_output_only_enqueues_archive_copy(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ch.cr, "log_output", lambda *args, **kwargs: "artifact")
    monkeypatch.setattr(ch, "_enqueue_archive_copy", lambda key, path: calls.append((key, path)))

    out_path = tmp_path / "out.txt"
    _write_file(out_path, "output")

    ch.log_output_only(
        key="beam_output_plans_xml",
        path=str(out_path),
        description="mock output",
    )

    assert calls == [("beam_output_plans_xml", str(out_path))]


def test_log_and_set_output_enqueues_archive_copy_and_sets_coupler(monkeypatch, tmp_path):
    calls = []
    coupler = DummyCoupler()
    monkeypatch.setattr(ch.cr, "log_output", lambda *args, **kwargs: "artifact")
    monkeypatch.setattr(ch.cr, "current_run", lambda: None)
    monkeypatch.setattr(ch, "_enqueue_archive_copy", lambda key, path: calls.append((key, path)))

    out_path = tmp_path / "out.txt"
    _write_file(out_path, "output")

    ch.log_and_set_output(
        key="linkstats",
        path=str(out_path),
        description="mock output",
        coupler=coupler,
    )

    assert calls == [("linkstats", str(out_path))]
    assert coupler.get("linkstats") is not None


def test_mocked_workflow_archives_logged_outputs(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    workspace = DummyWorkspace(local_root)
    coupler = DummyCoupler()
    outputs_holder = StepOutputsHolder()
    scenario = ExecutingScenario()
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2030, iteration=0)

    def _mock_step(*, workspace, coupler, **_kwargs):
        file_out = Path(workspace.full_path) / "mock" / "linkstats.csv.gz"
        dir_out = Path(workspace.full_path) / "mock" / "skims.zarr"
        _write_file(file_out, "stats")
        _write_file(dir_out / "0" / "values", "zarr")
        ch.log_and_set_output(
            key="linkstats",
            path=str(file_out),
            description="mock linkstats",
            coupler=coupler,
        )
        ch.log_output_only(
            key="asim_input_skims_zarr_archived",
            path=str(dir_out),
            description="mock zarr archive",
        )

    _mock_step.__consist_step__ = SimpleNamespace(model="mock_archive_step", outputs=["linkstats"])

    run_workflow(
        stage_name="mock_archive_stage",
        steps=[StepRef(name="mock_archive_step", step_func=_mock_step)],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix="2030_iter0",
        runtime_kwargs_extra={"coupler": coupler, "outputs_holder": outputs_holder},
    )
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    assert (archive_root / "mock" / "linkstats.csv.gz").exists()
    assert (archive_root / "mock" / "skims.zarr" / "0" / "values").exists()
    assert coupler.get("linkstats") is not None
