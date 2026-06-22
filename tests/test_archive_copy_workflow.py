from pathlib import Path
from types import SimpleNamespace
import hashlib
import logging
import queue

import pytest

from pilates.utils import coupler_helpers as ch
from pilates.workflows.artifact_keys import ASIM_SHARROW_CACHE_DIR
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
    ch._archive_pending_tasks.clear()
    ch._archive_queued_destinations.clear()
    ch._archive_inflight_signature.clear()
    ch._archive_last_copied_signature.clear()
    ch.recovery_root_adoption.reset_recovery_root_adoption_state()
    monkeypatch.delenv("PILATES_ENABLE_ARCHIVE_COPY", raising=False)
    monkeypatch.delenv("PILATES_LOCAL_RUN_DIR", raising=False)
    monkeypatch.delenv("PILATES_ARCHIVE_RUN_DIR", raising=False)
    yield
    ch.stop_archive_worker(timeout=1)
    ch._archive_queue = None
    ch._archive_thread = None
    ch._archive_pending_tasks.clear()
    ch._archive_queued_destinations.clear()
    ch._archive_inflight_signature.clear()
    ch._archive_last_copied_signature.clear()
    ch.recovery_root_adoption.reset_recovery_root_adoption_state()


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


def test_local_archive_copy_does_not_write_recovery_roots(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    recovery_root_calls = []
    tracker = SimpleNamespace(
        current_consist=None,
        set_artifact_recovery_roots=lambda *args, **kwargs: recovery_root_calls.append(
            (args, kwargs)
        ),
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(ch.cr, "log_output", lambda *args, **kwargs: "artifact")

    source = local_root / "beam" / "output" / "inputs" / "households.csv.gz"
    _write_file(source, "households")
    ch.log_output_only(
        key="beam_input_households_archived",
        path=str(source),
        description="mock BEAM input snapshot",
        facet={
            "artifact_family": "beam_input_archived",
            "source_role": "households_beam_in",
            "snapshot_role": "beam_input_households",
            "snapshot_reason": "exact_rewind",
            "storage_event": "snapshot_copy",
            "year": 2030,
            "iteration": 0,
        },
    )
    ch.flush_archive_queue(timeout=5)

    assert recovery_root_calls == []


@pytest.mark.parametrize(
    "key",
    [
        "usim_input_archive_2030",
        "usim_population_source_h5",
    ],
)
def test_phase2_recovery_root_registration_adopts_only_narrow_h5_families(
    monkeypatch, tmp_path, key
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    registration_calls = []
    tracker = SimpleNamespace(
        register_run_output_recovery_copies=lambda run_id, recovery_root, **kwargs: (
            registration_calls.append((run_id, recovery_root, kwargs))
            or SimpleNamespace(
                registered={key: object()},
                blocked={},
                summary="registered=1",
            )
        ),
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(ch.cr, "current_run", lambda: SimpleNamespace(id="run-123"))
    monkeypatch.setattr(ch.cr, "current_run_id", lambda: "run-123")
    monkeypatch.setattr(
        ch,
        "_find_current_run_output_artifact",
        lambda *, key, path: SimpleNamespace(key=key, container_uri=str(path)),
    )

    source = local_root / "urbansim" / "data" / "model_data_2030.h5"
    _write_file(source, "h5")
    assert ch.archive_copy_now(key=key, path=str(source))

    expected_hash = hashlib.sha256(b"h5").hexdigest()
    assert registration_calls == [
        (
            "run-123",
            str(archive_root),
            {
                "append": True,
                "content_hashes": {key: expected_hash},
                "verify": True,
            },
        )
    ]


def test_phase2_recovery_root_registration_skips_blocked_h5_family(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    registration_calls = []
    tracker = SimpleNamespace(
        register_run_output_recovery_copies=lambda *args, **kwargs: (
            registration_calls.append((args, kwargs))
        )
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(ch.cr, "current_run", lambda: SimpleNamespace(id="run-123"))
    monkeypatch.setattr(ch.cr, "current_run_id", lambda: "run-123")
    monkeypatch.setattr(ch, "_find_current_run_output_artifact", lambda **_kwargs: None)

    source = local_root / "urbansim" / "data" / "model_data_2030.h5"
    _write_file(source, "h5")
    assert ch.archive_copy_now(key="usim_datastore_h5", path=str(source))

    assert registration_calls == []


def test_phase2_recovery_root_registration_prefers_artifact_hash_when_full_hashing(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    registration_calls = []
    tracker = SimpleNamespace(
        identity=SimpleNamespace(hashing_strategy="full"),
        register_run_output_recovery_copies=lambda run_id, recovery_root, **kwargs: (
            registration_calls.append((run_id, recovery_root, kwargs))
            or SimpleNamespace(
                registered={"usim_population_source_h5": object()},
                blocked={},
                summary="registered=1",
            )
        ),
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(ch.cr, "current_run", lambda: SimpleNamespace(id="run-123"))
    monkeypatch.setattr(ch.cr, "current_run_id", lambda: "run-123")

    source = local_root / "urbansim" / "data" / "model_data_2030.h5"
    _write_file(source, "h5")
    artifact = SimpleNamespace(
        key="usim_population_source_h5",
        hash="artifact-hash-123",
        container_uri=str(source),
    )
    monkeypatch.setattr(
        ch,
        "_find_current_run_output_artifact",
        lambda *, key, path: artifact,
    )
    monkeypatch.setattr(
        ch.recovery_root_adoption,
        "_sha256_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sha called")),
    )
    assert ch.archive_copy_now(key="usim_population_source_h5", path=str(source))

    assert registration_calls == [
        (
            "run-123",
            str(archive_root),
            {
                "append": True,
                "content_hashes": {"usim_population_source_h5": "artifact-hash-123"},
                "verify": True,
            },
        )
    ]


def test_resolve_existing_path_does_not_materialize_plain_paths_from_archive(
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
        resolved = ch.resolve_existing_path(str(local_path))
    assert resolved is None
    assert not local_path.exists()
    assert "materializing from archive" not in caplog.text
    assert "Materialized local path from archive" not in caplog.text


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


def test_archive_copy_allows_beam_raw_od_zarr_directories(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = (
        local_root
        / "beam"
        / "beam_output"
        / "sfbay"
        / "year-2017-iteration-0"
        / "ITERS"
        / "it.1"
        / "1.activitySimODSkims_current.zarr"
    )
    _write_file(directory / "0" / "values", "zarr")

    ch._enqueue_archive_copy("raw_od_skims_zarr_2019_0", str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = (
        archive_root
        / "beam"
        / "beam_output"
        / "sfbay"
        / "year-2017-iteration-0"
        / "ITERS"
        / "it.1"
        / "1.activitySimODSkims_current.zarr"
        / "0"
        / "values"
    )
    assert archived.exists()
    assert archived.read_text() == "zarr"


def test_archive_copy_allows_beam_config_reference_snapshot_directory(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = (
        local_root
        / "beam"
        / "beam_output"
        / "inputs-year-2019-iteration-0"
        / "beam_input_config_references_archived"
    )
    _write_file(directory / "scenario" / "network.csv", "network")

    ch._enqueue_archive_copy("beam_input_config_references_archived", str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = (
        archive_root
        / "beam"
        / "beam_output"
        / "inputs-year-2019-iteration-0"
        / "beam_input_config_references_archived"
        / "scenario"
        / "network.csv"
    )
    assert archived.exists()
    assert archived.read_text() == "network"


def test_archive_copy_allows_activitysim_sharrow_cache_directory(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = local_root / "shared_cache" / "numba"
    _write_file(directory / "nested" / "entry.bin", "cache")

    ch._enqueue_archive_copy(ASIM_SHARROW_CACHE_DIR, str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / "shared_cache" / "numba" / "nested" / "entry.bin"
    assert archived.exists()
    assert archived.read_text() == "cache"


@pytest.mark.parametrize(
    "key, relpath",
    [
        ("urbansim_bootstrap_data_root", "urbansim/data/hsize_ct_000.csv"),
        ("beam_mutable_data_dir", "beam/input/test/beam.conf"),
        ("activitysim_bootstrap_data_root", "activitysim/data/households.csv"),
        (
            "activitysim_bootstrap_configs_root",
            "activitysim/configs/configs/settings.yaml",
        ),
    ],
)
def test_archive_copy_allows_bootstrap_runtime_directories(
    monkeypatch, tmp_path, key, relpath
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    path = local_root / relpath
    _write_file(path, "bootstrap")

    directory = path.parent
    if key == "urbansim_bootstrap_data_root":
        directory = local_root / "urbansim" / "data"
    if path.name == "households.csv":
        directory = local_root / "activitysim" / "data"
    if key == "activitysim_bootstrap_configs_root":
        directory = local_root / "activitysim" / "configs"

    ch._enqueue_archive_copy(key, str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / relpath
    assert archived.exists()
    assert archived.read_text() == "bootstrap"


def test_archive_copy_allows_atlas_year_input_directory(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = local_root / "atlas" / "atlas_input" / "year2030"
    _write_file(directory / "vehicles_output.RData", "atlas-rdata")

    ch.enqueue_archive_copy(
        key="atlas_input_year_dir_2030",
        path=str(directory),
    )
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = (
        archive_root / "atlas" / "atlas_input" / "year2030" / "vehicles_output.RData"
    )
    assert archived.exists()
    assert archived.read_text() == "atlas-rdata"


def test_archive_copy_dedupes_same_signature(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "beam" / "output" / "linkstats.csv.gz"
    _write_file(source, "linkstats")

    copy_calls = []
    original_copy2 = ch.shutil.copy2

    def _counting_copy2(src, dst, *args, **kwargs):
        copy_calls.append((src, dst))
        return original_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(ch.shutil, "copy2", _counting_copy2)

    ch._enqueue_archive_copy("linkstats", str(source))
    ch._enqueue_archive_copy("linkstats", str(source))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / "beam" / "output" / "linkstats.csv.gz"
    assert archived.exists()
    assert archived.read_text() == "linkstats"
    assert len(copy_calls) == 1


def test_archive_copy_coalesces_pending_updates_for_same_destination(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    def _ensure_queue_only():
        if ch._archive_queue is None:
            ch._archive_queue = queue.Queue()

    monkeypatch.setattr(ch, "_ensure_archive_worker", _ensure_queue_only)

    source = local_root / ".workflow" / "year_2018_iteration_0.yaml"
    _write_file(source, "manifest-v1")
    ch._enqueue_archive_copy(
        "workflow_manifest",
        str(source),
    )

    _write_file(source, "manifest-v2")
    ch._enqueue_archive_copy(
        "workflow_manifest",
        str(source),
    )

    dest = str(archive_root / ".workflow" / "year_2018_iteration_0.yaml")
    assert ch._archive_queue is not None
    assert ch._archive_queue.qsize() == 1
    assert dest in ch._archive_pending_tasks

    key, pending_src, pending_dest, _is_dir, _signature = ch._archive_pending_tasks[
        dest
    ]
    assert key == "workflow_manifest"
    assert pending_src == str(source)
    assert pending_dest == dest


def test_archive_copy_now_copies_file_and_preserves_relative_path(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / ".workflow" / "year_2018_iteration_0.yaml"
    _write_file(source, "manifest")

    assert ch.archive_copy_now(key="workflow_manifest", path=str(source)) is True

    archived = archive_root / ".workflow" / "year_2018_iteration_0.yaml"
    assert archived.exists()
    assert archived.read_text() == "manifest"


def test_workflow_manifest_is_tracked_as_restart_support_only(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    registration_calls = []
    tracker = SimpleNamespace(
        register_run_output_recovery_copies=lambda *args, **kwargs: (
            registration_calls.append((args, kwargs))
        )
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(ch.cr, "current_run", lambda: SimpleNamespace(id="run-123"))
    monkeypatch.setattr(ch.cr, "current_run_id", lambda: "run-123")

    source = local_root / ".workflow" / "year_2018_iteration_0.yaml"
    _write_file(source, "manifest")
    assert ch.archive_copy_now(key="workflow_manifest", path=str(source)) is True
    assert ch.archive_copy_now(key="workflow_manifest", path=str(source)) is True

    archived = archive_root / ".workflow" / "year_2018_iteration_0.yaml"
    assert archived.exists()
    assert archived.read_text() == "manifest"
    assert registration_calls == []


def test_archive_copy_destination_returns_preserved_relative_path(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "model_data_2021.h5"
    _write_file(source, "h5")

    assert ch.archive_copy_destination(
        key="usim_population_source_h5",
        path=str(source),
    ) == str(archive_root / "urbansim" / "data" / "model_data_2021.h5")


def test_archive_copy_now_force_recopies_matching_signature(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "model_data_2021.h5"
    archived = archive_root / "urbansim" / "data" / "model_data_2021.h5"
    _write_file(source, "fresh")
    _write_file(archived, "stale")

    signature = ch._archive_path_signature(str(source), is_dir=False)
    ch._archive_last_copied_signature[str(archived)] = signature

    assert (
        ch.archive_copy_now(
            key="usim_population_source_h5",
            path=str(source),
            force=True,
        )
        is True
    )
    assert archived.read_text() == "fresh"


def test_flush_archive_queue_can_fail_on_timeout():
    ch._archive_queue = queue.Queue()
    ch._archive_queue.put(("pending",))
    with pytest.raises(TimeoutError, match="Flush timed out"):
        ch.flush_archive_queue(timeout=0.01, fail_on_timeout=True)


def test_log_output_only_enqueues_archive_copy(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ch.cr, "log_output", lambda *args, **kwargs: "artifact")
    monkeypatch.setattr(
        ch, "_enqueue_archive_copy", lambda key, path: calls.append((key, path))
    )

    out_path = tmp_path / "out.txt"
    _write_file(out_path, "output")

    ch.log_output_only(
        key="beam_output_plans_xml",
        path=str(out_path),
        description="mock output",
    )

    assert calls == [("beam_output_plans_xml", str(out_path))]


def test_log_and_set_output_enqueues_archive_copy_and_sets_coupler(
    monkeypatch, tmp_path
):
    calls = []
    coupler = DummyCoupler()
    monkeypatch.setattr(ch.cr, "log_output", lambda *args, **kwargs: "artifact")
    monkeypatch.setattr(ch.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        ch, "_enqueue_archive_copy", lambda key, path: calls.append((key, path))
    )

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

    _mock_step.__consist_step__ = SimpleNamespace(
        model="mock_archive_step", outputs=["linkstats"]
    )

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
