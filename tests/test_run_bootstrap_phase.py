from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.generic.records import FileRecord, RecordStore
from pilates.runtime import launcher as run_module
from pilates.utils import consist_db_snapshot as snapshot_module
from pilates.workflows import binding as bootstrap_runtime
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
)
from pilates.workflows.state import WorkflowState


class DummyWorkspace:
    def __init__(self, full_path="/tmp/bootstrap", settings=None):
        self.full_path = full_path
        self.settings = settings
        self.input_data = {}
        self.output_data = {}

    def get_usim_mutable_data_dir(self):
        return os.path.join(self.full_path, "urbansim", "data")

    def get_asim_mutable_data_dir(self):
        return os.path.join(self.full_path, "activitysim", "data")

    def get_asim_mutable_configs_dir(self):
        return os.path.join(self.full_path, "activitysim", "configs")

    def get_asim_output_dir(self):
        return os.path.join(self.full_path, "activitysim", "output")

    def get_atlas_mutable_input_dir(self):
        return os.path.join(self.full_path, "atlas", "atlas_input")

    def get_beam_mutable_data_dir(self):
        return os.path.join(self.full_path, "beam", "input")


class DummySnapshotTracker:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def snapshot_db(self, dest_path: str, checkpoint: bool = True):
        self.calls.append({"dest_path": dest_path, "checkpoint": checkpoint})
        if self.fail:
            raise RuntimeError("simulated snapshot failure")
        output_path = Path(dest_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("db", encoding="utf-8")
        metadata = {
            "run_id": "run-test",
            "snapshot_ts_utc": "2026-02-24T00:00:00Z",
        }
        (
            output_path.parent
            / snapshot_module.snapshot_meta_filename(output_path.name)
        ).write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )


class TraceCapableTrackerStub:
    def trace(self, **_kwargs):
        return nullcontext()

    def find_matching_run(self, **_kwargs):
        return None

    def register_run_output_recovery_copies(self, *_args, **_kwargs):
        return None

    def log_h5_container(
        self,
        path,
        key=None,
        direction="output",
        *,
        container_recovery_unit=None,
        child_recovery_policy=None,
        representation_policy=None,
        **_meta,
    ):
        return (
            path,
            key,
            direction,
            container_recovery_unit,
            child_recovery_policy,
            representation_policy,
        )


def _settings(code_identity=None, *, local_data_input_folder=None):
    return SimpleNamespace(
        run=SimpleNamespace(
            consist_code_identity=code_identity,
            region="test",
            start_year=2017,
            models=SimpleNamespace(
                land_use="urbansim",
                activity_demand="activitysim",
                vehicle_ownership="atlas",
                traffic_assignment="beam",
            ),
        ),
        atlas=SimpleNamespace(scenario="baseline"),
        activitysim=SimpleNamespace(main_configs_dir="configs"),
        beam=SimpleNamespace(config="beam.conf"),
        urbansim=SimpleNamespace(
            region_mappings={"region_to_region_id": {"test": "000"}},
            input_file_template="usim_{region_id}.h5",
            input_file_template_year=None,
            output_file_template="usim_{year}.h5",
            local_data_input_folder=local_data_input_folder,
        ),
    )


def _state():
    return SimpleNamespace(
        start_year=2017,
        current_year=2017,
        forecast_year=2017,
    )


class _WorkspaceStub:
    def __init__(
        self,
        _settings,
        local_root: str,
        folder_name: str,
        *,
        archive_run_dir: str | None = None,
    ):
        self.full_path = os.path.join(local_root, folder_name)
        self.archive_run_dir = archive_run_dir
        os.makedirs(self.full_path, exist_ok=True)

    def get_atlas_mutable_input_dir(self):
        return os.path.join(self.full_path, "atlas", "atlas_input")


def test_seed_bootstrap_artifacts_to_coupler_publishes_beam_defaults(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plans = scenario_dir / "plans.parquet"
    households = scenario_dir / "households.parquet"
    persons = scenario_dir / "persons.parquet"
    plans.write_text("plans", encoding="utf-8")
    households.write_text("households", encoding="utf-8")
    persons.write_text("persons", encoding="utf-8")
    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None
    assert isinstance(coupler.get("plans_beam_in"), str)
    assert isinstance(coupler.get("households_beam_in"), str)
    assert isinstance(coupler.get("persons_beam_in"), str)


def test_seed_bootstrap_artifacts_to_coupler_publishes_initial_urbansim_roles(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    workspace = DummyWorkspace(full_path=str(tmp_path))
    staged_h5 = Path(workspace.get_usim_mutable_data_dir()) / "usim_000.h5"
    staged_h5.parent.mkdir(parents=True, exist_ok=True)
    staged_h5.write_text("bootstrap datastore", encoding="utf-8")
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=_settings(
            local_data_input_folder=workspace.get_usim_mutable_data_dir()
        ),
        state=_state(),
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("usim_datastore_base_h5") == str(staged_h5)
    assert coupler.get("usim_datastore_h5") == str(staged_h5)


def test_seed_bootstrap_artifacts_to_coupler_preserves_current_urbansim_role(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {"usim_datastore_h5": "/existing/current.h5"}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    workspace = DummyWorkspace(full_path=str(tmp_path))
    staged_h5 = Path(workspace.get_usim_mutable_data_dir()) / "usim_000.h5"
    staged_h5.parent.mkdir(parents=True, exist_ok=True)
    staged_h5.write_text("bootstrap datastore", encoding="utf-8")
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=_settings(
            local_data_input_folder=workspace.get_usim_mutable_data_dir()
        ),
        state=_state(),
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("usim_datastore_base_h5") == str(staged_h5)
    assert coupler.get("usim_datastore_h5") == "/existing/current.h5"


@pytest.mark.parametrize(
    ("state", "expected_current"),
    [
        (
            SimpleNamespace(
                start_year=2017,
                current_year=2023,
                forecast_year=2028,
            ),
            None,
        ),
        (
            SimpleNamespace(
                start_year=2017,
                current_year=2017,
                forecast_year=2017,
            ),
            "bootstrap datastore",
        ),
    ],
    ids=["later_year", "restart_at_initial_year"],
)
def test_seed_bootstrap_artifacts_to_coupler_publishes_current_urbansim_role_only_at_initial_frontier(
    tmp_path,
    state,
    expected_current,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    workspace = DummyWorkspace(full_path=str(tmp_path))
    staged_h5 = Path(workspace.get_usim_mutable_data_dir()) / "usim_000.h5"
    staged_h5.parent.mkdir(parents=True, exist_ok=True)
    staged_h5.write_text("bootstrap datastore", encoding="utf-8")
    coupler = DummyCoupler()
    coupler.set("usim_datastore_base_h5", str(staged_h5))

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=_settings(
            local_data_input_folder=workspace.get_usim_mutable_data_dir()
        ),
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("usim_datastore_base_h5") == str(staged_h5)
    current = coupler.get("usim_datastore_h5")
    if expected_current is None:
        assert current is None
    else:
        assert Path(current).read_text(encoding="utf-8") == expected_current


def test_seed_bootstrap_artifacts_to_coupler_skips_urbansim_roles_without_consumer(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    workspace = DummyWorkspace(full_path=str(tmp_path))
    staged_h5 = Path(workspace.get_usim_mutable_data_dir()) / "usim_000.h5"
    staged_h5.parent.mkdir(parents=True, exist_ok=True)
    staged_h5.write_text("bootstrap datastore", encoding="utf-8")
    settings = _settings(local_data_input_folder=workspace.get_usim_mutable_data_dir())
    settings.run.models = SimpleNamespace(
        land_use=None,
        activity_demand=None,
        vehicle_ownership=None,
        traffic_assignment=None,
    )
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=_state(),
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("usim_datastore_base_h5") is None
    assert coupler.get("usim_datastore_h5") is None


def test_seed_bootstrap_artifacts_to_coupler_falls_back_to_config_exchange_folder(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    default_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    default_dir.mkdir(parents=True, exist_ok=True)
    config_dir = default_dir / "2018"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (config_dir / name).write_text(name, encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None


def test_seed_bootstrap_artifacts_to_coupler_prefers_yaml_scenario_folder_when_present(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    default_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    default_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (default_dir / name).write_text(f"default-{name}", encoding="utf-8")

    config_dir = default_dir / "2018"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (config_dir / name).write_text(f"config-{name}", encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") == str(default_dir / "plans.parquet")
    assert coupler.get("households_beam_in") == str(default_dir / "households.parquet")
    assert coupler.get("persons_beam_in") == str(default_dir / "persons.parquet")


@pytest.mark.parametrize("extension", ["csv", "csv.gz"])
def test_seed_bootstrap_artifacts_to_coupler_falls_back_to_csv_formats_when_parquet_missing(
    tmp_path, extension
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim" / "2018"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("plans", "households", "persons"):
        (scenario_dir / f"{stem}.{extension}").write_text(stem, encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None
    assert isinstance(coupler.get("plans_beam_in"), str)
    assert isinstance(coupler.get("households_beam_in"), str)
    assert isinstance(coupler.get("persons_beam_in"), str)


def test_seed_bootstrap_artifacts_to_coupler_publishes_initial_warmstart(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    router_dir = tmp_path / "beam" / "input" / "sfbay" / "r5" / "network"
    router_dir.mkdir(parents=True, exist_ok=True)
    warmstart = router_dir / "init.linkstats.csv.gz"
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="r5/network/init.linkstats.csv.gz",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") is not None
    assert isinstance(coupler.get("linkstats_warmstart"), str)


def test_seed_bootstrap_artifacts_to_coupler_uses_configured_warmstart_path(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    warmstart = tmp_path / "beam" / "input" / "sfbay" / "custom" / "warmstart.parquet"
    warmstart.parent.mkdir(parents=True, exist_ok=True)
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="custom/warmstart.parquet",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") == str(warmstart)


def test_seed_bootstrap_artifacts_to_coupler_expands_router_directory_in_warmstart_path(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    warmstart = (
        tmp_path
        / "beam"
        / "input"
        / "sfbay"
        / "r5"
        / "network"
        / "init.linkstats.parquet"
    )
    warmstart.parent.mkdir(parents=True, exist_ok=True)
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="{router_directory}/init.linkstats.parquet",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") == str(warmstart)


def test_seed_bootstrap_artifacts_to_coupler_does_not_probe_router_dir_when_warmstart_unset(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    router_dir = tmp_path / "beam" / "input" / "sfbay" / "r5" / "network"
    router_dir.mkdir(parents=True, exist_ok=True)
    (router_dir / "init.linkstats.csv.gz").write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") is None


def test_seed_bootstrap_artifacts_publishes_beam_only_exchange_and_warmstart(
    tmp_path,
    monkeypatch,
):
    class Coupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    exchange_paths = {
        BEAM_PLANS_IN: tmp_path / "plans.parquet",
        BEAM_HOUSEHOLDS_IN: tmp_path / "households.parquet",
        BEAM_PERSONS_IN: tmp_path / "persons.parquet",
    }
    for path in exchange_paths.values():
        path.write_text(path.stem, encoding="utf-8")

    warmstart_path = tmp_path / "beam" / "input" / "test" / "warmstart.parquet"
    warmstart_path.parent.mkdir(parents=True)
    warmstart_path.write_text("linkstats", encoding="utf-8")

    monkeypatch.setattr(
        "pilates.workflows.binding.beam_exchange.register_existing_beam_exchange_inputs",
        lambda **_kwargs: RecordStore(
            recordList=[
                FileRecord(file_path=str(path), short_name=key)
                for key, path in exchange_paths.items()
            ]
        ),
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="test",
            models=SimpleNamespace(traffic_assignment="beam", activity_demand=None),
        ),
        beam=SimpleNamespace(warmstart_linkstats_path="warmstart.parquet"),
    )
    coupler = Coupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=SimpleNamespace(),
        workspace=DummyWorkspace(full_path=str(tmp_path)),
        coupler=coupler,
    )

    assert coupler.values == {
        **{key: str(path) for key, path in exchange_paths.items()},
        LINKSTATS_WARMSTART: str(warmstart_path),
    }


def test_seed_bootstrap_artifacts_publishes_warmstart_with_activitysim(
    tmp_path,
):
    class Coupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    warmstart_path = tmp_path / "beam" / "input" / "test" / "warmstart.parquet"
    warmstart_path.parent.mkdir(parents=True)
    warmstart_path.write_text("linkstats", encoding="utf-8")
    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="test",
            models=SimpleNamespace(
                traffic_assignment="beam",
                activity_demand="activitysim",
            ),
        ),
        beam=SimpleNamespace(warmstart_linkstats_path="warmstart.parquet"),
    )
    coupler = Coupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=SimpleNamespace(),
        workspace=DummyWorkspace(full_path=str(tmp_path)),
        coupler=coupler,
    )

    assert coupler.values == {LINKSTATS_WARMSTART: str(warmstart_path)}


def test_seed_bootstrap_artifacts_to_coupler_does_not_publish_activitysim_runtime_caches(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    zarr_dir = tmp_path / "activitysim" / "output" / "cache" / "skims.zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / "zarr.json").write_text("{}", encoding="utf-8")

    sharrow_cache = tmp_path / "shared_cache" / "numba"
    sharrow_cache.mkdir(parents=True, exist_ok=True)
    (sharrow_cache / "cache.bin").write_text("cache", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand="activitysim", travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("zarr_skims") is None


def test_format_restart_command_uses_config_and_archive_state():
    settings = SimpleNamespace(settings_file="scenarios/settings-seattle.yaml")

    command = run_module._format_restart_command(
        settings=settings,
        archive_state_path="/tmp/pilates run/run_state.yaml",
    )

    assert (
        command
        == "python run.py -c scenarios/settings-seattle.yaml -S '/tmp/pilates run/run_state.yaml'"
    )


def test_format_hpc_restart_command_requires_account_placeholder():
    settings = SimpleNamespace(settings_file="scenarios/settings-seattle.yaml")

    command = run_module._format_hpc_restart_command(
        settings=settings,
        archive_state_path="/tmp/pilates run/run_state.yaml",
    )

    assert (
        command
        == "./hpc/job_runner.sh -c scenarios/settings-seattle.yaml -a '<slurm_account>' -s '/tmp/pilates run/run_state.yaml'"
    )


def test_resolve_run_storage_roots_expands_env_vars(monkeypatch, tmp_path):
    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    monkeypatch.setenv("PILATES_TEST_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("PILATES_TEST_LOCAL_ROOT", str(local_root))

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory="$PILATES_TEST_ARCHIVE_ROOT",
            local_workspace_root="$PILATES_TEST_LOCAL_ROOT",
        )
    )

    resolved_archive_root, resolved_local_root = run_module._resolve_run_storage_roots(
        settings
    )

    assert resolved_archive_root == str(archive_root.resolve())
    assert resolved_local_root == str(local_root.resolve())


def test_resolve_run_storage_roots_defaults_local_to_archive_root(tmp_path):
    archive_root = tmp_path / "archive-root"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=None,
        )
    )

    resolved_archive_root, resolved_local_root = run_module._resolve_run_storage_roots(
        settings
    )

    assert resolved_archive_root == str(archive_root.resolve())
    assert resolved_local_root == str(archive_root.resolve())


def test_prepare_run_context_resolves_storage_tracker_and_state_paths(
    tmp_path, monkeypatch
):
    class SnapshotStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class TrackerStub:
        def __init__(self):
            self.trace_calls = []

        def trace(self, **kwargs):
            self.trace_calls.append(kwargs)
            return nullcontext()

        def find_matching_run(self, **_kwargs):
            return None

        def register_run_output_recovery_copies(self, *_args, **_kwargs):
            return None

        def log_h5_container(
            self,
            path,
            key=None,
            direction="output",
            *,
            container_recovery_unit=None,
            child_recovery_policy=None,
            representation_policy=None,
            **_meta,
        ):
            return (
                path,
                key,
                direction,
                container_recovery_unit,
                child_recovery_policy,
                representation_policy,
            )

    class StateStub:
        def __init__(self):
            self.run_info_path = None
            self.file_loc = None
            self.mirror_file_loc = None

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="seattle",
            output_directory=str(tmp_path / "archive-root"),
            local_workspace_root=str(tmp_path / "local-root"),
            enable_archive_copy=False,
            output_run_name="prepare-context-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file="scenarios/settings-seattle.yaml",
    )
    state = StateStub()

    tracker = TrackerStub()
    create_tracker_kwargs = {}

    def _create_tracker(**kwargs):
        create_tracker_kwargs.update(kwargs)
        return tracker

    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: 7)
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", _create_tracker)
    monkeypatch.setattr(
        run_module,
        "ConsistDbSnapshotManager",
        lambda **kwargs: SnapshotStub(**kwargs),
    )
    monkeypatch.setattr(run_module, "Workspace", _WorkspaceStub)

    prepared = run_module._prepare_run_context(settings=settings, state=state)

    assert prepared.settings is settings
    assert prepared.state is state
    assert prepared.tracker is tracker
    assert prepared.cache_epoch == 7
    assert prepared.is_restart_run is False
    assert prepared.run_name.startswith("pilates-run--seattle--prepare-context-test--")
    assert prepared.archive_run_dir.startswith(str(tmp_path / "archive-root"))
    assert prepared.local_run_dir.startswith(str(tmp_path / "local-root"))
    assert prepared.workspace.archive_run_dir == prepared.archive_run_dir
    assert state.file_loc == prepared.archive_state_path
    assert state.mirror_file_loc == prepared.local_state_path
    assert state.run_info_path == prepared.archive_state_path
    assert create_tracker_kwargs["run_dir"] == prepared.archive_run_dir
    assert create_tracker_kwargs["mounts"]["workspace"] == prepared.local_run_dir
    assert create_tracker_kwargs["mounts"]["beam_input"] == os.path.join(
        prepared.local_run_dir,
        "beam",
        "input",
    )
    assert create_tracker_kwargs["archive_mounts"] == {
        "workspace": ".",
        "beam_input": "beam/input",
    }
    assert tracker.trace_calls == [
        {
            "name": "workspace_setup",
            "model": "pilates_orchestrator",
            "tags": [f"scenario_id:{prepared.scenario_id}"],
        }
    ]


def test_restart_rewinds_uncommitted_activitysim_to_upstream_provider_boundary() -> (
    None
):
    class _StageValue(str):
        @property
        def name(self) -> str:
            return str(self)

    class _Stage:
        land_use = _StageValue("land_use")
        vehicle_ownership_model = _StageValue("vehicle_ownership_model")
        supply_demand_loop = _StageValue("supply_demand_loop")
        activity_demand = _StageValue("activity_demand")
        traffic_assignment = _StageValue("traffic_assignment")

    class _State:
        Stage = _Stage
        is_restart_run = True
        current_major_stage = _Stage.supply_demand_loop
        current_sub_stage = _Stage.activity_demand
        current_inner_iter = 0
        sub_stage_progress = "runner"
        write_count = 0
        major_stage_order = (
            _Stage.land_use,
            _Stage.vehicle_ownership_model,
            _Stage.supply_demand_loop,
        )

        def is_enabled(self, stage):
            return stage in {
                self.Stage.land_use,
                self.Stage.vehicle_ownership_model,
                self.Stage.activity_demand,
                self.Stage.traffic_assignment,
            }

        @property
        def iteration(self):
            return self.current_inner_iter

        @iteration.setter
        def iteration(self, value):
            self.current_inner_iter = value

        def should_run(self, major_stage, iteration=0, sub_stage=None):
            return (
                major_stage == self.Stage.supply_demand_loop
                and iteration == self.current_inner_iter
                and sub_stage == self.Stage.activity_demand
            )

        def write_state(self):
            self.write_count += 1

    state = _State()

    assert run_module._rewind_uncommitted_activitysim_restart_to_provider_boundary(
        state=state
    )
    assert state.current_major_stage == state.Stage.land_use
    assert state.current_sub_stage is None
    assert state.current_inner_iter == 0
    assert state.sub_stage_progress is None
    assert state.write_count == 1


def test_main_logs_restart_instructions_on_failure(tmp_path, monkeypatch, caplog):
    class SnapshotStub:
        def final_snapshot(self):
            return True

    class StateStub:
        def __init__(self):
            self.run_info_path = None
            self.file_loc = None
            self.mirror_file_loc = None

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="seattle",
            output_directory=str(tmp_path / "archive-root"),
            local_workspace_root=str(tmp_path / "local-root"),
            enable_archive_copy=False,
            output_run_name="restart-hint-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file="scenarios/settings-seattle.yaml",
    )
    state = StateStub()

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: 1)
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(
        run_module.cr, "create_tracker", lambda **_kwargs: TraceCapableTrackerStub()
    )
    monkeypatch.setattr(
        run_module, "ConsistDbSnapshotManager", lambda **_kwargs: SnapshotStub()
    )
    monkeypatch.setattr(run_module, "Workspace", _WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    def _fail_contract(**_kwargs):
        raise RuntimeError("simulated pre-scenario failure")

    monkeypatch.setattr(run_module, "_build_scenario_runtime_contract", _fail_contract)

    run_module._RUN_FAILURE_CONTEXT.clear()
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="simulated pre-scenario failure"):
            run_module.main()
        run_module._log_restart_instructions_on_failure()

    assert "Run failed. Restart command:" in caplog.text
    assert "python run.py -c scenarios/settings-seattle.yaml -S " in caplog.text
    assert (
        "./hpc/job_runner.sh -c scenarios/settings-seattle.yaml -a '<slurm_account>' -s "
        in caplog.text
    )
    assert "run_state.yaml" in caplog.text


def test_pre_scenario_failure_emits_registered_failure_hooks():
    class FailureSink:
        def __init__(self):
            self.calls = []

        def on_run_failed(self, run, error):
            self.calls.append((run, error))

    notifier = FailureSink()
    publisher = FailureSink()
    prepared = SimpleNamespace(
        run_name="pilates-run--seattle--base",
        run_notifier=notifier,
        run_publisher=publisher,
    )
    error = RuntimeError("contract failed")

    run_module._emit_pre_scenario_failure(prepared, error)

    assert notifier.calls[0][1] is error
    assert publisher.calls[0][1] is error
    failure_run = notifier.calls[0][0]
    assert failure_run.id == "pilates-run--seattle--base"
    assert failure_run.model_name == "pilates_orchestrator"
    assert failure_run.status == "failed"
    assert "scenario_header" in failure_run.tags
    assert failure_run.meta["launcher_failure"] is True


def test_resume_rewind_guardrail_blocks_by_default_and_allows_override(tmp_path):
    archive_state_path = tmp_path / "archive-run" / "run_state.yaml"
    archive_state_path.parent.mkdir(parents=True, exist_ok=True)
    archive_state_path.write_text(
        "year: 2022\nstage: null\niteration: 0\nasim_compiled: false\n",
        encoding="utf-8",
    )
    state = SimpleNamespace(current_year=2019)

    with pytest.raises(RuntimeError, match="Refusing rewind resume"):
        run_module._enforce_resume_rewind_guardrail(
            state=state,
            archive_state_path=str(archive_state_path),
            allow_rewind_resume=False,
        )

    run_module._enforce_resume_rewind_guardrail(
        state=state,
        archive_state_path=str(archive_state_path),
        allow_rewind_resume=True,
    )


def test_main_enables_external_paths_for_archive_to_local_tracker_topology(
    tmp_path, monkeypatch
):
    class StopAfterTrackerInit(RuntimeError):
        pass

    class StateStub:
        def __init__(self):
            self.run_info_path = None
            self.file_loc = None
            self.mirror_file_loc = None

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

    created_workspaces = []

    class WorkspaceStub(_WorkspaceStub):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_workspaces.append(self)

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="seattle",
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="tracker-topology-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file=None,
    )
    state = StateStub()
    tracker_kwargs = {}

    def _capture_create_tracker(**kwargs):
        tracker_kwargs.update(kwargs)
        return TraceCapableTrackerStub()

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch"
    )
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", _capture_create_tracker)
    monkeypatch.setattr(
        run_module,
        "ConsistDbSnapshotManager",
        lambda **_kwargs: SimpleNamespace(final_snapshot=lambda: True),
    )
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    def _stop_after_tracker_init(**_kwargs):
        raise StopAfterTrackerInit("stop after tracker init")

    monkeypatch.setattr(
        run_module, "_build_scenario_runtime_contract", _stop_after_tracker_init
    )

    with pytest.raises(StopAfterTrackerInit, match="stop after tracker init"):
        run_module.main()

    archive_run_dir = Path(tracker_kwargs["run_dir"])
    repo_root = Path(__file__).resolve().parents[1]
    inputs_mount = Path(tracker_kwargs["mounts"]["inputs"])
    workspace_mount = Path(tracker_kwargs["mounts"]["workspace"])
    beam_input_mount = Path(tracker_kwargs["mounts"]["beam_input"])
    project_root = Path(tracker_kwargs["project_root"])
    assert archive_run_dir.parent == archive_root
    assert archive_run_dir.name.startswith(
        f"pilates-run--{settings.run.region}--{settings.run.output_run_name}--"
    )
    assert inputs_mount == repo_root.resolve()
    assert workspace_mount.parent == local_root
    assert workspace_mount.name == archive_run_dir.name
    assert beam_input_mount == workspace_mount / "beam" / "input"
    assert tracker_kwargs["archive_mounts"] == {
        "workspace": ".",
        "beam_input": "beam/input",
    }
    assert project_root == repo_root.resolve()
    assert tracker_kwargs["allow_external_paths"] is True
    assert created_workspaces[-1].full_path == str(workspace_mount)
    assert created_workspaces[-1].archive_run_dir == str(archive_run_dir)


def test_main_restart_fails_when_atlas_archive_tracker_is_unavailable(
    tmp_path, monkeypatch
):
    class SnapshotStub:
        def final_snapshot(self):
            return True

    class StateStub:
        Stage = WorkflowState.Stage

        def __init__(self, run_info_path: str):
            self.run_info_path = run_info_path
            self.is_restart_run = True
            self.file_loc = None
            self.mirror_file_loc = None
            self.current_major_stage = WorkflowState.Stage.vehicle_ownership_model
            self.current_year = 2023
            self.start_year = 2017
            self.forecast_year = 2023

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    run_name = "restart-strict-atlas-repair"
    archive_run_dir = archive_root / run_name
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = archive_run_dir / "run_state.yaml"
    run_state_path.write_text("state", encoding="utf-8")
    year2017 = archive_run_dir / "atlas" / "atlas_input" / "year2017"
    year2017.mkdir(parents=True, exist_ok=True)
    for filename in (
        "households.csv",
        "blocks.csv",
        "persons.csv",
        "residential.csv",
        "jobs.csv",
    ):
        (year2017 / filename).write_text("seed\n", encoding="utf-8")

    year2021 = archive_run_dir / "atlas" / "atlas_input" / "year2021"
    year2021.mkdir(parents=True, exist_ok=True)
    for filename in (
        "households.csv",
        "blocks.csv",
        "persons.csv",
        "grave.csv",
        "residential.csv",
        "jobs.csv",
    ):
        (year2021 / filename).write_text("prior\n", encoding="utf-8")
    (year2021 / "vehicles_output.RData").write_text("vehicles\n", encoding="utf-8")
    (year2021 / "households_output.RData").write_text("households\n", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused-on-restart",
            restart_strict=True,
            models=SimpleNamespace(vehicle_ownership="atlas"),
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file=None,
        allow_rewind_resume=False,
    )
    state = StateStub(str(archive_run_dir / "missing-continuation" / "run_state.yaml"))

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch"
    )
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(
        run_module.cr, "create_tracker", lambda **_kwargs: TraceCapableTrackerStub()
    )
    monkeypatch.setattr(
        run_module, "ConsistDbSnapshotManager", lambda **_kwargs: SnapshotStub()
    )
    monkeypatch.setattr(run_module, "Workspace", _WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)
    monkeypatch.setattr(
        run_module,
        "_build_scenario_runtime_contract",
        lambda **_kwargs: {
            "scenario_kwargs": {},
            "schema_steps_all": (),
            "schema_steps_enabled": (),
            "coupler_schema": {},
            "required_output_keys": (),
        },
    )
    monkeypatch.setattr(
        run_module.cr,
        "scenario",
        lambda *_args, **_kwargs: _OuterScenarioStub([]),
    )
    monkeypatch.setattr(
        run_module,
        "seed_bootstrap_artifacts_to_coupler",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run_module.restart_runtime,
        "restore_restart_coupler_snapshot",
        lambda **_kwargs: {},
    )
    with pytest.raises(
        RuntimeError,
        match="ATLAS restart requires an archive Consist tracker",
    ):
        run_module.main()

    local_run_dir = local_root / run_name
    assert not (local_run_dir / "atlas" / "atlas_input" / "year2017").exists()
    assert not (local_run_dir / "atlas" / "atlas_input" / "year2021").exists()


def test_resolve_consist_db_paths_uses_local_run_dir_by_default():
    settings = SimpleNamespace(
        run=SimpleNamespace(),
        shared=SimpleNamespace(
            database=SimpleNamespace(
                enabled=True, path="/global/scratch/provenance.duckdb"
            )
        ),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path == "/local/job123/run-abc/.consist/provenance.duckdb"
    assert archive_path == "/global/scratch/run-abc/.consist/provenance.duckdb"


def test_resolve_consist_db_paths_uses_configured_path_when_local_disabled():
    settings = SimpleNamespace(
        run=SimpleNamespace(consist_db_local_run=False),
        shared=SimpleNamespace(
            database=SimpleNamespace(
                enabled=True, path="/global/scratch/provenance.duckdb"
            )
        ),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path == "/global/scratch/provenance.duckdb"
    assert archive_path == "/global/scratch/provenance.duckdb"


def test_resolve_consist_db_paths_returns_none_when_db_disabled():
    settings = SimpleNamespace(
        run=SimpleNamespace(),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path="ignored")),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path is None
    assert archive_path is None


def test_mirror_consist_db_to_archive_copies_db_and_wal(tmp_path):
    local_db = tmp_path / "local" / "provenance.duckdb"
    local_db.parent.mkdir(parents=True, exist_ok=True)
    local_db.write_text("db", encoding="utf-8")
    local_wal = tmp_path / "local" / "provenance.duckdb.wal"
    local_wal.write_text("wal", encoding="utf-8")

    archive_db = tmp_path / "archive" / "provenance.duckdb"
    snapshot_module.mirror_consist_db_to_archive(str(local_db), str(archive_db))

    assert archive_db.read_text(encoding="utf-8") == "db"
    assert (tmp_path / "archive" / "provenance.duckdb.wal").read_text(
        encoding="utf-8"
    ) == "wal"


def test_restore_local_consist_db_from_snapshot_hydrates_missing_local_db(tmp_path):
    archive_run_dir = tmp_path / "archive-run"
    latest_dir = snapshot_module.snapshot_latest_dir(str(archive_run_dir))
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "provenance.duckdb").write_text("db", encoding="utf-8")
    (latest_dir / "provenance.duckdb.wal").write_text("wal", encoding="utf-8")
    (
        latest_dir / snapshot_module.snapshot_meta_filename("provenance.duckdb")
    ).write_text(
        json.dumps({"snapshot_ts_utc": "2026-02-24T01:02:03Z"}),
        encoding="utf-8",
    )

    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_restore_on_start=True,
            consist_db_restore_strict=False,
        )
    )

    restored = snapshot_module.restore_local_consist_db_from_snapshot(
        settings=settings,
        local_db_path=str(local_db),
        archive_run_dir=str(archive_run_dir),
    )

    assert restored is True
    assert local_db.read_text(encoding="utf-8") == "db"
    assert local_db.with_suffix(".duckdb.wal").read_text(encoding="utf-8") == "wal"
    assert (
        local_db.parent / snapshot_module.snapshot_meta_filename(local_db.name)
    ).exists()


def test_seed_local_consist_db_from_shared_hydrates_missing_local_db(tmp_path):
    shared_db = tmp_path / "shared" / "provenance.duckdb"
    shared_db.parent.mkdir(parents=True, exist_ok=True)
    shared_db.write_text("db", encoding="utf-8")
    shared_wal = shared_db.with_suffix(".duckdb.wal")
    shared_wal.write_text("wal", encoding="utf-8")

    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_seed_from_shared_on_start=True,
            consist_db_seed_strict=False,
        )
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(shared_db),
    )

    assert seeded is True
    assert local_db.read_text(encoding="utf-8") == "db"
    assert local_db.with_suffix(".duckdb.wal").read_text(encoding="utf-8") == "wal"


def test_seed_local_consist_db_from_shared_disabled_returns_false(tmp_path):
    shared_db = tmp_path / "shared" / "provenance.duckdb"
    shared_db.parent.mkdir(parents=True, exist_ok=True)
    shared_db.write_text("db", encoding="utf-8")
    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(consist_db_seed_from_shared_on_start=False)
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(shared_db),
    )

    assert seeded is False
    assert not local_db.exists()


def test_seed_local_consist_db_from_shared_missing_source_returns_false(tmp_path):
    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_seed_from_shared_on_start=True,
            consist_db_seed_strict=False,
        )
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(tmp_path / "shared" / "missing.duckdb"),
    )

    assert seeded is False
    assert not local_db.exists()


def test_snapshot_manager_triggers_outer_iteration_snapshots(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    local_db_path = tmp_path / "local" / ".consist" / "provenance.duckdb"
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(local_db_path),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    did_snapshot = manager.on_outer_iteration_boundary(year=2018, iteration=0)

    assert did_snapshot is True
    assert len(tracker.calls) == 1
    latest_db = (
        snapshot_module.snapshot_latest_dir(str(tmp_path / "archive-run"))
        / "provenance.duckdb"
    )
    assert latest_db.exists()


def test_activitysim_boundary_forces_named_snapshot() -> None:
    calls: list[dict[str, object]] = []
    snapshot_manager = SimpleNamespace(
        snapshot=lambda **kwargs: calls.append(kwargs) or True
    )

    assert run_module._snapshot_activitysim_boundary(
        snapshot_manager,
        year=2018,
        iteration=2,
    )
    assert calls == [
        {
            "reason": "after_activitysim_y2018_i2",
            "checkpoint": True,
        }
    ]


def test_snapshot_manager_interval_snapshot_safe_point_behavior(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=3600,
            consist_db_snapshot_on_outer_iteration=False,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    first = manager.maybe_snapshot_interval(reason="safe_point_1")
    second = manager.maybe_snapshot_interval(reason="safe_point_2")

    assert first is True
    assert second is False
    assert len(tracker.calls) == 1


def test_snapshot_manager_final_snapshot(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    did_snapshot = manager.final_snapshot()

    assert did_snapshot is True
    assert len(tracker.calls) == 1
    assert "finalize" in Path(tracker.calls[0]["dest_path"]).parent.name


def test_snapshot_failures_are_non_fatal_and_logged(tmp_path, caplog):
    tracker = DummySnapshotTracker(fail=True)
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    with caplog.at_level("WARNING"):
        did_snapshot = manager.on_outer_iteration_boundary(year=2018, iteration=0)

    assert did_snapshot is False
    assert "snapshot failed" in caplog.text.lower()


def test_snapshot_retention_keeps_expected_number_of_snapshots(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=0,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=2,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    manager.snapshot(reason="snap_a", checkpoint=True)
    manager.snapshot(reason="snap_b", checkpoint=True)
    manager.snapshot(reason="snap_c", checkpoint=True)

    history_dir = snapshot_module.snapshot_history_dir(str(tmp_path / "archive-run"))
    history_entries = [path for path in history_dir.iterdir() if path.is_dir()]
    assert len(history_entries) == 2


class _OuterScenarioStub:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.coupler = SimpleNamespace(
            declare_outputs=lambda *_args, **_kwargs: events.append("declare_outputs")
        )

    def __enter__(self):
        self._events.append("scenario_enter")
        return self

    def __exit__(self, *_args):
        self._events.append("scenario_exit")
        return False

    def capture_coupler_snapshot(self, **_kwargs):
        return {}


class _OuterAdmissionState:
    def __init__(self, *, run_models: bool) -> None:
        self._run_models = run_models

    def __iter__(self):
        return iter((2020,) if self._run_models else ())

    def should_run(self, stage) -> bool:
        return self._run_models and stage == WorkflowState.Stage.land_use

    def complete_step(self, _stage) -> None:
        return None


def _prepared_outer_admission_run(
    tmp_path: Path,
    *,
    is_restart_run: bool = False,
    run_models: bool = False,
):
    settings = SimpleNamespace()
    state = _OuterAdmissionState(
        run_models=run_models,
    )
    workspace = SimpleNamespace(full_path=str(tmp_path / "workspace"))
    tracker = SimpleNamespace()
    return SimpleNamespace(
        settings=settings,
        state=state,
        surface=SimpleNamespace(),
        workspace=workspace,
        runtime_context=SimpleNamespace(),
        tracker=tracker,
        snapshot_manager=SimpleNamespace(final_snapshot=lambda: True),
        run_name="outer-admission-test",
        scenario_id="test",
        seed=None,
        cache_epoch=1,
        is_restart_run=is_restart_run,
        archive_run_dir=str(tmp_path / "archive-run"),
        local_run_dir=str(tmp_path / "local-run"),
        archive_state_path=str(tmp_path / "archive-run" / "run_state.yaml"),
        local_state_path=str(tmp_path / "local-run" / "run_state.yaml"),
        local_consist_db_path=None,
        archive_consist_db_path=None,
        run_notifier=None,
        run_publisher=None,
    )


def test_main_admits_bootstrap_datastore_inside_outer_run_before_seeding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing outer-run hook would seed bootstrap outputs before admission."""
    events: list[str] = []
    prepared = _prepared_outer_admission_run(tmp_path)
    existing_reports = {"beam_linkstats_warmstart": {"outcome": "verified"}}
    active_outer_run = SimpleNamespace(meta={"admission_reports": existing_reports})
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_module, "_prepare_run_context", lambda **_kwargs: prepared)
    monkeypatch.setattr(
        run_module,
        "_build_scenario_runtime_contract",
        lambda **_kwargs: {
            "scenario_kwargs": {},
            "schema_steps_all": (),
            "schema_steps_enabled": (),
            "coupler_schema": {},
            "required_output_keys": (),
        },
    )
    monkeypatch.setattr(
        run_module.cr,
        "scenario",
        lambda *_args, **_kwargs: _OuterScenarioStub(events),
    )
    monkeypatch.setattr(run_module.cr, "current_run", lambda: active_outer_run)
    monkeypatch.setattr(
        run_module,
        "preflight_bootstrap_urbansim_datastore_admission",
        lambda **kwargs: events.append("admission") or captured.update(kwargs),
    )
    monkeypatch.setattr(
        run_module,
        "has_initial_skims_admission_policy",
        lambda _settings: True,
    )
    monkeypatch.setattr(
        run_module,
        "requires_prior_beam_skim_handoff",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "preflight_activitysim_initial_skims_admission",
        lambda **_kwargs: events.append("skims_admission"),
    )
    monkeypatch.setattr(
        run_module,
        "seed_bootstrap_artifacts_to_coupler",
        lambda **_kwargs: events.append("seed"),
    )
    monkeypatch.setattr(
        run_module,
        "_rewind_uncommitted_activitysim_restart_to_provider_boundary",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run_module,
        "log_promotion_instructions_on_success",
        lambda **_kwargs: None,
    )

    run_module.main()

    assert events == [
        "scenario_enter",
        "declare_outputs",
        "admission",
        "skims_admission",
        "seed",
        "scenario_exit",
    ]
    assert captured == {
        "settings": prepared.settings,
        "metadata_logger": prepared.tracker,
        "workspace_path": Path(prepared.workspace.full_path),
        "report_dir": Path(prepared.archive_run_dir),
        "existing_admission_reports": existing_reports,
    }


def test_main_strict_bootstrap_admission_rejects_before_seeding_or_model_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected datastore must stop before the coupler or any model can consume it."""
    events: list[str] = []
    prepared = _prepared_outer_admission_run(
        tmp_path,
        run_models=True,
    )

    monkeypatch.setattr(run_module, "_prepare_run_context", lambda **_kwargs: prepared)
    monkeypatch.setattr(
        run_module,
        "_build_scenario_runtime_contract",
        lambda **_kwargs: {
            "scenario_kwargs": {},
            "schema_steps_all": (),
            "schema_steps_enabled": (),
            "coupler_schema": {},
            "required_output_keys": (),
        },
    )
    monkeypatch.setattr(
        run_module.cr,
        "scenario",
        lambda *_args, **_kwargs: _OuterScenarioStub(events),
    )
    monkeypatch.setattr(run_module.cr, "current_run", lambda: SimpleNamespace(meta={}))
    monkeypatch.setattr(
        run_module,
        "preflight_bootstrap_urbansim_datastore_admission",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("strict rejection")),
    )
    monkeypatch.setattr(
        run_module,
        "seed_bootstrap_artifacts_to_coupler",
        lambda **_kwargs: events.append("seed"),
    )
    monkeypatch.setattr(
        run_module,
        "run_land_use_stage",
        lambda **_kwargs: events.append("model_execution"),
    )

    with pytest.raises(RuntimeError, match="strict rejection"):
        run_module.main()

    assert events == ["scenario_enter", "declare_outputs", "scenario_exit"]


def test_main_ordinary_resume_skips_admission_and_seeds_coupler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary resume has no bootstrap datastore to admit."""
    events: list[str] = []
    prepared = _prepared_outer_admission_run(tmp_path, is_restart_run=True)

    monkeypatch.setattr(run_module, "_prepare_run_context", lambda **_kwargs: prepared)
    monkeypatch.setattr(
        run_module.restart_runtime,
        "restore_restart_coupler_snapshot",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        run_module,
        "_build_scenario_runtime_contract",
        lambda **_kwargs: {
            "scenario_kwargs": {},
            "schema_steps_all": (),
            "schema_steps_enabled": (),
            "coupler_schema": {},
            "required_output_keys": (),
        },
    )
    monkeypatch.setattr(
        run_module.cr,
        "scenario",
        lambda *_args, **_kwargs: _OuterScenarioStub(events),
    )
    monkeypatch.setattr(
        run_module,
        "preflight_bootstrap_urbansim_datastore_admission",
        lambda **_kwargs: events.append("admission"),
    )
    monkeypatch.setattr(
        run_module,
        "has_initial_skims_admission_policy",
        lambda _settings: True,
    )
    monkeypatch.setattr(
        run_module,
        "requires_prior_beam_skim_handoff",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "preflight_activitysim_initial_skims_admission",
        lambda **_kwargs: events.append("skims_admission"),
    )
    monkeypatch.setattr(
        run_module,
        "seed_bootstrap_artifacts_to_coupler",
        lambda **_kwargs: events.append("seed"),
    )
    monkeypatch.setattr(
        run_module,
        "_rewind_uncommitted_activitysim_restart_to_provider_boundary",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run_module,
        "log_promotion_instructions_on_success",
        lambda **_kwargs: None,
    )

    run_module.main()

    assert events == [
        "scenario_enter",
        "declare_outputs",
        "seed",
        "scenario_exit",
    ]
