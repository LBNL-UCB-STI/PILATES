from __future__ import annotations

import os
from pathlib import Path

import pytest


_DISABLE_CONSIST_LOGGING_BASENAMES = {
    "test_activitysim_compile_run_handshake.py",
    "test_archive_copy_workflow.py",
    "test_manifest_cache_parity.py",
    "test_workflow_contract_validation.py",
}


@pytest.fixture(autouse=True)
def _disable_consist_logging_for_isolated_step_tests(request):
    yield


@pytest.fixture(autouse=True)
def _run_tests_from_repo_root(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow_ci: tests excluded from the default CI pytest run because they are"
        " long-running end-to-end coverage",
    )
    config.addinivalue_line(
        "markers",
        "contract_harness: micro-world contract-validating workflow boundary tests"
        " run in their own dedicated CI step",
    )


@pytest.fixture
def golden_stub_env(tmp_path, monkeypatch):
    """Expose the golden stub environment without globally loading its module."""
    from tests.test_golden_stub_workflow import golden_stub_env as _golden_stub_env

    yield from _golden_stub_env.__wrapped__(tmp_path, monkeypatch)


# ============================================================================
# Shared strongly-typed conforming Workspace & WorkflowRuntimeContext doubles
# ============================================================================

from typing import Any, Callable, Optional  # noqa: E402

from pilates.config.models import PilatesConfig  # noqa: E402
from pilates.runtime.context import WorkflowRuntimeContext  # noqa: E402
from pilates.workflows.state import WorkflowState  # noqa: E402
from pilates.workflows.surface import EnabledWorkflowSurface  # noqa: E402
from pilates.workspace import Workspace  # noqa: E402


def make_conforming_settings(
    tmp_path: Path | str,
    **overrides: Any,
) -> PilatesConfig:
    """Build a complete, strongly-typed PilatesConfig rooted under tmp_path."""
    root = Path(tmp_path)
    config: dict[str, Any] = {
        "run": {
            "region": "test",
            "scenario": "test",
            "start_year": 2020,
            "end_year": 2021,
            "output_directory": str(root / "outputs"),
            "output_run_name": "test_run",
            "supply_demand_iters": 1,
            "models": {
                "land_use": "urbansim",
                "travel": "beam",
                "activity_demand": "activitysim",
                "vehicle_ownership": "atlas",
            },
        },
        "shared": {
            "geography": {
                "FIPS": {"county": ["00001"]},
                "local_crs": "EPSG:4326",
            },
            "skims": {"fname": "skims.omx"},
            "database": {
                "enabled": False,
                "type": "duckdb",
                "path": str(root / "db.duckdb"),
            },
        },
        "infrastructure": {
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
        "urbansim": {
            "local_data_input_folder": str(root / "usim_input"),
            "local_mutable_data_folder": "urbansim/data",
            "client_base_folder": "/usim",
            "client_data_folder": "/usim/data",
            "input_file_template": "usim_{region_id}.h5",
            "input_file_template_year": "usim_{region_id}_{year}.h5",
            "output_file_template": "usim_{year}.h5",
            "command_template": "run_usim",
            "region_mappings": {"region_to_region_id": {"test": "000"}},
        },
        "atlas": {
            "host_input_folder": "atlas/input",
            "warmstart_input_folder": "atlas/warmstart",
            "host_mutable_input_folder": "atlas/atlas_input",
            "host_output_folder": "atlas/atlas_output",
            "container_input_folder": "/atlas/input",
            "container_output_folder": "/atlas/output",
            "basedir": "/atlas",
            "codedir": "/atlas/code",
            "command_template": "atlas {0}",
        },
        "activitysim": {
            "local_input_folder": "activitysim/input",
            "local_mutable_data_folder": "activitysim/data",
            "local_output_folder": "activitysim/output",
            "local_configs_folder": "activitysim/configs",
            "local_mutable_configs_folder": "activitysim/configs_mutable",
            "validation_folder": "activitysim/validation",
            "command_template": "asim run",
            "final_plans_folder": "activitysim/final_plans",
            "region_mappings": {"region_to_subdir": {"test": "test"}},
        },
        "beam": {
            "config": "beam.conf",
            "local_input_folder": "beam/input",
            "local_mutable_data_folder": "beam/input",
            "local_output_folder": "beam/output",
            "scenario_folder": "beam/scenario",
            "router_directory": "router",
            "skims_shapefile": "beam/skims.shp",
            "skim_zone_source_id_col": "id",
            "skim_zone_geoid_col": "geoid",
        },
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in config and isinstance(config[key], dict):
            config[key].update(val)
        else:
            config[key] = val
    return PilatesConfig(**config)


class _CallablePathGetter:
    """Callable path getter double supporting optional mock-style return_value overrides."""

    def __init__(
        self,
        default_fn: Callable[[], str],
        initial_override: Optional[Any] = None,
    ) -> None:
        self._default_fn = default_fn
        self.return_value: Optional[str] = None
        if initial_override is not None:
            if callable(initial_override):
                self._default_fn = initial_override
            else:
                self.return_value = str(initial_override)

    def __call__(self) -> str:
        if self.return_value is not None:
            return str(self.return_value)
        return str(self._default_fn())


class ConformingTestWorkspace(Workspace):
    """
    Strongly-typed Workspace double that conforms to the real Workspace interface,
    with explicit support for keyword overrides for path getters and properties.
    """

    def __init__(
        self,
        settings: PilatesConfig,
        output_path: str,
        folder_name: str = "run",
        *,
        archive_run_dir: Optional[str] = None,
        on_archive_commit: Optional[Callable[[str], None]] = None,
        expected_prefix_steps: int = 0,
        full_path: Optional[str | Path] = None,
        usim_mutable_data_dir: Optional[str | Path | Callable[[], str]] = None,
        asim_mutable_data_dir: Optional[str | Path | Callable[[], str]] = None,
        asim_mutable_configs_dir: Optional[str | Path | Callable[[], str]] = None,
        asim_output_dir: Optional[str | Path | Callable[[], str]] = None,
        asim_runtime_cache_dir: Optional[str | Path | Callable[[], str]] = None,
        beam_mutable_data_dir: Optional[str | Path | Callable[[], str]] = None,
        beam_output_dir: Optional[str | Path | Callable[[], str]] = None,
        beam_config_dir: Optional[str | Path | Callable[[], str]] = None,
        atlas_mutable_input_dir: Optional[str | Path | Callable[[], str]] = None,
        atlas_output_dir: Optional[str | Path | Callable[[], str]] = None,
        setup_directories: bool = True,
        **extra_attrs: Any,
    ) -> None:
        self._full_path_override = str(full_path) if full_path is not None else None
        self._setup_dirs = setup_directories
        super().__init__(
            settings=settings,
            output_path=output_path,
            folder_name=folder_name,
            archive_run_dir=archive_run_dir,
            on_archive_commit=on_archive_commit,
            expected_prefix_steps=expected_prefix_steps,
        )
        self.get_usim_mutable_data_dir = _CallablePathGetter(
            super().get_usim_mutable_data_dir, usim_mutable_data_dir
        )
        self.get_asim_mutable_data_dir = _CallablePathGetter(
            super().get_asim_mutable_data_dir, asim_mutable_data_dir
        )
        self.get_asim_mutable_configs_dir = _CallablePathGetter(
            super().get_asim_mutable_configs_dir, asim_mutable_configs_dir
        )
        self.get_asim_output_dir = _CallablePathGetter(
            super().get_asim_output_dir, asim_output_dir
        )
        self.get_asim_runtime_cache_dir = _CallablePathGetter(
            super().get_asim_runtime_cache_dir, asim_runtime_cache_dir
        )
        self.get_beam_mutable_data_dir = _CallablePathGetter(
            super().get_beam_mutable_data_dir, beam_mutable_data_dir
        )
        self.get_beam_output_dir = _CallablePathGetter(
            super().get_beam_output_dir, beam_output_dir
        )
        self.get_beam_config_dir = _CallablePathGetter(
            lambda: os.path.join(self.get_beam_mutable_data_dir(), "conf"),
            beam_config_dir,
        )
        self.get_atlas_mutable_input_dir = _CallablePathGetter(
            super().get_atlas_mutable_input_dir, atlas_mutable_input_dir
        )
        self.get_atlas_output_dir = _CallablePathGetter(
            super().get_atlas_output_dir, atlas_output_dir
        )
        for key, val in extra_attrs.items():
            setattr(self, key, val)

    @property
    def full_path(self) -> str:
        if self._full_path_override is not None:
            return self._full_path_override
        return super().full_path

    def _setup_directories(self) -> None:
        if not self._setup_dirs:
            return
        try:
            super()._setup_directories()
        except OSError:
            pass


def make_conforming_workspace(
    tmp_path: Path | str,
    settings: Optional[PilatesConfig] = None,
    *,
    output_path: Optional[str] = None,
    folder_name: str = "run",
    archive_run_dir: Optional[str] = None,
    on_archive_commit: Optional[Callable[[str], None]] = None,
    expected_prefix_steps: int = 0,
    **overrides: Any,
) -> ConformingTestWorkspace:
    """Factory helper to create a fully initialized ConformingTestWorkspace."""
    root = Path(tmp_path)
    if settings is None:
        settings = make_conforming_settings(root)
    resolved_output_path = output_path if output_path is not None else str(root)

    # Remap get_* overrides to their corresponding field names
    remap = {
        "get_usim_mutable_data_dir": "usim_mutable_data_dir",
        "get_asim_mutable_data_dir": "asim_mutable_data_dir",
        "get_asim_mutable_configs_dir": "asim_mutable_configs_dir",
        "get_asim_output_dir": "asim_output_dir",
        "get_asim_runtime_cache_dir": "asim_runtime_cache_dir",
        "get_beam_mutable_data_dir": "beam_mutable_data_dir",
        "get_beam_output_dir": "beam_output_dir",
        "get_beam_config_dir": "beam_config_dir",
        "get_atlas_mutable_input_dir": "atlas_mutable_input_dir",
        "get_atlas_output_dir": "atlas_output_dir",
    }
    normalized_overrides: dict[str, Any] = {}
    for key, val in overrides.items():
        mapped_key = remap.get(key, key)
        normalized_overrides[mapped_key] = val

    return ConformingTestWorkspace(
        settings=settings,
        output_path=resolved_output_path,
        folder_name=folder_name,
        archive_run_dir=archive_run_dir,
        on_archive_commit=on_archive_commit,
        expected_prefix_steps=expected_prefix_steps,
        **normalized_overrides,
    )


def make_conforming_workflow_runtime_context(
    tmp_path: Path | str,
    *,
    settings: Optional[PilatesConfig] = None,
    state: Optional[WorkflowState] = None,
    workspace: Optional[Workspace] = None,
    surface: Optional[EnabledWorkflowSurface] = None,
    on_archive_commit: Optional[Callable[[str], None]] = None,
    expected_prefix_steps: int = 0,
    **workspace_kwargs: Any,
) -> WorkflowRuntimeContext:
    """Factory helper to create a fully initialized conforming WorkflowRuntimeContext."""
    root = Path(tmp_path)
    if settings is None:
        settings = make_conforming_settings(root)
    if workspace is None:
        workspace = make_conforming_workspace(
            root, settings=settings, **workspace_kwargs
        )
    if state is None:
        state = WorkflowState.from_settings(settings)
    return WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
        on_archive_commit=on_archive_commit,
        expected_prefix_steps=expected_prefix_steps,
    )


@pytest.fixture
def conforming_workspace_factory():
    return make_conforming_workspace


@pytest.fixture
def conforming_context_factory():
    return make_conforming_workflow_runtime_context


@pytest.fixture
def conforming_workspace(tmp_path: Path) -> ConformingTestWorkspace:
    return make_conforming_workspace(tmp_path)


@pytest.fixture
def conforming_context(tmp_path: Path) -> WorkflowRuntimeContext:
    return make_conforming_workflow_runtime_context(tmp_path)


@pytest.fixture(autouse=True)
def _clear_global_tracker():
    from pilates.utils.consist_runtime import set_tracker

    yield
    set_tracker(None)
