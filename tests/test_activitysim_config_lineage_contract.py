from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.utils.consist_config import (
    build_activitysim_identity_inputs,
    build_step_consist_kwargs,
)
from pilates.workflows.steps import StepOutputsHolder, make_activitysim_run_step


class DummyCoupler:
    def get(self, key, default=None):
        return default

    def set(self, key, value) -> None:
        return None


class DummyWorkspace:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.full_path = str(root)

    def get_asim_mutable_configs_dir(self) -> str:
        return str(self._root / "activitysim" / "configs")

    def get_asim_mutable_data_dir(self) -> str:
        return str(self._root / "activitysim" / "data")


class DummyStepContext:
    def __init__(self, *, settings, workspace) -> None:
        self._runtime = {
            "settings": settings,
            "workspace": workspace,
        }

    def get_runtime(self, name, default=None):
        return self._runtime.get(name, default)


def _make_activitysim_settings(*, main_configs_dir: str = "base", chunk_size: int = 0):
    return SimpleNamespace(
        activitysim=SimpleNamespace(
            household_sample_size=100,
            chunk_size=chunk_size,
            num_processes=4,
            file_format="csv",
            warm_start_activities=False,
            replan_iters=0,
            replan_hh_samp_size=0,
            replan_after=0,
            random_seed=42,
            database=SimpleNamespace(
                enabled=False,
                use_processed_data=False,
                year=2018,
            ),
            local_mutable_configs_folder="activitysim/configs",
            main_configs_dir=main_configs_dir,
        ),
    )


def _write_settings_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("models: []\n")


def test_activitysim_identity_inputs_anchor_to_mutable_configs_root(tmp_path):
    settings = _make_activitysim_settings()
    configs_root = tmp_path / "activitysim" / "configs"
    _write_settings_yaml(configs_root / "base" / "settings.yaml")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "ignore.txt").write_text("unrelated")

    identity_inputs = build_activitysim_identity_inputs(settings, str(tmp_path))

    assert identity_inputs == [("asim_mutable_configs", configs_root)]


def test_activitysim_identity_config_changes_for_relevant_settings_but_not_identity_root(
    tmp_path,
):
    configs_root = tmp_path / "activitysim" / "configs"
    _write_settings_yaml(configs_root / "base" / "settings.yaml")

    default_settings = _make_activitysim_settings(chunk_size=0)
    changed_settings = _make_activitysim_settings(chunk_size=8192)

    default_kwargs = build_step_consist_kwargs(
        "activitysim_run",
        default_settings,
        workspace_path=str(tmp_path),
    )
    changed_kwargs = build_step_consist_kwargs(
        "activitysim_run",
        changed_settings,
        workspace_path=str(tmp_path),
    )

    assert default_kwargs["identity_inputs"] == changed_kwargs["identity_inputs"]
    assert default_kwargs["identity_inputs"] == [("asim_mutable_configs", configs_root)]
    assert default_kwargs["config"] != changed_kwargs["config"]


def test_activitysim_run_adapter_ignores_irrelevant_workspace_changes(tmp_path):
    pytest.importorskip("consist")

    workspace = DummyWorkspace(tmp_path)
    settings = _make_activitysim_settings(main_configs_dir="base")
    base_root = Path(workspace.get_asim_mutable_configs_dir()) / "base"
    _write_settings_yaml(base_root / "settings.yaml")

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__

    initial_adapter = meta.adapter(
        DummyStepContext(settings=settings, workspace=workspace)
    )
    assert initial_adapter.root_dirs == [base_root]

    (tmp_path / "beam" / "input" / "seattle").mkdir(parents=True)
    (tmp_path / "beam" / "input" / "seattle" / "main.conf").write_text("beam")
    (tmp_path / "notes.txt").write_text("ignore")

    adapter_after_irrelevant_changes = meta.adapter(
        DummyStepContext(settings=settings, workspace=workspace)
    )
    assert adapter_after_irrelevant_changes.root_dirs == [base_root]


def test_activitysim_run_adapter_adds_overlay_roots_when_config_tree_expands(tmp_path):
    pytest.importorskip("consist")

    workspace = DummyWorkspace(tmp_path)
    settings = _make_activitysim_settings(main_configs_dir="base")
    configs_root = Path(workspace.get_asim_mutable_configs_dir())
    base_root = configs_root / "base"
    _write_settings_yaml(base_root / "settings.yaml")

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__

    base_only_adapter = meta.adapter(
        DummyStepContext(settings=settings, workspace=workspace)
    )
    assert base_only_adapter.root_dirs == [base_root]

    mp_root = configs_root / "configs_mp"
    compile_root = configs_root / "configs_sh_compile"
    _write_settings_yaml(mp_root / "settings.yaml")
    _write_settings_yaml(compile_root / "settings.yaml")

    expanded_adapter = meta.adapter(
        DummyStepContext(settings=settings, workspace=workspace)
    )
    assert expanded_adapter.root_dirs == [base_root, mp_root, compile_root]
