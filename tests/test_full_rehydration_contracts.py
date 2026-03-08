from pathlib import Path
from types import SimpleNamespace

import run as run_module
from workflow_state import WorkflowState


class _WorkspaceStub:
    def __init__(self, full_path: str, settings=None):
        self.full_path = full_path
        self.settings = settings
        self.input_data = {}
        self.output_data = {}

    def get_usim_mutable_data_dir(self):
        return str(Path(self.full_path) / "urbansim" / "data")

    def get_asim_mutable_data_dir(self):
        return str(Path(self.full_path) / "activitysim" / "data")

    def get_asim_mutable_configs_dir(self):
        return str(Path(self.full_path) / "activitysim" / "configs")

    def get_asim_output_dir(self):
        return str(Path(self.full_path) / "activitysim" / "output")

    def get_atlas_mutable_input_dir(self):
        return str(Path(self.full_path) / "atlas" / "atlas_input")


def _restart_settings():
    return SimpleNamespace(
        run=SimpleNamespace(
            region="test",
            models=SimpleNamespace(
                activity_demand="activitysim",
                vehicle_ownership="atlas",
            ),
        ),
        atlas=SimpleNamespace(scenario="baseline"),
        activitysim=SimpleNamespace(main_configs_dir="configs"),
        urbansim=SimpleNamespace(
            region_mappings={"region_to_region_id": {"test": "000"}},
            input_file_template="usim_{region_id}.h5",
            output_file_template="usim_{year}.h5",
        ),
    )


def _write_file(path: Path, contents: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def test_full_rehydrate_repairs_activitysim_config_tree_and_preserves_existing_files(
    tmp_path,
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"

    local_main_settings = _write_file(
        local_run_dir / "activitysim" / "configs" / "configs" / "settings.yaml",
        "local-main\n",
    )

    _write_file(
        archive_run_dir / "activitysim" / "configs" / "configs" / "settings.yaml",
        "archive-main\n",
    )
    _write_file(
        archive_run_dir / "activitysim" / "configs" / "configs_extended" / "settings.yaml",
        "archive-extended\n",
    )
    _write_file(
        archive_run_dir / "activitysim" / "configs" / "configs_mp" / "settings.yaml",
        "archive-mp\n",
    )
    _write_file(
        archive_run_dir / "activitysim" / "configs" / "configs_mp" / "constants.yaml",
        "archive-constants\n",
    )
    _write_file(
        archive_run_dir
        / "activitysim"
        / "configs"
        / "configs_sh_compile"
        / "settings.yaml",
        "archive-compile\n",
    )

    summary = run_module._rehydrate_full_local_run_from_archive(
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )

    assert summary["copied"] >= 4
    assert summary["skipped_existing"] >= 1
    assert local_main_settings.read_text(encoding="utf-8") == "local-main\n"
    assert (
        local_run_dir / "activitysim" / "configs" / "configs_extended" / "settings.yaml"
    ).read_text(encoding="utf-8") == "archive-extended\n"
    assert (
        local_run_dir / "activitysim" / "configs" / "configs_mp" / "settings.yaml"
    ).read_text(encoding="utf-8") == "archive-mp\n"
    assert (
        local_run_dir / "activitysim" / "configs" / "configs_mp" / "constants.yaml"
    ).read_text(encoding="utf-8") == "archive-constants\n"
    assert (
        local_run_dir
        / "activitysim"
        / "configs"
        / "configs_sh_compile"
        / "settings.yaml"
    ).read_text(encoding="utf-8") == "archive-compile\n"


def test_full_rehydrate_clears_activitysim_restart_missing_artifacts(tmp_path):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    settings = _restart_settings()
    workspace = _WorkspaceStub(str(local_run_dir), settings=settings)
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        asim_compiled=False,
    )

    for filename in ("households.csv", "persons.csv", "land_use.csv"):
        _write_file(
            archive_run_dir / "activitysim" / "data" / filename,
            f"archive-{filename}\n",
        )
    _write_file(
        archive_run_dir / "urbansim" / "data" / "usim_000.h5",
        "archive-usim\n",
    )
    for dirname in (
        "configs",
        "configs_extended",
        "configs_mp",
        "configs_sh_compile",
    ):
        _write_file(
            archive_run_dir / "activitysim" / "configs" / dirname / "settings.yaml",
            f"{dirname}: true\n",
        )

    before = run_module._find_missing_restart_local_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    assert {item["key"] for item in before} >= {
        "activitysim_input_households.csv",
        "activitysim_input_persons.csv",
        "activitysim_input_land_use.csv",
        "activitysim_config_settings_yaml_configs",
        "activitysim_config_settings_yaml_configs_mp",
        "activitysim_config_settings_yaml_configs_sh_compile",
    }

    summary = run_module._rehydrate_full_local_run_from_archive(
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )

    assert summary["copied"] >= 8

    after = run_module._find_missing_restart_local_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    assert after == []


def test_full_rehydrate_restores_representative_urbansim_and_atlas_paths(tmp_path):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"

    local_year_dir = local_run_dir / "atlas" / "atlas_input" / "year2023"
    local_year_file = _write_file(local_year_dir / "vehicles_output.RData", "local-rdata\n")

    _write_file(
        archive_run_dir / "urbansim" / "data" / "usim_000.h5",
        "archive-h5\n",
    )
    _write_file(
        archive_run_dir / "atlas" / "atlas_input" / "psid_names.Rdat",
        "archive-psid\n",
    )
    _write_file(
        archive_run_dir / "atlas" / "atlas_input" / "year2023" / "vehicles_output.RData",
        "archive-rdata\n",
    )
    _write_file(
        archive_run_dir / "atlas" / "atlas_input" / "year2023" / "metadata.txt",
        "archive-meta\n",
    )

    summary = run_module._rehydrate_full_local_run_from_archive(
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )

    assert summary["copied"] >= 3
    assert summary["skipped_existing"] >= 1
    assert (
        local_run_dir / "urbansim" / "data" / "usim_000.h5"
    ).read_text(encoding="utf-8") == "archive-h5\n"
    assert (
        local_run_dir / "atlas" / "atlas_input" / "psid_names.Rdat"
    ).read_text(encoding="utf-8") == "archive-psid\n"
    assert local_year_file.read_text(encoding="utf-8") == "local-rdata\n"
    assert (
        local_run_dir / "atlas" / "atlas_input" / "year2023" / "metadata.txt"
    ).read_text(encoding="utf-8") == "archive-meta\n"
