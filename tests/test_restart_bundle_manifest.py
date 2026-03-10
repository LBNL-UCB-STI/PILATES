from types import SimpleNamespace
from typing import Optional

from pilates.utils.restart_bundle import build_restart_bundle_manifest


class _WorkspaceStub:
    def __init__(
        self,
        atlas_input_dir: Optional[str],
        asim_base_dir: Optional[str] = None,
        usim_base_dir: Optional[str] = None,
    ):
        self._atlas_input_dir = atlas_input_dir
        self._asim_base_dir = asim_base_dir
        self._usim_base_dir = usim_base_dir

    def get_atlas_mutable_input_dir(self):
        return self._atlas_input_dir

    def get_asim_mutable_data_dir(self):
        return f"{self._asim_base_dir}/data"

    def get_asim_mutable_configs_dir(self):
        return f"{self._asim_base_dir}/configs"

    def get_asim_output_dir(self):
        return f"{self._asim_base_dir}/output"

    def get_usim_mutable_data_dir(self):
        return self._usim_base_dir


def _settings_with_atlas_vehicle_ownership():
    return SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(vehicle_ownership="atlas", activity_demand=None),
            start_year=2017,
            end_year=2050,
        ),
        atlas=SimpleNamespace(scenario="baseline"),
    )


def _settings_without_atlas_vehicle_ownership():
    return SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use=None, vehicle_ownership="none", activity_demand=None
            ),
            start_year=2017,
            end_year=2050,
        ),
        atlas=SimpleNamespace(scenario="baseline"),
    )


def _settings_with_activitysim():
    return SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use=None,
                vehicle_ownership="none",
                activity_demand="activitysim",
            ),
            region="test",
            start_year=2017,
            end_year=2050,
        ),
        activitysim=SimpleNamespace(main_configs_dir="configs"),
        urbansim=SimpleNamespace(
            input_file_template="usim_{region_id}.h5",
            region_mappings={"region_to_region_id": {"test": "000"}},
        ),
        atlas=SimpleNamespace(scenario="baseline"),
    )


def _settings_without_urbansim_land_use():
    return SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                land_use=None, vehicle_ownership="none", activity_demand=None
            ),
            start_year=2017,
            end_year=2050,
        ),
        atlas=SimpleNamespace(scenario="baseline"),
        urbansim=None,
    )


def test_restart_bundle_includes_atlas_static_candidates_for_atlas_vehicle_ownership(
    tmp_path,
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    atlas_input_dir = local_run_dir / "atlas" / "atlas_input"
    atlas_input_dir.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / "run_state.yaml").parent.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / "run_state.yaml").write_text("year: 2023\n", encoding="utf-8")

    for relpath in ("psid_names.Rdat", "accessbility_2015.RData"):
        path = atlas_input_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

        archive_path = archive_run_dir / "atlas" / "atlas_input" / relpath
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text("x", encoding="utf-8")

    manifest = build_restart_bundle_manifest(
        archive_run_dir=str(archive_run_dir),
        local_run_dir=str(local_run_dir),
        settings=_settings_with_atlas_vehicle_ownership(),
        workspace=_WorkspaceStub(str(atlas_input_dir)),
        state=SimpleNamespace(current_year=2023),
        local_consist_db_path=None,
    )

    keys = {item["key"] for item in manifest["artifacts"]}
    assert "atlas_static::psid_names.Rdat" in keys
    assert "atlas_static::accessbility_2015.RData" in keys


def test_restart_bundle_skips_atlas_static_candidates_when_vehicle_ownership_not_atlas(
    tmp_path,
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    atlas_input_dir = local_run_dir / "atlas" / "atlas_input"
    atlas_input_dir.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / "run_state.yaml").parent.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / "run_state.yaml").write_text("year: 2023\n", encoding="utf-8")

    manifest = build_restart_bundle_manifest(
        archive_run_dir=str(archive_run_dir),
        local_run_dir=str(local_run_dir),
        settings=_settings_without_atlas_vehicle_ownership(),
        workspace=_WorkspaceStub(str(atlas_input_dir)),
        state=SimpleNamespace(current_year=2023),
        local_consist_db_path=None,
    )

    keys = {item["key"] for item in manifest["artifacts"]}
    assert not any(key.startswith("atlas_static::") for key in keys)


def test_restart_bundle_skips_atlas_year_candidates_when_vehicle_ownership_not_atlas(
    tmp_path,
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    (archive_run_dir / "run_state.yaml").parent.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / "run_state.yaml").write_text("year: 2023\n", encoding="utf-8")

    manifest = build_restart_bundle_manifest(
        archive_run_dir=str(archive_run_dir),
        local_run_dir=str(local_run_dir),
        settings=_settings_without_atlas_vehicle_ownership(),
        workspace=_WorkspaceStub(None),
        state=SimpleNamespace(current_year=2023),
        local_consist_db_path=None,
    )

    keys = {item["key"] for item in manifest["artifacts"]}
    assert not any(key.startswith("atlas_input_year") for key in keys)
    assert not any(key.startswith("atlas_output_year") for key in keys)


def test_restart_bundle_includes_activitysim_zarr_candidate(tmp_path):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    atlas_input_dir = local_run_dir / "atlas" / "atlas_input"
    asim_base_dir = local_run_dir / "activitysim"
    usim_base_dir = local_run_dir / "urbansim" / "data"
    atlas_input_dir.mkdir(parents=True, exist_ok=True)
    usim_base_dir.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / "run_state.yaml").parent.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / "run_state.yaml").write_text("year: 2023\n", encoding="utf-8")

    # Create local+archive zarr paths so manifest metadata can mark exists_in_archive.
    zarr_local = asim_base_dir / "output" / "cache" / "skims.zarr" / "0"
    zarr_local.mkdir(parents=True, exist_ok=True)
    (zarr_local / "values").write_text("x", encoding="utf-8")

    zarr_archive = archive_run_dir / "activitysim" / "output" / "cache" / "skims.zarr" / "0"
    zarr_archive.mkdir(parents=True, exist_ok=True)
    (zarr_archive / "values").write_text("x", encoding="utf-8")

    usim_local = usim_base_dir / "usim_000.h5"
    usim_local.write_text("x", encoding="utf-8")
    usim_archive = archive_run_dir / "urbansim" / "data" / "usim_000.h5"
    usim_archive.parent.mkdir(parents=True, exist_ok=True)
    usim_archive.write_text("x", encoding="utf-8")

    manifest = build_restart_bundle_manifest(
        archive_run_dir=str(archive_run_dir),
        local_run_dir=str(local_run_dir),
        settings=_settings_with_activitysim(),
        workspace=_WorkspaceStub(
            str(atlas_input_dir),
            str(asim_base_dir),
            str(usim_base_dir),
        ),
        state=SimpleNamespace(current_year=2023),
        local_consist_db_path=None,
    )

    keys = {item["key"] for item in manifest["artifacts"]}
    assert "usim_datastore_base_h5" in keys
    assert "zarr_skims" in keys
    assert "asim_sharrow_cache_dir" in keys
    assert "activitysim_config_dir_configs" in keys
    assert "activitysim_config_dir_configs_extended" in keys
    assert "activitysim_config_dir_configs_mp" in keys
    assert "activitysim_config_dir_configs_sh_compile" in keys


def test_restart_bundle_skips_urbansim_candidates_when_land_use_not_urbansim(
    tmp_path,
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    (archive_run_dir / "run_state.yaml").parent.mkdir(parents=True, exist_ok=True)
    (archive_run_dir / "run_state.yaml").write_text("year: 2023\n", encoding="utf-8")

    manifest = build_restart_bundle_manifest(
        archive_run_dir=str(archive_run_dir),
        local_run_dir=str(local_run_dir),
        settings=_settings_without_urbansim_land_use(),
        workspace=_WorkspaceStub(None, None, None),
        state=SimpleNamespace(current_year=2023),
        local_consist_db_path=None,
    )

    keys = {item["key"] for item in manifest["artifacts"]}
    assert "usim_datastore_base_h5" not in keys
    assert "usim_datastore_current_h5" not in keys
