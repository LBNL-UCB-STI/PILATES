import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim.runner import (
    ActivitySimLaunchContext,
    ActivitysimNumbaWarmup,
    ActivitysimRunner,
    _asim_container_environment,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        activitysim=SimpleNamespace(
            local_mutable_data_folder="activitysim/data",
            local_output_folder="activitysim/output",
            local_mutable_configs_folder="activitysim/configs",
            local_configs_folder="pilates/activitysim/configs",
            main_configs_dir="configs_extended",
            file_format="parquet",
            persist_sharrow_cache=None,
            region_mappings={
                "region_to_subdir": {
                    "sfbay": "activitysim/examples/prototype_mtc_clean"
                }
            },
        ),
    )


def test_activitysim_docker_vols_include_configs_mp_mount():
    settings = _settings()
    working_dir = "/tmp/pilates-workdir"
    vols = ActivitysimRunner.get_asim_docker_vols(settings, working_dir=working_dir)

    expected_local = os.path.abspath(
        os.path.join(working_dir, "activitysim/configs", "configs_mp")
    )
    expected_bind = "/activitysim/activitysim/examples/prototype_mtc_clean/configs_mp"

    assert expected_local in vols
    assert vols[expected_local]["bind"] == expected_bind
    assert vols[expected_local]["mode"] == "rw"


def test_activitysim_docker_vols_mount_only_the_resolved_staged_launch_tree(
    tmp_path: Path,
) -> None:
    """A native launch must not rediscover poisoned workspace input roots."""
    settings = _settings()
    staged_data = tmp_path / "native-launch" / "data"
    staged_configs = tmp_path / "native-launch" / "configs"
    for dirname in ("configs", "configs_extended", "configs_mp", "configs_sh_compile"):
        (staged_configs / dirname).mkdir(parents=True)
    launch_context = ActivitySimLaunchContext(
        workspace_root=tmp_path,
        mutable_data_dir=staged_data,
        output_dir=tmp_path / "private-output",
        compile_output_dir=tmp_path / "private-compile-output",
        mutable_configs_dir=staged_configs,
        runtime_cache_dir=tmp_path / "private-output" / "cache",
        runtime_zarr_path=tmp_path / "private-output" / "cache" / "skims.zarr",
        shared_cache_dir=tmp_path / "shared-cache",
        shared_tmp_dir=tmp_path / "tmp",
    )
    poisoned_working_dir = tmp_path / "poisoned-workspace"

    vols = ActivitysimRunner.get_asim_docker_vols(
        settings,
        working_dir=str(poisoned_working_dir),
        launch_context=launch_context,
    )

    assert str(poisoned_working_dir) not in " ".join(vols)
    assert vols[os.path.abspath(staged_data)]["mode"] == "ro"
    for dirname in ("configs_extended", "configs_mp", "configs_sh_compile"):
        assert vols[os.path.abspath(staged_configs / dirname)]["mode"] == "ro"
    assert vols[os.path.abspath(launch_context.output_dir)]["mode"] == "rw"


def test_activitysim_docker_vols_reject_missing_staged_config_before_launch(
    tmp_path: Path,
) -> None:
    """A missing staged overlay must fail instead of falling back to workspace."""
    settings = _settings()
    staged_configs = tmp_path / "native-launch" / "configs"
    (staged_configs / "configs_extended").mkdir(parents=True)
    (staged_configs / "configs_mp").mkdir()
    source = tmp_path / "identity-configs" / "configs_extended"
    source.mkdir(parents=True)
    launch_context = ActivitySimLaunchContext(
        workspace_root=tmp_path,
        mutable_data_dir=tmp_path / "native-launch" / "data",
        output_dir=tmp_path / "private-output",
        compile_output_dir=tmp_path / "private-compile-output",
        mutable_configs_dir=staged_configs,
        runtime_cache_dir=tmp_path / "private-output" / "cache",
        runtime_zarr_path=tmp_path / "private-output" / "cache" / "skims.zarr",
        shared_cache_dir=tmp_path / "shared-cache",
        shared_tmp_dir=tmp_path / "tmp",
        config_source_roots=(source,),
    )

    with pytest.raises(RuntimeError, match="missing staged configuration directory"):
        ActivitysimRunner.get_asim_docker_vols(settings, launch_context=launch_context)


def test_activitysim_numba_warmup_args_skip_configs_mp():
    settings = _settings()
    working_dir = "/tmp/pilates-workdir"
    vols = ActivitysimRunner.get_asim_docker_vols(settings, working_dir=working_dir)

    args = ActivitysimNumbaWarmup.get_asim_additional_args(settings, vols, True)

    assert (
        "/activitysim/activitysim/examples/prototype_mtc_clean/configs_mp" not in args
    )
    assert (
        "/activitysim/activitysim/examples/prototype_mtc_clean/configs_sh_compile"
        in args
    )
    assert "/activitysim/activitysim/examples/prototype_mtc_clean/configs" in args


def test_activitysim_numba_warmup_args_keep_one_compile_output_destination():
    settings = _settings()
    working_dir = "/tmp/pilates-workdir"
    vols = ActivitysimRunner.get_asim_docker_vols(settings, working_dir=working_dir)
    vols["/tmp/private-compile-cache"] = {
        "bind": "/activitysim/activitysim/examples/prototype_mtc_clean/output/cache",
        "mode": "rw",
    }

    args = ActivitysimNumbaWarmup.get_asim_additional_args(settings, vols, True)

    output_root = "/activitysim/activitysim/examples/prototype_mtc_clean/output"
    assert args.count("-o") == 1
    assert args[args.index("-o") + 1] == output_root


def test_activitysim_run_args_include_configs_mp():
    settings = _settings()
    working_dir = "/tmp/pilates-workdir"
    vols = ActivitysimRunner.get_asim_docker_vols(settings, working_dir=working_dir)

    args = ActivitysimRunner.get_asim_additional_args(settings, vols, False)

    main_idx = args.index(
        "/activitysim/activitysim/examples/prototype_mtc_clean/configs"
    )
    mp_idx = args.index(
        "/activitysim/activitysim/examples/prototype_mtc_clean/configs_mp"
    )

    assert main_idx < mp_idx


def test_activitysim_container_environment_blocks_user_site():
    env = _asim_container_environment()

    assert env["NUMBA_CACHE_DIR"] == "/app/numba_cache/numba"
    assert env["XDG_CACHE_HOME"] == "/app/numba_cache"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_activitysim_container_environment_passes_zarr_debug_flags(monkeypatch):
    monkeypatch.setenv("ASIM_DEBUG_ZARR_WRITE", "1")
    monkeypatch.setenv("ASIM_DEBUG_ZARR_PROBE", "1")
    monkeypatch.setenv("ASIM_DEBUG_ZARR_PROBE_ONLY", "1")
    monkeypatch.setenv("ASIM_DEBUG_ZARR_PROBE_DIR", "/tmp/asim-zarr-probe")
    monkeypatch.setenv("ASIM_DEBUG_ZARR_PROBE_LIMIT", "3")

    env = _asim_container_environment()

    assert env["ASIM_DEBUG_ZARR_WRITE"] == "1"
    assert env["ASIM_DEBUG_ZARR_PROBE"] == "1"
    assert env["ASIM_DEBUG_ZARR_PROBE_ONLY"] == "1"
    assert env["ASIM_DEBUG_ZARR_PROBE_DIR"] == "/tmp/asim-zarr-probe"
    assert env["ASIM_DEBUG_ZARR_PROBE_LIMIT"] == "3"
