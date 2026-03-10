import os
from types import SimpleNamespace

from pilates.activitysim.runner import ActivitysimCompileRunner, ActivitysimRunner


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
            region_mappings={"region_to_subdir": {"sfbay": "activitysim/examples/prototype_mtc_clean"}},
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


def test_activitysim_compile_args_skip_configs_mp():
    settings = _settings()
    working_dir = "/tmp/pilates-workdir"
    vols = ActivitysimRunner.get_asim_docker_vols(settings, working_dir=working_dir)

    args = ActivitysimCompileRunner.get_asim_additional_args(
        settings, vols, True
    )

    assert "/activitysim/activitysim/examples/prototype_mtc_clean/configs_mp" not in args
    assert "/activitysim/activitysim/examples/prototype_mtc_clean/configs_sh_compile" in args
    assert "/activitysim/activitysim/examples/prototype_mtc_clean/configs" in args


def test_activitysim_run_args_include_configs_mp():
    settings = _settings()
    working_dir = "/tmp/pilates-workdir"
    vols = ActivitysimRunner.get_asim_docker_vols(settings, working_dir=working_dir)

    args = ActivitysimRunner.get_asim_additional_args(settings, vols, False)

    assert "/activitysim/activitysim/examples/prototype_mtc_clean/configs_mp" in args
