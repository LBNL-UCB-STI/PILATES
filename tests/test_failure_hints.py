from types import SimpleNamespace

from pilates.runtime import failure_hints


def test_format_restart_command_uses_config_and_archive_state():
    settings = SimpleNamespace(settings_file="scenarios/settings-seattle.yaml")

    command = failure_hints.format_restart_command(
        settings=settings,
        archive_state_path="/tmp/pilates run/run_state.yaml",
    )

    assert (
        command
        == "python run.py -c scenarios/settings-seattle.yaml -S '/tmp/pilates run/run_state.yaml'"
    )


def test_format_hpc_restart_command_requires_account_placeholder():
    settings = SimpleNamespace(settings_file="scenarios/settings-seattle.yaml")

    command = failure_hints.format_hpc_restart_command(
        settings=settings,
        archive_state_path="/tmp/pilates run/run_state.yaml",
    )

    assert (
        command
        == "./hpc/job_runner.sh -c scenarios/settings-seattle.yaml -a '<slurm_account>' -s '/tmp/pilates run/run_state.yaml'"
    )


def test_format_promotion_command_uses_recovery_root_and_shared_db():
    settings = SimpleNamespace(
        settings_file="scenarios/settings-seattle.yaml",
        run=SimpleNamespace(
            recovery_archive_roots=[
                "/clusterfs/beem-core-data-nfs/pilates-outputs",
                "/clusterfs/secondary/pilates-outputs",
            ]
        ),
        shared=SimpleNamespace(
            database=SimpleNamespace(
                path="/clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb"
            )
        ),
    )

    command = failure_hints.format_promotion_command(
        settings=settings,
        archive_run_dir="/global/scratch/run",
    )

    assert (
        command
        == "python -m pilates.runtime.promote_run_archive -c scenarios/settings-seattle.yaml --run-dir /global/scratch/run --root /clusterfs/beem-core-data-nfs/pilates-outputs --root /clusterfs/secondary/pilates-outputs --merge-main-db /clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb"
    )


def test_log_restart_instructions_uses_context_state_path(caplog):
    context = {
        "settings": SimpleNamespace(settings_file="settings.yaml"),
        "state": SimpleNamespace(run_info_path="/tmp/run/run_state.yaml"),
        "archive_run_dir": "/tmp/run",
        "local_run_dir": "/local/run",
    }

    failure_hints.log_restart_instructions_on_failure(
        logger=failure_hints.logger,
        context=context,
    )

    assert "Run failed. Restart command:" in caplog.text
    assert "python run.py -c settings.yaml -S /tmp/run/run_state.yaml" in caplog.text
    assert (
        "./hpc/job_runner.sh -c settings.yaml -a '<slurm_account>' -s /tmp/run/run_state.yaml"
        in caplog.text
    )
    assert "archive run dir: /tmp/run" in caplog.text
    assert "local run dir: /local/run" in caplog.text


def test_log_restart_instructions_includes_promotion_command_when_available(caplog):
    context = {
        "settings": SimpleNamespace(
            settings_file="settings.yaml",
            run=SimpleNamespace(
                recovery_archive_roots=["/clusterfs/beem-core-data-nfs/pilates-outputs"]
            ),
            shared=SimpleNamespace(
                database=SimpleNamespace(
                    path="/clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb"
                )
            ),
        ),
        "state": SimpleNamespace(run_info_path="/tmp/run/run_state.yaml"),
        "archive_run_dir": "/tmp/run",
        "local_run_dir": "/local/run",
    }

    failure_hints.log_restart_instructions_on_failure(
        logger=failure_hints.logger,
        context=context,
    )

    assert "Run promotion command for NFS archive:" in caplog.text
    assert (
        "python -m pilates.runtime.promote_run_archive -c settings.yaml --run-dir /tmp/run --root /clusterfs/beem-core-data-nfs/pilates-outputs --merge-main-db /clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb"
        in caplog.text
    )
