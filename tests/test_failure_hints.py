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


def test_format_promotion_command_uses_merge_db_only_when_no_roots():
    settings = SimpleNamespace(
        settings_file="scenarios/settings-seattle.yaml",
        run=SimpleNamespace(recovery_archive_roots=[]),
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
        == "python -m pilates.runtime.promote_run_archive -c scenarios/settings-seattle.yaml --run-dir /global/scratch/run --merge-main-db /clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb --merge-db-only"
    )


def test_format_promotion_nohup_command_wraps_output_log():
    settings = SimpleNamespace(
        settings_file="scenarios/settings-seattle.yaml",
        run=SimpleNamespace(
            recovery_archive_roots=["/clusterfs/beem-core-data-nfs/pilates-outputs"]
        ),
        shared=SimpleNamespace(
            database=SimpleNamespace(
                path="/clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb"
            )
        ),
    )

    command = failure_hints.format_promotion_nohup_command(
        settings=settings,
        archive_run_dir="/global/scratch/run",
    )

    assert (
        command
        == "nohup python -m pilates.runtime.promote_run_archive -c scenarios/settings-seattle.yaml --run-dir /global/scratch/run --root /clusterfs/beem-core-data-nfs/pilates-outputs --merge-main-db /clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb > /global/scratch/run/.workflow/diagnostics/promotion_to_nfs.log 2>&1 < /dev/null &"
    )


def test_format_consist_shell_command_prefers_main_db():
    command = failure_hints.format_consist_shell_command(
        archive_run_dir="/global/scratch/run",
        run_db_path="/global/scratch/run/.consist/snapshots/latest/provenance.duckdb",
        main_db_path="/clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb",
    )

    assert (
        command
        == "consist shell --db-path /clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb --run-dir /global/scratch/run --trust-db"
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
    assert "Detached nohup command:" not in caplog.text


def test_log_promotion_instructions_on_success(caplog):
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
        "archive_run_dir": "/tmp/run",
    }

    with caplog.at_level("INFO"):
        failure_hints.log_promotion_instructions_on_success(
            logger=failure_hints.logger,
            context=context,
        )

    assert "Run promotion command for NFS archive:" in caplog.text
    assert (
        "python -m pilates.runtime.promote_run_archive -c settings.yaml --run-dir /tmp/run --root /clusterfs/beem-core-data-nfs/pilates-outputs --merge-main-db /clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb"
        in caplog.text
    )
    assert "Detached nohup command:" in caplog.text
    assert (
        "nohup python -m pilates.runtime.promote_run_archive -c settings.yaml --run-dir /tmp/run --root /clusterfs/beem-core-data-nfs/pilates-outputs --merge-main-db /clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb > /tmp/run/.workflow/diagnostics/promotion_to_nfs.log 2>&1 < /dev/null &"
        in caplog.text
    )


def test_log_consist_cli_instructions(caplog):
    with caplog.at_level("INFO"):
        failure_hints.log_consist_cli_instructions(
            logger=failure_hints.logger,
            archive_run_dir="/global/scratch/run",
            run_db_path="/global/scratch/run/.consist/provenance.duckdb",
            main_db_path="/clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb",
            success=True,
        )

    assert "Consist shell command:" in caplog.text
    assert (
        "consist shell --db-path /clusterfs/beem-core-data-nfs/pilates-main/provenance.duckdb --run-dir /global/scratch/run --trust-db"
        in caplog.text
    )
