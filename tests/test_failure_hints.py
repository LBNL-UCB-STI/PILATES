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
