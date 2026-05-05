from __future__ import annotations

import logging
import shlex
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)
RUN_FAILURE_CONTEXT: Dict[str, Any] = {}


def clear_run_failure_context() -> None:
    RUN_FAILURE_CONTEXT.clear()


def set_run_failure_context(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        if value is None:
            continue
        RUN_FAILURE_CONTEXT[key] = value


def format_restart_command(
    *,
    settings: Optional[Any],
    archive_state_path: Optional[str],
) -> Optional[str]:
    config_path = None
    if settings is not None:
        config_path = settings.settings_file
    if not config_path and not archive_state_path:
        return None

    command = ["python", "run.py"]
    if config_path:
        command.extend(["-c", str(config_path)])
    if archive_state_path:
        command.extend(["-S", str(archive_state_path)])
    return " ".join(shlex.quote(part) for part in command)


def format_hpc_restart_command(
    *,
    settings: Optional[Any],
    archive_state_path: Optional[str],
) -> Optional[str]:
    config_path = None
    if settings is not None:
        config_path = settings.settings_file
    if not config_path and not archive_state_path:
        return None

    command = ["./hpc/job_runner.sh"]
    if config_path:
        command.extend(["-c", str(config_path)])
    command.extend(["-a", "<slurm_account>"])
    if archive_state_path:
        command.extend(["-s", str(archive_state_path)])
    return " ".join(shlex.quote(part) for part in command)


def log_restart_instructions_on_failure(
    *,
    logger: logging.Logger,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    context = RUN_FAILURE_CONTEXT if context is None else context
    settings = context.get("settings")
    state = context.get("state")
    archive_run_dir = context.get("archive_run_dir")
    local_run_dir = context.get("local_run_dir")
    archive_state_path = context.get("archive_state_path")
    if archive_state_path is None and state is not None:
        archive_state_path = state.run_info_path

    command = format_restart_command(
        settings=settings,
        archive_state_path=archive_state_path,
    )
    if command is None:
        return

    logger.error("Run failed. Restart command:")
    logger.error("  %s", command)
    if archive_run_dir:
        command_hpc = format_hpc_restart_command(
            settings=settings,
            archive_state_path=archive_state_path,
        )
        logger.error("  HPC command: %s", command_hpc)
    if archive_state_path:
        logger.error("  state file: %s", archive_state_path)
    if archive_run_dir:
        logger.error("  archive run dir: %s", archive_run_dir)
    if local_run_dir:
        logger.error("  local run dir: %s", local_run_dir)
