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


def _settings_recovery_roots(settings: Optional[Any]) -> list[str]:
    if settings is None:
        return []
    try:
        roots = settings.run.recovery_archive_roots
    except AttributeError:
        return []
    if not roots:
        return []
    return [str(root) for root in roots if root]


def _settings_shared_db_path(settings: Optional[Any]) -> Optional[str]:
    if settings is None:
        return None
    try:
        db_path = settings.shared.database.path
    except AttributeError:
        return None
    if not db_path:
        return None
    return str(db_path)


def _promotion_log_path(archive_run_dir: str) -> str:
    return str((f"{archive_run_dir}/.workflow/diagnostics/promotion_to_nfs.log"))


def format_consist_shell_command(
    *,
    archive_run_dir: Optional[str],
    run_db_path: Optional[str],
    main_db_path: Optional[str],
) -> Optional[str]:
    if not run_db_path and not main_db_path:
        return None

    db_path = main_db_path or run_db_path
    if db_path is None:
        return None

    command = [
        "consist",
        "shell",
        "--db-path",
        str(db_path),
    ]
    if archive_run_dir:
        command.extend(["--run-dir", str(archive_run_dir)])
    command.append("--trust-db")
    return " ".join(shlex.quote(part) for part in command)


def format_promotion_command(
    *,
    settings: Optional[Any],
    archive_run_dir: Optional[str],
) -> Optional[str]:
    config_path = None
    if settings is not None:
        config_path = settings.settings_file
    if not config_path or not archive_run_dir:
        return None

    command = [
        "python",
        "-m",
        "pilates.runtime.promote_run_archive",
        "-c",
        str(config_path),
        "--run-dir",
        str(archive_run_dir),
    ]
    recovery_roots = _settings_recovery_roots(settings)
    for root in recovery_roots:
        command.extend(["--root", root])
    shared_db_path = _settings_shared_db_path(settings)
    if shared_db_path:
        command.extend(["--merge-main-db", shared_db_path])
        if not recovery_roots:
            command.append("--merge-db-only")
    return " ".join(shlex.quote(part) for part in command)


def format_promotion_nohup_command(
    *,
    settings: Optional[Any],
    archive_run_dir: Optional[str],
) -> Optional[str]:
    promotion_command = format_promotion_command(
        settings=settings,
        archive_run_dir=archive_run_dir,
    )
    if promotion_command is None or not archive_run_dir:
        return None

    log_path = _promotion_log_path(str(archive_run_dir))
    return f"nohup {promotion_command} > {shlex.quote(log_path)} 2>&1 < /dev/null &"


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

    promotion_command = format_promotion_command(
        settings=settings,
        archive_run_dir=archive_run_dir,
    )
    if promotion_command is not None:
        logger.error("Run promotion command for NFS archive:")
        logger.error("  %s", promotion_command)


def log_promotion_instructions_on_success(
    *,
    logger: logging.Logger,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    context = RUN_FAILURE_CONTEXT if context is None else context
    settings = context.get("settings")
    archive_run_dir = context.get("archive_run_dir")

    promotion_command = format_promotion_command(
        settings=settings,
        archive_run_dir=archive_run_dir,
    )
    if promotion_command is None:
        return

    logger.info("Run promotion command for NFS archive:")
    logger.info("  %s", promotion_command)

    nohup_command = format_promotion_nohup_command(
        settings=settings,
        archive_run_dir=archive_run_dir,
    )
    if nohup_command is not None:
        logger.info("Detached nohup command:")
        logger.info("  %s", nohup_command)


def log_consist_cli_instructions(
    *,
    logger: logging.Logger,
    archive_run_dir: Optional[str],
    run_db_path: Optional[str],
    main_db_path: Optional[str],
    success: bool,
) -> None:
    command = format_consist_shell_command(
        archive_run_dir=archive_run_dir,
        run_db_path=run_db_path,
        main_db_path=main_db_path,
    )
    if command is None:
        return

    log_fn = logger.info if success else logger.error
    log_fn("Consist shell command:")
    log_fn("  %s", command)
