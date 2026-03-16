"""Thin CLI entrypoint for PILATES runtime orchestration."""

from pilates.runtime import launcher as launcher_runtime


def main() -> None:
    launcher_runtime.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        launcher_runtime._log_restart_instructions_on_failure()
        raise
