import logging
import os
from pathlib import Path
from typing import Optional

from pilates.beam.config_hocon import (
    BeamConfigHoconError,
    beam_config_env_overrides,
    resolve_beam_config_value,
)
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


def _find_static_region_root(settings, region: str) -> Path | None:
    from pilates.beam.preprocessor import BeamPreprocessor

    try:
        return Path(BeamPreprocessor._find_beam_production_path(settings, region))
    except (AttributeError, ValueError, OSError):
        return None


def resolve_initial_linkstats_path(settings, workspace) -> Optional[str]:
    """
    Resolve the initial BEAM warm-start linkstats path from settings.

    Initial warmstart requires a configured path and respects an explicitly
    disabled BEAM warm start. Relative paths are resolved against the mutable
    BEAM region input root, and ``{router_directory}`` is expanded from
    ``beam.router_directory`` when present.
    """
    from pilates.workspace import is_dry_run_workspace

    if is_dry_run_workspace(workspace):
        return None

    beam_settings = get_setting(settings, "beam")
    run_settings = get_setting(settings, "run")
    if beam_settings is None or run_settings is None:
        return None

    region = get_setting(run_settings, "region")
    if not region:
        return None

    configured_path = get_setting(beam_settings, "warmstart_linkstats_path")
    if not configured_path:
        return None

    beam_region_root = Path(workspace.get_beam_mutable_data_dir()) / region
    static_root = None
    config_name = get_setting(beam_settings, "config")
    if config_name:
        config_root = beam_region_root
        if not (config_root / config_name).is_file():
            static_root = _find_static_region_root(settings, str(region))
            config_root = static_root or config_root
        config_path = config_root / config_name
        if config_path.is_file():
            try:
                warmstart_type = resolve_beam_config_value(
                    config_path,
                    key="beam.warmStart.type",
                    env_overrides=beam_config_env_overrides(
                        settings, config_root=config_root
                    ),
                )
            except BeamConfigHoconError:
                # Launch validation owns malformed HOCON.
                warmstart_type = None
            if isinstance(warmstart_type, str) and warmstart_type.lower() == "disabled":
                return None

    candidate = os.path.expanduser(os.fspath(configured_path))
    router_directory = get_setting(beam_settings, "router_directory")
    if "{router_directory}" in candidate:
        if not router_directory:
            logger.warning(
                "[BEAM warmstart] warmstart_linkstats_path uses {router_directory} "
                "but beam.router_directory is not configured."
            )
            return None
        candidate = candidate.replace("{router_directory}", router_directory)

    rel_candidate = candidate
    if not os.path.isabs(candidate):
        candidate = os.path.join(beam_region_root, candidate)
    candidate = os.path.normpath(candidate)

    if os.path.exists(candidate):
        return candidate

    if not os.path.isabs(rel_candidate):
        static_root = static_root or _find_static_region_root(settings, str(region))
        if static_root is not None:
            static_candidate = os.path.normpath(static_root / rel_candidate)
            if os.path.exists(static_candidate):
                return static_candidate

    logger.warning(
        "[BEAM warmstart] Configured warmstart_linkstats_path not found: %s",
        candidate,
    )
    return None
