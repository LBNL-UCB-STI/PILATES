import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_initial_linkstats_path(settings, workspace) -> Optional[str]:
    """
    Resolve the initial BEAM warm-start linkstats path from settings.

    Initial warmstart is opt-in. When ``beam.warmstart_linkstats_path`` is unset,
    no router-directory probing occurs. Relative paths are resolved against the
    mutable BEAM region input root, and ``{router_directory}`` is expanded from
    ``beam.router_directory`` when present.
    """
    beam_settings = getattr(settings, "beam", None)
    run_settings = getattr(settings, "run", None)
    if beam_settings is None or run_settings is None:
        return None

    region = getattr(run_settings, "region", None)
    if not region:
        return None

    configured_path = getattr(beam_settings, "warmstart_linkstats_path", None)
    if not configured_path:
        return None

    beam_region_root = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        region,
    )

    candidate = os.path.expanduser(os.fspath(configured_path))
    router_directory = getattr(beam_settings, "router_directory", None)
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
        try:
            from pilates.beam.preprocessor import BeamPreprocessor

            static_region_root = BeamPreprocessor._find_beam_production_path(
                settings, str(region)
            )
            static_candidate = os.path.normpath(
                os.path.join(static_region_root, rel_candidate)
            )
            if os.path.exists(static_candidate):
                return static_candidate
        except Exception:
            pass

    logger.warning(
        "[BEAM warmstart] Configured warmstart_linkstats_path not found: %s",
        candidate,
    )
    return None
