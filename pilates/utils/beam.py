import os
from pathlib import Path
from pilates.utils.path_utils import find_project_root
from pilates.utils.settings_helper import get as get_setting


def get_beam_source_dir(settings):
    """
    Robustly resolve the BEAM source directory, matching logic in pilates/beam/preprocessor.py.
    Returns the absolute path to the BEAM source directory for the given region.
    """
    region = get_setting(settings, "run.region")
    # Find the project root by searching upwards for 'pilates' or '.git'
    pilates_root = find_project_root()
    if pilates_root is None:
        pilates_root = os.path.realpath(os.getcwd())
    beam_source_dir = os.path.abspath(
        os.path.join(
            pilates_root,
            "pilates",
            "beam",
            "production",
            region,
        )
    )
    # If not found, try with "sources/PILATES" in the path (for symlinked or alternate layouts)
    if not os.path.exists(beam_source_dir):
        alt_root = os.path.join(pilates_root, "sources", "PILATES")
        alt_beam_source_dir = os.path.abspath(
            os.path.join(
                alt_root,
                "pilates",
                "beam",
                "production",
                region,
            )
        )
        if os.path.exists(alt_beam_source_dir):
            beam_source_dir = alt_beam_source_dir
    return beam_source_dir


def get_beam_omx_skims_name(settings, default: str = "skims.omx") -> str:
    """
    Resolve the downstream OMX skim filename for BEAM outputs.

    ``shared.skims.fname`` may point at a canonical Zarr skim store. Downstream
    OMX consumers still need a sibling ``.omx`` export, so normalize any
    non-OMX filename to the same basename with an ``.omx`` suffix.
    """
    configured = get_setting(settings, "shared.skims.fname", default)
    if not isinstance(configured, str) or not configured:
        return default
    if configured.endswith(".omx"):
        return configured
    return str(Path(configured).with_suffix(".omx"))
